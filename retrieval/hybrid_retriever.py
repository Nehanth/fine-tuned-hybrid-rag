"""
Hybrid Retriever for RAG pipeline.
Combines dense semantic search, sparse TF-IDF search, and metadata boosting.
"""

import json
import numpy as np
import joblib
import torch
import yaml
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from finetune.train_weights import WeightLearner

from .metadata_boosting import batch_compute_boost, filter_documents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_retrieval_components(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load all components needed for hybrid retrieval using config.yaml.
    Always loads learned weights if available.
    
    Args:
        config (dict, optional): Configuration dictionary. If None, loads from config.yaml
        
    Returns:
        Dict: Dictionary containing all loaded components
    """
    if config is None:
        config = load_config()
    
    print("Loading hybrid retrieval components...")
    
    # Get paths and settings from config
    dense_embeddings_path = config["embeddings"]["dense_path"]
    tfidf_vectorizer_path = config["embeddings"]["tfidf_vectorizer_path"]
    documents_path = config["dataset"]["processed_path"]
    model_name = config["model"]["encoder"]
    device = config["model"]["device"]
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dense embeddings
    print(f"Loading dense embeddings from {dense_embeddings_path}...")
    dense_embeddings = np.load(dense_embeddings_path)
    print(f"Loaded dense embeddings shape: {dense_embeddings.shape}")
    
    # Load TF-IDF vectorizer
    print(f"Loading TF-IDF vectorizer from {tfidf_vectorizer_path}...")
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    
    # Load documents
    print(f"Loading documents from {documents_path}...")
    documents = []
    with open(documents_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            documents.append(json.loads(line.strip()))
    print(f"Loaded {len(documents)} documents")
    
    # Initialize sentence transformer for query encoding
    print(f"Loading sentence transformer model: {model_name}")
    sentence_model = SentenceTransformer(model_name, device=device)
    print(f"Using device: {device}")
    
    components = {
        "dense_embeddings": dense_embeddings,
        "tfidf_vectorizer": tfidf_vectorizer,
        "documents": documents,
        "sentence_model": sentence_model,
        "device": device,
        "model_name": model_name,
        "config": config
    }
    
    # Load required learned weights
    learned_weights_path = "finetune/learned_weights.pth"
    if os.path.exists(learned_weights_path):
        print("Loading learned combination weights...")
        weight_learner = WeightLearner()
        weight_learner.load_state_dict(torch.load(learned_weights_path, map_location='cpu'))
        weight_learner.eval()
        components["weight_learner"] = weight_learner
        
        # Show what weights were learned
        learned_weights = weight_learner.get_weights()
        print(f"Learned weights loaded: dense={learned_weights[0]:.3f}, sparse={learned_weights[1]:.3f}, boost={learned_weights[2]:.3f}")
    else:
        print(f"Learned weights not found at {learned_weights_path}")
        print("   Run 'python finetune/train_weights.py' first to train weights")
        raise FileNotFoundError(f"Learned weights required but not found at {learned_weights_path}")
    
    print("All components loaded successfully!")
    return components


def hybrid_retrieve(query: str, 
                   components: Dict[str, Any],
                   user_filters: Optional[Dict[str, Any]] = None,
                   top_k: Optional[int] = None,
                   candidate_pool_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents for a given query using hybrid retrieval
    
    Args:
        query: Query text
        components: Dictionary of loaded retrieval components (include weight_learner)
        user_filters: User-specified filters and preferences
        top_k: Number of top documents to return (uses config default if None)
        candidate_pool_size: Size of candidate pool for dense retrieval (uses config default if None)
        
    Returns:
        List: List of retrieved documents with scores and metadata
    """
    if user_filters is None:
        user_filters = {}
    
    # Get defaults from config if not provided
    config = components["config"]
    if top_k is None:
        top_k = config["retrieval"]["top_k"]
    if candidate_pool_size is None:
        candidate_pool_size = config["retrieval"]["candidate_pool_size"]
    
    # Ensure we have learned weights
    if "weight_learner" not in components:
        raise ValueError("weight_learner not found in components. System requires learned weights!")
    
    print(f"Retrieving top-{top_k} documents for query: '{query[:50]}...'")
    
    # Step 1: Encode query
    print("Encoding query...")
    query_dense = components["sentence_model"].encode([query], convert_to_numpy=True)[0]
    query_sparse = components["tfidf_vectorizer"].transform([query]).toarray()[0]
    
    # Step 2: Compute dense similarities and get candidate pool
    print(f"Computing dense similarities (top-{candidate_pool_size})...")
    query_embedding = query_dense.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, components["dense_embeddings"])[0]
    top_indices = np.argsort(similarities)[::-1][:candidate_pool_size]
    dense_similarities = similarities[top_indices]
    candidate_indices = top_indices
    
    # Step 3: Apply hard filters if specified
    if user_filters and any(filter_key in user_filters for filter_key in 
           ['min_year', 'max_year', 'required_fields', 'excluded_venues']):
        print("Applying hard filters...")
        candidate_docs = [components["documents"][i] for i in candidate_indices]
        filter_mask = filter_documents(candidate_docs, user_filters)
        
        # Filter candidates
        candidate_indices = candidate_indices[filter_mask]
        dense_similarities = dense_similarities[filter_mask]
        
        print(f"{len(candidate_indices)} candidates after filtering")
    
    # Handle case where filtering eliminates all candidates
    if len(candidate_indices) == 0:
        print("Warning: All candidates filtered out. Returning empty results.")
        return []
    
    # Step 4: Compute sparse similarities for candidates
    print("Computing sparse similarities...")
    candidate_texts = [components["documents"][i]["text"] for i in candidate_indices]
    candidate_vectors = components["tfidf_vectorizer"].transform(candidate_texts).toarray()
    query_vector = query_sparse.reshape(1, -1)
    sparse_similarities = cosine_similarity(query_vector, candidate_vectors)[0]
    
    # Step 5: Apply metadata boosting
    print("Applying metadata boosting...")
    candidate_metadata = [components["documents"][i]["metadata"] for i in candidate_indices]
    boost_scores_raw = np.array(batch_compute_boost(candidate_metadata, user_filters))
    # Normalize boost scores to match training (1.0-1.5 → 0.0-1.0)
    boost_scores = (boost_scores_raw - 1.0) / 0.5
    
    # Step 6: Combine scores using learned weights
    print("Combining scores with learned weights...")
    
    dense_tensor = torch.tensor(dense_similarities, dtype=torch.float32)
    sparse_tensor = torch.tensor(sparse_similarities, dtype=torch.float32)
    boost_tensor = torch.tensor(boost_scores, dtype=torch.float32)
    
    with torch.no_grad():
        final_scores = components["weight_learner"](dense_tensor, sparse_tensor, boost_tensor).numpy()
    
    # Show what weights were used
    learned_weights = components["weight_learner"].get_weights()
    print(f"   Learned weights: dense={learned_weights[0]:.3f}, sparse={learned_weights[1]:.3f}, boost={learned_weights[2]:.3f}")
    
    # Step 7: Get top-k results
    top_k_indices = np.argsort(final_scores)[::-1][:top_k]
    
    # Step 8: Format results
    results = []
    for i, idx in enumerate(top_k_indices):
        doc_idx = candidate_indices[idx]
        doc = components["documents"][doc_idx].copy()
        
        # Add scores to result
        doc["scores"] = {
            "final_score": float(final_scores[idx]),
            "dense_score": float(dense_similarities[idx]),
            "sparse_score": float(sparse_similarities[idx]),
            "boost_score": float(boost_scores_raw[idx])     
        }
        doc["rank"] = i + 1
        doc["doc_id"] = doc_idx
        
        results.append(doc)
    
    print(f"Retrieved {len(results)} documents")
    return results

