"""
Hybrid Retriever for RAG pipeline.
Combines dense semantic search, sparse TF-IDF search, and metadata boosting.
"""

import json
import numpy as np
import joblib
import torch
import yaml
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .metadata_boosting import compute_boost, batch_compute_boost, filter_documents


def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_retrieval_components(config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load all components needed for hybrid retrieval using config.yaml.
    
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
    
    print("All components loaded successfully!")
    
    return {
        "dense_embeddings": dense_embeddings,
        "tfidf_vectorizer": tfidf_vectorizer,
        "documents": documents,
        "sentence_model": sentence_model,
        "device": device,
        "model_name": model_name,
        "config": config
    }


def hybrid_retrieve(query: str, 
                   components: Dict[str, Any],
                   user_filters: Optional[Dict[str, Any]] = None,
                   top_k: Optional[int] = None,
                   candidate_pool_size: Optional[int] = None,
                   weights: Optional[Tuple[float, float, float]] = None) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents for a given query using hybrid retrieval.
    
    Args:
        query: Query text
        components: Dictionary of loaded retrieval components
        user_filters: User-specified filters and preferences
        top_k: Number of top documents to return (uses config default if None)
        candidate_pool_size: Size of candidate pool for dense retrieval (uses config default if None)
        weights (tuple): Weights for (dense, sparse, boost) score combination (uses config default if None)
        
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
    if weights is None:
        weights = (
            config["scoring"]["dense_weight"],
            config["scoring"]["sparse_weight"], 
            config["scoring"]["boost_weight"]
        )
    
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
    
    # Step 4: Compute sparse similarities for candidates
    print("Computing sparse similarities...")
    candidate_texts = [components["documents"][i]["text"] for i in candidate_indices]
    candidate_vectors = components["tfidf_vectorizer"].transform(candidate_texts).toarray()
    query_vector = query_sparse.reshape(1, -1)
    sparse_similarities = cosine_similarity(query_vector, candidate_vectors)[0]
    
    # Step 5: Apply metadata boosting
    print("Applying metadata boosting...")
    candidate_metadata = [components["documents"][i]["metadata"] for i in candidate_indices]
    boost_scores = np.array(batch_compute_boost(candidate_metadata, user_filters))
    
    # Step 6: Combine scores (inline scoring logic)
    print("Combining scores...")
    dense_weight, sparse_weight, boost_weight = weights
    final_scores = []
    for dense, sparse, boost in zip(dense_similarities, sparse_similarities, boost_scores):
        base_score = (dense_weight * dense + sparse_weight * sparse)
        final_score = base_score + boost_weight * boost
        final_scores.append(final_score)
    final_scores = np.array(final_scores)
    
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
            "boost_score": float(boost_scores[idx])
        }
        doc["rank"] = i + 1
        doc["doc_id"] = doc_idx
        
        results.append(doc)
    
    print(f"Retrieved {len(results)} documents")
    return results


def get_document_by_id(doc_id: int, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get a specific document by its index.
    
    Args:
        doc_id (int): Document index
        documents (List): List of document dictionaries
        
    Returns:
        Dict: Document with metadata
    """
    if 0 <= doc_id < len(documents):
        return documents[doc_id]
    else:
        raise ValueError(f"Document ID {doc_id} out of range [0, {len(documents)})")


def get_retrieval_stats(components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get retrieval system statistics.
    
    Args:
        components (Dict): Dictionary of loaded retrieval components
        
    Returns:
        Dict: Statistics about the loaded components
    """
    return {
        "num_documents": len(components["documents"]),
        "dense_embedding_shape": components["dense_embeddings"].shape,
        "tfidf_vocab_size": len(components["tfidf_vectorizer"].vocabulary_),
        "model_name": components["model_name"],
        "device": components["device"]
    }


