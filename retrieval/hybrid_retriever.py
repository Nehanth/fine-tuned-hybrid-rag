"""
Hybrid Retriever for RAG pipeline.
Combines dense semantic search, sparse TF-IDF search, and metadata boosting.
"""

import json
import numpy as np
import joblib
import torch
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .metadata_boosting import compute_boost, batch_compute_boost, filter_documents
from .scorer import hybrid_score, batch_hybrid_score


def load_retrieval_components(
    dense_embeddings_path: str = "embeddings/dense.npy",
    tfidf_vectorizer_path: str = "embeddings/tfidf_vectorizer.pkl",
    documents_path: str = "data/processed_docs.jsonl",
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Load all components needed for hybrid retrieval.
    
    Args:
        dense_embeddings_path (str): Path to pre-computed dense embeddings
        tfidf_vectorizer_path (str): Path to fitted TF-IDF vectorizer  
        documents_path (str): Path to processed documents JSONL file
        model_name (str): SentenceTransformer model name for query encoding
        device (str): Device for sentence transformer ("auto", "cpu", "cuda")
        
    Returns:
        Dict: Dictionary containing all loaded components
    """
    print("Loading hybrid retrieval components...")
    
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
        "model_name": model_name
    }


def encode_query_dense(query: str, sentence_model) -> np.ndarray:
    """
    Encode query using dense sentence transformer.
    
    Args:
        query (str): Query text
        sentence_model: Loaded SentenceTransformer model
        
    Returns:
        np.ndarray: Dense query embedding
    """
    return sentence_model.encode([query], convert_to_numpy=True)[0]


def encode_query_sparse(query: str, tfidf_vectorizer) -> np.ndarray:
    """
    Encode query using TF-IDF vectorizer.
    
    Args:
        query (str): Query text
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        np.ndarray: Sparse query vector
    """
    return tfidf_vectorizer.transform([query]).toarray()[0]


def compute_dense_similarities(query_embedding: np.ndarray, 
                              dense_embeddings: np.ndarray,
                              top_k: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between query and all document embeddings.
    
    Args:
        query_embedding (np.ndarray): Query embedding vector
        dense_embeddings (np.ndarray): Pre-computed document embeddings
        top_k (int): Number of top candidates to return for efficiency
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (similarities, indices) sorted by similarity
    """
    # Reshape query for cosine similarity computation
    query_embedding = query_embedding.reshape(1, -1)
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, dense_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_similarities = similarities[top_indices]
    
    return top_similarities, top_indices


def compute_sparse_similarities(query_vector: np.ndarray, 
                               candidate_indices: np.ndarray,
                               documents: List[Dict[str, Any]],
                               tfidf_vectorizer) -> np.ndarray:
    """
    Compute TF-IDF similarities for candidate documents.
    
    Args:
        query_vector (np.ndarray): Query TF-IDF vector
        candidate_indices (np.ndarray): Indices of candidate documents
        documents (List): List of document dictionaries
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        np.ndarray: TF-IDF similarities for candidate documents
    """
    # Transform candidate documents on-demand
    candidate_texts = [documents[i]["text"] for i in candidate_indices]
    candidate_vectors = tfidf_vectorizer.transform(candidate_texts).toarray()
    
    # Compute cosine similarities
    query_vector = query_vector.reshape(1, -1)
    similarities = cosine_similarity(query_vector, candidate_vectors)[0]
    
    return similarities


def apply_metadata_boosting(candidate_indices: np.ndarray, 
                           documents: List[Dict[str, Any]],
                           user_filters: Dict[str, Any]) -> np.ndarray:
    """
    Apply metadata boosting to candidate documents.
    
    Args:
        candidate_indices (np.ndarray): Indices of candidate documents
        documents (List): List of document dictionaries
        user_filters (dict): User-specified filters and preferences
        
    Returns:
        np.ndarray: Boost scores for candidate documents
    """
    candidate_metadata = [documents[i]["metadata"] for i in candidate_indices]
    boost_scores = batch_compute_boost(candidate_metadata, user_filters)
    
    return np.array(boost_scores)


def combine_scores(dense_scores: np.ndarray, 
                  sparse_scores: np.ndarray,
                  boost_scores: np.ndarray,
                  weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)) -> np.ndarray:
    """
    Combine dense, sparse, and boost scores using the scorer module.
    
    Args:
        dense_scores (np.ndarray): Dense similarity scores
        sparse_scores (np.ndarray): Sparse similarity scores  
        boost_scores (np.ndarray): Metadata boost scores
        weights (tuple): Weights for (dense, sparse, boost) components
        
    Returns:
        np.ndarray: Combined final scores
    """
    final_scores = batch_hybrid_score(
        dense_scores.tolist(), 
        sparse_scores.tolist(), 
        boost_scores.tolist(),
        weights
    )
    
    return np.array(final_scores)


def hybrid_retrieve(query: str, 
                   components: Dict[str, Any],
                   user_filters: Optional[Dict[str, Any]] = None,
                   top_k: int = 10,
                   candidate_pool_size: int = 1000,
                   weights: Tuple[float, float, float] = (0.6, 0.3, 0.1)) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents for a given query using hybrid retrieval.
    
    Args:
        query (str): Query text
        components (Dict): Dictionary of loaded retrieval components
        user_filters (dict): User-specified filters and preferences
        top_k (int): Number of top documents to return
        candidate_pool_size (int): Size of candidate pool for dense retrieval
        weights (tuple): Weights for (dense, sparse, boost) score combination
        
    Returns:
        List[Dict]: List of retrieved documents with scores and metadata
    """
    if user_filters is None:
        user_filters = {}
    
    print(f"Retrieving top-{top_k} documents for query: '{query[:50]}...'")
    
    # Step 1: Encode query
    print("Encoding query...")
    query_dense = encode_query_dense(query, components["sentence_model"])
    query_sparse = encode_query_sparse(query, components["tfidf_vectorizer"])
    
    # Step 2: Compute dense similarities and get candidate pool
    print(f"Computing dense similarities (top-{candidate_pool_size})...")
    dense_similarities, candidate_indices = compute_dense_similarities(
        query_dense, components["dense_embeddings"], top_k=candidate_pool_size
    )
    
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
    sparse_similarities = compute_sparse_similarities(
        query_sparse, candidate_indices, components["documents"], components["tfidf_vectorizer"]
    )
    
    # Step 5: Apply metadata boosting
    print("Applying metadata boosting...")
    boost_scores = apply_metadata_boosting(candidate_indices, components["documents"], user_filters or {})
    
    # Step 6: Combine scores
    print("Combining scores...")
    final_scores = combine_scores(dense_similarities, sparse_similarities, boost_scores, weights)
    
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


