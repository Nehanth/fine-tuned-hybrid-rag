"""
Scorer module for hybrid retrieval system.
Combines dense, sparse, and metadata boost scores into a final ranking score.
"""

def hybrid_score(dense_score: float, tfidf_score: float, boost_score: float, 
                 weights: tuple = (0.6, 0.3, 0.1)) -> float:
    """
    Combine dense, sparse, and boost scores into a final hybrid score.
    
    Args:
        dense_score (float): Cosine similarity score from dense embeddings [0, 1]
        tfidf_score (float): Cosine similarity score from TF-IDF sparse vectors [0, 1]
        boost_score (float): Metadata boost multiplier (typically around 1.0)
        weights (tuple): Weights for (dense, sparse, boost) components. Default: (0.6, 0.3, 0.1)
    
    Returns:
        float: Final combined score for ranking
    
    Example:
        >>> score = hybrid_score(0.8, 0.6, 1.2, weights=(0.6, 0.3, 0.1))
        >>> print(f"Final score: {score:.4f}")
    """
    # Validate weights sum to 1.0
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
    
    # Validate input ranges
    if not (0 <= dense_score <= 1):
        raise ValueError(f"Dense score must be in [0, 1], got {dense_score}")
    if not (0 <= tfidf_score <= 1):
        raise ValueError(f"TF-IDF score must be in [0, 1], got {tfidf_score}")
    if boost_score < 0:
        raise ValueError(f"Boost score must be non-negative, got {boost_score}")
    
    dense_weight, sparse_weight, boost_weight = weights
    
    # Compute weighted combination
    # Dense and sparse scores are weighted and summed
    # Boost score is applied as both an additive component and multiplicative factor
    base_score = (dense_weight * dense_score + sparse_weight * tfidf_score)
    
    # Combine scores using weighted additive approach
    final_score = base_score + boost_weight * boost_score
    
    return final_score


def batch_hybrid_score(dense_scores: list, tfidf_scores: list, boost_scores: list,
                       weights: tuple = (0.6, 0.3, 0.1)) -> list:
    """
    Compute hybrid scores for multiple documents efficiently.
    
    Args:
        dense_scores (list): List of dense similarity scores
        tfidf_scores (list): List of TF-IDF similarity scores  
        boost_scores (list): List of metadata boost scores
        weights (tuple): Weights for score combination
    
    Returns:
        list: List of final hybrid scores
    """
    if not (len(dense_scores) == len(tfidf_scores) == len(boost_scores)):
        raise ValueError("All score lists must have the same length")
    
    return [
        hybrid_score(dense, tfidf, boost, weights)
        for dense, tfidf, boost in zip(dense_scores, tfidf_scores, boost_scores)
    ]