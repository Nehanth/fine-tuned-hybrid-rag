"""
Retrieval package for hybrid RAG pipeline.
Combines dense semantic search, sparse TF-IDF, and metadata boosting.
"""

from .hybrid_retriever import (
    load_retrieval_components,
    hybrid_retrieve,
    get_retrieval_stats,
    get_document_by_id,
    encode_query_dense,
    encode_query_sparse,
    compute_dense_similarities,
    compute_sparse_similarities,
    apply_metadata_boosting,
    combine_scores
)
from .metadata_boosting import compute_boost, batch_compute_boost, filter_documents
from .scorer import hybrid_score, batch_hybrid_score

__all__ = [
    "load_retrieval_components",
    "hybrid_retrieve",
    "get_retrieval_stats",
    "get_document_by_id",
    "encode_query_dense",
    "encode_query_sparse", 
    "compute_dense_similarities",
    "compute_sparse_similarities",
    "apply_metadata_boosting",
    "combine_scores",
    "compute_boost",
    "batch_compute_boost", 
    "filter_documents",
    "hybrid_score",
    "batch_hybrid_score"
]

__version__ = "0.1.0" 