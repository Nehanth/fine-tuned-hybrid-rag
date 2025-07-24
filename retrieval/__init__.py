"""
Retrieval package for hybrid RAG pipeline.
Combines dense semantic search, sparse TF-IDF, and metadata boosting.
"""

from .hybrid_retriever import (
    load_retrieval_components,
    hybrid_retrieve,
    get_retrieval_stats,
    get_document_by_id
)
from .metadata_boosting import compute_boost, batch_compute_boost, filter_documents

__all__ = [
    "load_retrieval_components",
    "hybrid_retrieve",
    "get_retrieval_stats",
    "get_document_by_id",
    "compute_boost",
    "batch_compute_boost", 
    "filter_documents"
]