"""
Metadata boosting module for hybrid retrieval system.
Computes boost scores based on document metadata and user preferences.
"""

from typing import Dict, List, Any


def compute_boost(metadata: Dict[str, Any], user_filters: Dict[str, Any]) -> float:
    """
    Compute metadata boost score based on document metadata and user filters.
    
    Args:
        metadata (dict): Document metadata containing:
            - year (int): Publication year
            - venue (str): Publication venue
            - fieldsOfStudy (list): List of research fields
            - authors (list): List of author information
        user_filters (dict): User-specified filters/preferences:
            - field (str, optional): Preferred field of study
            - venue (str, optional): Preferred venue
            - year_after (int, optional): Boost papers after this year
            - year_before (int, optional): Boost papers before this year
            - author (str, optional): Preferred author name
    
    Returns:
        float: Boost multiplier (1.0 = no boost, >1.0 = positive boost, <1.0 = negative boost)
    
    Example:
        >>> metadata = {
        ...     "year": 2020,
        ...     "venue": "NeurIPS",
        ...     "fieldsOfStudy": ["Computer Science", "Machine Learning"],
        ...     "authors": [{"name": "Jane Doe"}]
        ... }
        >>> filters = {"field": "Computer Science", "year_after": 2018}
        >>> boost = compute_boost(metadata, filters)
        >>> print(f"Boost score: {boost:.2f}")
    """
    boost_score = 1.0  # Default: no boost
    
    # Field of study boost
    if "field" in user_filters and user_filters["field"]:
        preferred_field = user_filters["field"].lower()
        doc_fields = metadata.get("fieldsOfStudy", [])
        
        if doc_fields and isinstance(doc_fields, list):
            # Check if any document field matches the preferred field
            for field in doc_fields:
                if field and preferred_field in field.lower():
                    boost_score *= 1.2  # 20% boost for field match
                    break
    
    # Venue boost
    if "venue" in user_filters and user_filters["venue"]:
        preferred_venue = user_filters["venue"].lower()
        doc_venue = metadata.get("venue", "")
        
        if doc_venue and preferred_venue in doc_venue.lower():
            boost_score *= 1.15  # 15% boost for venue match
    
    # Year-based boost
    doc_year = metadata.get("year")
    if doc_year and isinstance(doc_year, int):
        
        # Boost recent papers
        if "year_after" in user_filters and user_filters["year_after"]:
            year_threshold = user_filters["year_after"]
            if doc_year > year_threshold:
                # Linear boost based on how recent the paper is
                years_since = doc_year - year_threshold
                boost_score *= min(1.0 + 0.02 * years_since, 1.3)  # Max 30% boost
        
        # Boost papers before a certain year (for historical research)
        if "year_before" in user_filters and user_filters["year_before"]:
            year_threshold = user_filters["year_before"]
            if doc_year < year_threshold:
                boost_score *= 1.1  # 10% boost for historical papers
    
    # Author boost
    if "author" in user_filters and user_filters["author"]:
        preferred_author = user_filters["author"].lower()
        doc_authors = metadata.get("authors", [])
        
        if doc_authors and isinstance(doc_authors, list):
            for author in doc_authors:
                if isinstance(author, dict) and "name" in author:
                    author_name = author["name"].lower()
                    if preferred_author in author_name:
                        boost_score *= 1.25  # 25% boost for author match
                        break
    
    return boost_score


def batch_compute_boost(metadata_list: List[Dict[str, Any]], 
                       user_filters: Dict[str, Any]) -> List[float]:
    """
    Compute boost scores for multiple documents efficiently.
    
    Args:
        metadata_list (list): List of metadata dictionaries
        user_filters (dict): User-specified filters/preferences
    
    Returns:
        list: List of boost scores corresponding to each document
    """
    return [compute_boost(metadata, user_filters) for metadata in metadata_list]


def filter_documents(documents: List[Dict[str, Any]], 
                    user_filters: Dict[str, Any]) -> List[bool]:
    """
    Apply hard filters to determine which documents should be included.
    
    Args:
        documents (list): List of document dictionaries with metadata
        user_filters (dict): User-specified filters with hard constraints:
            - min_year (int): Minimum publication year
            - max_year (int): Maximum publication year
            - required_fields (list): Must contain at least one of these fields
            - excluded_venues (list): Exclude documents from these venues
    
    Returns:
        list: Boolean mask indicating which documents pass the filters
    """
    mask = []
    
    for doc in documents:
        metadata = doc.get("metadata", {})
        include = True
        
        # Year filters
        doc_year = metadata.get("year")
        if doc_year:
            if "min_year" in user_filters and doc_year < user_filters["min_year"]:
                include = False
            if "max_year" in user_filters and doc_year > user_filters["max_year"]:
                include = False
        
        # Required fields filter
        if "required_fields" in user_filters and user_filters["required_fields"]:
            required_fields = [f.lower() for f in user_filters["required_fields"]]
            doc_fields = metadata.get("fieldsOfStudy", [])
            
            if doc_fields:
                doc_fields_lower = [f.lower() for f in doc_fields if f]
                has_required_field = any(
                    any(req in doc_field for req in required_fields) 
                    for doc_field in doc_fields_lower
                )
                if not has_required_field:
                    include = False
            else:
                include = False  # No fields specified, exclude
        
        # Excluded venues filter
        if "excluded_venues" in user_filters and user_filters["excluded_venues"]:
            excluded_venues = [v.lower() for v in user_filters["excluded_venues"]]
            doc_venue = metadata.get("venue", "").lower()
            
            if any(excluded in doc_venue for excluded in excluded_venues):
                include = False
        
        mask.append(include)
    
    return mask