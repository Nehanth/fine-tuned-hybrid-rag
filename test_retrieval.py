#!/usr/bin/env python3
"""
Test script for hybrid retrieval system.
Demonstrates usage with example queries and filters.
"""

import sys
import json
from retrieval.hybrid_retriever import load_retrieval_components, hybrid_retrieve, get_retrieval_stats


def test_basic_retrieval():
    """Test basic hybrid search without metadata filtering."""
    print("=" * 60)
    print("Testing Basic Search (No Metadata Filtering)")
    print("=" * 60)
    print("What's happening:")
    print("   - MiniLM encoder: understands meaning (60% weight)")
    print("   - TF-IDF encoder: finds keywords (30% weight)")
    print("   - NO metadata filtering: all papers treated equally")
    print("   - Only semantic + keyword similarity matters\n")
    
    # Load the search system components
    components = load_retrieval_components()
    
    # Show system info
    stats = get_retrieval_stats(components)
    print(f"System Info:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Search for something
    query = "machine learning neural networks deep learning"
    results = hybrid_retrieve(query, components, top_k=5)
    
    print(f"Search Query: '{query}'")
    print(f"Top {len(results)} Results:")
    
    for i, result in enumerate(results):
        print(f"\n  {i+1}. [Score: {result['scores']['final_score']:.4f}]")
        print(f"     Paper: {result['text'][:100]}...")
        print(f"     Year: {result['metadata']['year']}")
        print(f"     Publisher: {result['metadata']['venue']}")
        print(f"     Fields: {result['metadata']['fieldsOfStudy']}")
        print(f"     Scores: Semantic={result['scores']['dense_score']:.3f}, "
              f"Keywords={result['scores']['sparse_score']:.3f}, "
              f"Boost={result['scores']['boost_score']:.3f} (same for all)")
    
    return components  # Return for reuse in next test


def test_metadata_boosting(components):
    """Test how metadata preferences boost relevant papers."""
    print("\n\n" + "=" * 60)
    print("Testing Search with Metadata Filtering")
    print("=" * 60)
    print("What's happening:")
    print("   - MiniLM encoder: semantic similarity (60%)")
    print("   - TF-IDF encoder: keyword matching (30%)")
    print("   - Metadata filtering: boost Computer Science papers (+20%)")
    print("   - Year filtering: bonus points for recent papers")
    print("   - Different papers get different boost scores\n")
    
    # Search with metadata preferences
    query = "artificial intelligence machine learning"
    filters = {
        "field": "Computer Science",
        "year_after": 2018
    }
    
    results = hybrid_retrieve(query, components, user_filters=filters, top_k=5)
    
    print(f"Search Query: '{query}'")
    print(f"Metadata Filters: {filters}")
    print(f"Top {len(results)} Results:")
    
    for i, result in enumerate(results):
        print(f"\n  {i+1}. [Score: {result['scores']['final_score']:.4f}]")
        print(f"     Paper: {result['text'][:100]}...")
        print(f"     Year: {result['metadata']['year']}")
        print(f"     Publisher: {result['metadata']['venue']}")
        print(f"     Fields: {result['metadata']['fieldsOfStudy']}")
        print(f"     Scores: Semantic={result['scores']['dense_score']:.3f}, "
              f"Keywords={result['scores']['sparse_score']:.3f}, "
              f"Boost={result['scores']['boost_score']:.3f} (varies by match)")


if __name__ == "__main__":
    try:
        print("Starting Hybrid Search System Tests")
        print("This will compare basic search vs. search with metadata filtering.\n")
        
        # Run first test and get components
        components = test_basic_retrieval()
        
        # Reuse components for second test (more efficient)
        test_metadata_boosting(components)
        
        print("\n\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 