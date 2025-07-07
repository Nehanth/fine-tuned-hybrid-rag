#!/usr/bin/env python3
"""
Model Evaluation Script

Tests base model vs fine-tuned model using the full hybrid retrieval system.
Tests on the actual full dataset to find evaluation documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from retrieval.hybrid_retriever import load_retrieval_components, hybrid_retrieve, get_retrieval_stats


def load_evaluation_dataset(filepath="eval/eval_dataset.json"):
    """Load the evaluation dataset from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_document_in_dataset(target_doc, documents):
    """Find a document in the full dataset by matching title."""
    target_title = target_doc['title'].lower().strip()
    
    for i, doc in enumerate(documents):
        if doc['text'].lower().startswith(target_title):
            return i
    
    # If not found by title start, try partial match
    for i, doc in enumerate(documents):
        if target_title in doc['text'].lower():
            return i
    
    return None


def evaluate_with_hybrid_system(components, query, correct_doc, distractor_docs, user_filters=None):
    """Evaluate using the full hybrid retrieval system on the actual dataset."""
    # Find the correct document in the full dataset
    correct_doc_id = find_document_in_dataset(correct_doc, components['documents'])
    
    # Find distractor documents
    distractor_doc_ids = []
    for distractor in distractor_docs:
        doc_id = find_document_in_dataset(distractor, components['documents'])
        if doc_id is not None:
            distractor_doc_ids.append(doc_id)
    
    # Run hybrid retrieval on full dataset
    results = hybrid_retrieve(
        query=query,
        components=components,
        user_filters=user_filters,
        top_k=100,  # Get more results to find our target docs
        candidate_pool_size=10000
    )
    
    # Find ranks of our target documents
    correct_rank = None
    distractor_ranks = []
    
    for i, result in enumerate(results):
        if result['doc_id'] == correct_doc_id:
            correct_rank = i + 1
        elif result['doc_id'] in distractor_doc_ids:
            distractor_ranks.append(i + 1)
    
    # Get the correct document's score
    correct_score = None
    if correct_rank is not None:
        correct_score = results[correct_rank - 1]['scores']['final_score']
    
    return {
        'results': results,
        'correct_rank': correct_rank,
        'correct_doc_id': correct_doc_id,
        'distractor_ranks': distractor_ranks,
        'correct_final_score': correct_score,
        'found_correct': correct_doc_id is not None,
        'found_distractors': len(distractor_doc_ids)
    }


def run_evaluation():
    """Main evaluation function using hybrid retrieval system."""
    print("Starting Hybrid Retrieval Evaluation")
    print("=" * 50)
    
    # Load evaluation dataset
    eval_data = load_evaluation_dataset()
    
    print(f"Dataset: {eval_data['name']}")
    print(f"Number of queries: {len(eval_data['queries'])}")
    print()
    
    # Load base model components
    print("Loading base model components...")
    base_components = load_retrieval_components()
    
    # Load fine-tuned model components  
    print("Loading fine-tuned model components...")
    ft_components = load_retrieval_components()
    # Replace the sentence model with fine-tuned version
    from sentence_transformers import SentenceTransformer
    ft_components['sentence_model'] = SentenceTransformer('finetune/model/')
    
    # Replace embeddings with fine-tuned embeddings
    print("Loading fine-tuned embeddings...")
    ft_components['dense_embeddings'] = np.load('embeddings/dense_finetuned.npy')
    print(f"Loaded fine-tuned embeddings shape: {ft_components['dense_embeddings'].shape}")
    
    print("Both systems loaded")
    print()
    
    # Store results
    base_results = []
    ft_results = []
    
    print("Running evaluation on each query...")
    print()
    
    # Evaluate each query
    for i, query_data in enumerate(eval_data['queries']):
        query = query_data['query']
        correct_doc = query_data['correct_doc']
        distractor_docs = query_data['distractor_docs']
        user_filters = query_data.get('user_filters', None)
        
        print(f"Query {i+1}: {query[:60]}...")
        print(f"Field: {query_data['field']}")
        if user_filters:
            print(f"🔍 METADATA FILTERING ENABLED:")
            for key, value in user_filters.items():
                print(f"    {key}: {value}")
        else:
            print("🔍 No metadata filtering (basic hybrid retrieval only)")
        
        # Evaluate with base model
        base_result = evaluate_with_hybrid_system(base_components, query, correct_doc, distractor_docs, user_filters)
        base_results.append(base_result)
        
        # Evaluate with fine-tuned model
        ft_result = evaluate_with_hybrid_system(ft_components, query, correct_doc, distractor_docs, user_filters)
        ft_results.append(ft_result)
        
        # Print comparison
        base_rank = base_result['correct_rank']
        ft_rank = ft_result['correct_rank']
        
        if base_rank and ft_rank:
            base_score = base_result['correct_final_score']
            ft_score = ft_result['correct_final_score']
            
            print(f"  Base Model:    Rank {base_rank}/100 (final score: {base_score:.3f})")
            print(f"  Fine-tuned:    Rank {ft_rank}/100 (final score: {ft_score:.3f})")
            
            # Show improvement status
            if ft_rank < base_rank:
                print("  Status: Fine-tuned ranked higher")
            elif base_rank < ft_rank:
                print("  Status: Base model ranked higher")
            else:
                print("  Status: Same rank")
        else:
            print(f"  Base Model:    Document found: {base_result['found_correct']}")
            print(f"  Fine-tuned:    Document found: {ft_result['found_correct']}")
            print("  Status: Could not find target document in dataset")
        
        print(f"  Found {base_result['found_distractors']}/{len(distractor_docs)} distractors")
        print()
    
    # Filter results where documents were found
    valid_base_results = [r for r in base_results if r['correct_rank'] is not None]
    valid_ft_results = [r for r in ft_results if r['correct_rank'] is not None]
    
    if not valid_base_results:
        print("No valid evaluation results - could not find target documents in dataset")
        return base_results, ft_results
    
    # Calculate metrics
    base_hits_1 = sum(1 for r in valid_base_results if r['correct_rank'] == 1)
    ft_hits_1 = sum(1 for r in valid_ft_results if r['correct_rank'] == 1)
    
    base_hits_10 = sum(1 for r in valid_base_results if r['correct_rank'] <= 10)
    ft_hits_10 = sum(1 for r in valid_ft_results if r['correct_rank'] <= 10)
    
    # MRR
    base_mrr = sum(1.0 / r['correct_rank'] for r in valid_base_results) / len(valid_base_results)
    ft_mrr = sum(1.0 / r['correct_rank'] for r in valid_ft_results) / len(valid_ft_results)
    
    # Average final scores
    base_avg_score = sum(r['correct_final_score'] for r in valid_base_results) / len(valid_base_results)
    ft_avg_score = sum(r['correct_final_score'] for r in valid_ft_results) / len(valid_ft_results)
    
    # Average ranks
    base_avg_rank = sum(r['correct_rank'] for r in valid_base_results) / len(valid_base_results)
    ft_avg_rank = sum(r['correct_rank'] for r in valid_ft_results) / len(valid_ft_results)
    
    # Print results
    print("HYBRID RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Valid evaluations: {len(valid_base_results)}/{len(base_results)}")
    print()
    print(f"{'Metric':<20} {'Base Model':<12} {'Fine-tuned':<12} {'Improvement':<12}")
    print("-" * 60)
    
    # Hits@1
    base_h1 = (base_hits_1 / len(valid_base_results)) * 100
    ft_h1 = (ft_hits_1 / len(valid_ft_results)) * 100
    print(f"{'Hits@1 (%)':<20} {base_h1:<12.1f} {ft_h1:<12.1f} {ft_h1-base_h1:<+12.1f}")
    
    # Hits@10
    base_h10 = (base_hits_10 / len(valid_base_results)) * 100
    ft_h10 = (ft_hits_10 / len(valid_ft_results)) * 100
    print(f"{'Hits@10 (%)':<20} {base_h10:<12.1f} {ft_h10:<12.1f} {ft_h10-base_h10:<+12.1f}")
    
    # MRR
    print(f"{'MRR':<20} {base_mrr:<12.3f} {ft_mrr:<12.3f} {ft_mrr-base_mrr:<+12.3f}")
    
    # Average rank
    print(f"{'Avg Rank':<20} {base_avg_rank:<12.1f} {ft_avg_rank:<12.1f} {ft_avg_rank-base_avg_rank:<+12.1f}")
    
    # Average final score
    print(f"{'Avg Final Score':<20} {base_avg_score:<12.3f} {ft_avg_score:<12.3f} {ft_avg_score-base_avg_score:<+12.3f}")
    
    print()
    
    # Summary
    print("SUMMARY")
    print("-" * 20)
    if ft_h1 > base_h1:
        print("Fine-tuning improved accuracy with hybrid retrieval")
    elif ft_h1 == base_h1:
        print("Fine-tuning maintained accuracy with hybrid retrieval")
    else:
        print("Base model had better accuracy with hybrid retrieval")
    
    if ft_avg_score > base_avg_score:
        print("Fine-tuning improved hybrid retrieval scores")
    else:
        print("Base model had higher hybrid retrieval scores")
    
    if ft_avg_rank < base_avg_rank:
        print("Fine-tuning improved average ranking")
    else:
        print("Base model had better average ranking")
    
    return base_results, ft_results


if __name__ == "__main__":
    try:
        base_results, ft_results = run_evaluation()
        print("\nHybrid retrieval evaluation completed successfully")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc() 