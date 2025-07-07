#!/usr/bin/env python3
"""
Model Evaluation Script

Tests base model vs fine-tuned model on academic paper retrieval tasks.
Calculates standard information retrieval metrics.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_evaluation_dataset(filepath="eval/eval_dataset.json"):
    """Load the evaluation dataset from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_models(base_model_name="all-MiniLM-L6-v2", finetuned_model_path="finetune/model/"):
    """Load both base and fine-tuned models."""
    print("Loading models...")
    base_model = SentenceTransformer(base_model_name)
    finetuned_model = SentenceTransformer(finetuned_model_path)
    print("Models loaded successfully")
    return base_model, finetuned_model


def prepare_documents(correct_doc, distractor_docs):
    """Prepare document texts by combining title and abstract."""
    all_docs = [correct_doc] + distractor_docs
    doc_texts = []
    
    for doc in all_docs:
        # Combine title and abstract like in training data
        if 'abstract' in doc:
            doc_text = f"{doc['title']} {doc['abstract']}"
        else:
            doc_text = doc['title']
        doc_texts.append(doc_text)
    
    return doc_texts


def evaluate_model(model, query, doc_texts):
    """Evaluate a single model on a query-document set."""
    # Encode query and documents
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(doc_texts)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get ranking (indices sorted by similarity, descending)
    ranking = np.argsort(similarities)[::-1]
    
    # Find rank of correct document (always index 0)
    correct_rank = np.where(ranking == 0)[0][0] + 1
    
    return {
        'similarities': similarities,
        'ranking': ranking,
        'correct_rank': correct_rank,
        'correct_similarity': similarities[0]
    }


def calculate_metrics(results):
    """Calculate information retrieval metrics from evaluation results."""
    total_queries = len(results)
    
    hits_at_1 = sum(1 for r in results if r['correct_rank'] == 1)
    hits_at_3 = sum(1 for r in results if r['correct_rank'] <= 3)
    
    # Mean Reciprocal Rank
    mrr = sum(1.0 / r['correct_rank'] for r in results) / total_queries
    
    # Average correct similarity score
    avg_similarity = sum(r['correct_similarity'] for r in results) / total_queries
    
    return {
        'hits_at_1': hits_at_1 / total_queries,
        'hits_at_3': hits_at_3 / total_queries,
        'mrr': mrr,
        'avg_similarity': avg_similarity
    }


def run_evaluation():
    """Main evaluation function."""
    print("Starting Model Evaluation")
    print("=" * 50)
    
    # Load data and models
    eval_data = load_evaluation_dataset()
    base_model, finetuned_model = load_models()
    
    print(f"Dataset: {eval_data['name']}")
    print(f"Number of queries: {len(eval_data['queries'])}")
    print()
    
    # Store results for each model
    base_results = []
    finetuned_results = []
    
    print("Running evaluation on each query...")
    print()
    
    # Evaluate each query
    for i, query_data in enumerate(eval_data['queries']):
        query = query_data['query']
        correct_doc = query_data['correct_doc']
        distractor_docs = query_data['distractor_docs']
        
        print(f"Query {i+1}: {query[:60]}...")
        print(f"Field: {query_data['field']}")
        
        # Prepare documents
        doc_texts = prepare_documents(correct_doc, distractor_docs)
        
        # Evaluate both models
        base_result = evaluate_model(base_model, query, doc_texts)
        finetuned_result = evaluate_model(finetuned_model, query, doc_texts)
        
        # Store results
        base_results.append(base_result)
        finetuned_results.append(finetuned_result)
        
        # Print comparison
        print(f"  Base Model:    Rank {base_result['correct_rank']}/4 (similarity: {base_result['correct_similarity']:.3f})")
        print(f"  Fine-tuned:    Rank {finetuned_result['correct_rank']}/4 (similarity: {finetuned_result['correct_similarity']:.3f})")
        
        # Show improvement status
        if finetuned_result['correct_rank'] < base_result['correct_rank']:
            print("  Status: Fine-tuned ranked higher")
        elif base_result['correct_rank'] < finetuned_result['correct_rank']:
            print("  Status: Base model ranked higher")
        else:
            print("  Status: Same rank")
        print()
    
    # Calculate final metrics
    base_metrics = calculate_metrics(base_results)
    finetuned_metrics = calculate_metrics(finetuned_results)
    
    # Print results
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Metric':<20} {'Base Model':<12} {'Fine-tuned':<12} {'Improvement':<12}")
    print("-" * 60)
    
    # Hits@1
    base_h1 = base_metrics['hits_at_1'] * 100
    ft_h1 = finetuned_metrics['hits_at_1'] * 100
    print(f"{'Hits@1 (%)':<20} {base_h1:<12.1f} {ft_h1:<12.1f} {ft_h1-base_h1:<+12.1f}")
    
    # Hits@3
    base_h3 = base_metrics['hits_at_3'] * 100
    ft_h3 = finetuned_metrics['hits_at_3'] * 100
    print(f"{'Hits@3 (%)':<20} {base_h3:<12.1f} {ft_h3:<12.1f} {ft_h3-base_h3:<+12.1f}")
    
    # MRR
    base_mrr = base_metrics['mrr']
    ft_mrr = finetuned_metrics['mrr']
    print(f"{'MRR':<20} {base_mrr:<12.3f} {ft_mrr:<12.3f} {ft_mrr-base_mrr:<+12.3f}")
    
    # Average similarity
    base_sim = base_metrics['avg_similarity']
    ft_sim = finetuned_metrics['avg_similarity']
    print(f"{'Avg Similarity':<20} {base_sim:<12.3f} {ft_sim:<12.3f} {ft_sim-base_sim:<+12.3f}")
    
    print()
    
    # Summary
    print("SUMMARY")
    print("-" * 20)
    if ft_h1 > base_h1:
        print("Fine-tuning improved accuracy")
    elif ft_h1 == base_h1:
        print("Fine-tuning maintained accuracy")
    else:
        print("Base model had better accuracy")
    
    if ft_sim > base_sim:
        print("Fine-tuning improved confidence scores")
    else:
        print("Base model had higher confidence scores")
    
    return base_metrics, finetuned_metrics


if __name__ == "__main__":
    try:
        base_metrics, finetuned_metrics = run_evaluation()
        print("\nEvaluation completed successfully")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc() 