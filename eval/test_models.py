#!/usr/bin/env python3
"""
BEIR-style evaluation for hybrid RAG retrieval system.
Compares base model vs fine-tuned model

Insipiration of this evaluation script: https://github.com/opendatahub-io/rag/blob/main/benchmarks/llama-stack-rag-with-beir/benchmark_beir_ls_vs_no_ls.py
"""


import argparse
import itertools
import json
import logging
import os
import pathlib
import sys
import time
import numpy as np
import pytrec_eval
import yaml
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import load_retrieval_components, hybrid_retrieve
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval system with BEIR methodology"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to evaluate (default: 100)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve (default: 10)"
    )
    return parser.parse_args()


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Wrapper for our hybrid retrieval system to match BEIR interface."""
    
    def __init__(self, components, model_name="base"):
        self.components = components
        self.model_name = model_name
        
    def retrieve(self, queries: Dict[str, str], top_k: int = 10) -> tuple:
        """Retrieve documents for queries and return results with timing."""
        results = {}
        times = {}
        
        for qid, query in queries.items():
            start_time = time.perf_counter()
            
            # Use our hybrid retrieval system
            retrieved_docs = hybrid_retrieve(
                query=query,
                components=self.components,
                top_k=top_k
            )
            
            end_time = time.perf_counter()
            times[qid] = end_time - start_time
            
            # Convert to BEIR format: {doc_id: score}
            doc_scores = {}
            for doc in retrieved_docs:
                doc_scores[str(doc['doc_id'])] = doc['scores']['final_score']
            
            results[qid] = doc_scores
            
        return results, times


def create_synthetic_qrels(documents: List[Dict], num_queries: int = 100) -> tuple:
    """Create synthetic queries and relevance judgments from our dataset."""
    import random
    
    queries = {}
    qrels = {}
    
    # Sample documents to create queries from
    sampled_indices = random.sample(range(len(documents)), min(num_queries, len(documents)))
    
    for i, doc_idx in enumerate(sampled_indices):
        doc = documents[doc_idx]
        qid = f"q{i}"
        doc_id = str(doc_idx)  # Use actual document index from the full dataset
        
        # Create query from document text
        text_words = doc.get('text', '').split()
        if len(text_words) < 10:
            continue
            
        # Use middle portion of text as query to avoid exact matches
        start_idx = len(text_words) // 4
        end_idx = start_idx + min(10, len(text_words) // 2)
        query_words = text_words[start_idx:end_idx]
        query = ' '.join(query_words)
        
        # Add domain-specific terms from ALL available metadata (matches training)
        if 'metadata' in doc:
            metadata = doc['metadata']
            metadata_terms = []
            
            # Add field of study
            fields = metadata.get('fieldsOfStudy', [])
            if fields and len(fields) > 0:
                main_field = fields[0] if isinstance(fields[0], str) else str(fields[0])
                metadata_terms.append(main_field)
            
            # Add venue information
            venue = metadata.get('venue', '')
            if venue and isinstance(venue, str) and len(venue.strip()) > 0:
                # Use venue abbreviation or first word for conciseness
                venue_term = venue.strip().split()[0] if ' ' in venue else venue.strip()
                metadata_terms.append(venue_term)
            
            # Add year information
            year = metadata.get('year')
            if year and isinstance(year, (int, float)):
                metadata_terms.append(str(int(year)))
            
            # Add author information (first author's last name)
            authors = metadata.get('authors', [])
            if authors and len(authors) > 0:
                first_author = authors[0]
                if isinstance(first_author, dict) and 'name' in first_author:
                    author_name = first_author['name']
                    # Extract last name (assuming "First Last" format)
                    if ' ' in author_name:
                        last_name = author_name.split()[-1]
                        metadata_terms.append(last_name)
                elif isinstance(first_author, str):
                    # Handle string author names
                    if ' ' in first_author:
                        last_name = first_author.split()[-1]
                        metadata_terms.append(last_name)
            
            # Combine metadata terms with query (limit to avoid overly long queries)
            if metadata_terms:
                # Use up to 3 metadata terms to keep queries reasonable (matches training)
                selected_terms = metadata_terms[:3]
                metadata_prefix = ' '.join(selected_terms)
                query = f"{metadata_prefix} {query}"
        
        queries[qid] = query
        
        # Create relevance judgments (the source doc is highly relevant)
        qrels[qid] = {doc_id: 1}  # Binary relevance
        
    return queries, qrels


def permutation_test_for_paired_samples(scores_a, scores_b, iterations=10_000):
    """Performs a permutation test of a given statistic on provided data."""
    from scipy.stats import permutation_test
    
    def _statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    
    result = permutation_test(
        data=(scores_a, scores_b),
        statistic=_statistic,
        n_resamples=iterations,
        alternative="two-sided",
        permutation_type="samples",
    )
    return float(result.pvalue)


def print_stats_significance(scores_a, scores_b, overview_label, label_a, label_b):
    """Print statistical significance test results."""
    mean_score_a = np.mean(scores_a)
    mean_score_b = np.mean(scores_b)
    
    p_value = permutation_test_for_paired_samples(scores_a, scores_b)
    
    print(overview_label)
    print(f" {label_a:<30}: {mean_score_a:>10.4f}")
    print(f" {label_b:<30}: {mean_score_b:>10.4f}")
    print(f" {'p_value':<30}: {p_value:>10.4f}")
    
    if p_value < 0.05:
        print("  p_value<0.05 so this result is statistically significant")
        higher_model = label_a if mean_score_a >= mean_score_b else label_b
        print(f"  {higher_model} has significantly higher performance")
        return True
    else:
        import math
        print("  p_value>=0.05 so this result is NOT statistically significant")
        print("  No significant difference detected")
        num_samples = len(scores_a)
        margin_of_error = 1 / math.sqrt(num_samples)
        print(f"  Margin of error: ±{margin_of_error:.1%} ({num_samples} samples)")
        return False


def get_metrics(all_scores):
    """Extract available metrics from scores."""
    for scores_for_dataset in all_scores.values():
        for scores_for_condition in scores_for_dataset.values():
            for scores_for_question in scores_for_condition.values():
                metrics = list(scores_for_question.keys())
                metrics.sort()
                return metrics
    return []


def print_scores(all_scores):
    """Print comparison of all scores with statistical significance."""
    metrics = get_metrics(all_scores)
    has_significant_difference = False
    
    for dataset_name, scores_for_dataset in all_scores.items():
        condition_labels = list(scores_for_dataset.keys())
        condition_labels.sort()
        
        for metric in metrics:
            if metric == "time":  # Skip timing for significance
                continue
                
            overview_label = f"{dataset_name} {metric}"
            for label_a, label_b in itertools.combinations(condition_labels, 2):
                scores_for_label_a = scores_for_dataset[label_a]
                scores_for_label_b = scores_for_dataset[label_b]
                
                scores_a = [score_group[metric] for score_group in scores_for_label_a.values()]
                scores_b = [score_group[metric] for score_group in scores_for_label_b.values()]
                
                is_significant = print_stats_significance(
                    scores_a, scores_b, overview_label, label_a, label_b
                )
                print()
                
                has_significant_difference = has_significant_difference or is_significant
    
    return has_significant_difference


def evaluate_hybrid_retrieval(num_queries=100, top_k=10):
    """Main evaluation function using BEIR methodology."""
    
    print("BEIR-Style Hybrid Retrieval Evaluation")
    print("=" * 50)
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Queries to evaluate: {num_queries}")
    print(f"Top-k retrieval: {top_k}")
    print()
    
    # Load full dataset for evaluation
    print("Loading dataset for evaluation...")
    documents = []
    with open(config["dataset"]["processed_path"], 'r') as f:
        for line in f:
            documents.append(json.loads(line.strip()))
    print(f"Loaded {len(documents)} documents")
    
    # Create synthetic queries and qrels
    print("Creating synthetic queries and relevance judgments...")
    queries, qrels = create_synthetic_qrels(documents, num_queries)
    print(f"Created {len(queries)} query-document pairs")
    print()
    
    # Load base model components
    print("Loading base model components...")
    base_components = load_retrieval_components(config)
    base_retriever = HybridRetriever(base_components, "BaseModel")
    
    # Load fine-tuned model components  
    print("Loading fine-tuned model components...")
    ft_components = load_retrieval_components(config)
    
    # Check if fine-tuned model exists
    ft_model_path = config["finetune"]["output_path"]
    ft_embeddings_path = config["embeddings"]["dense_finetuned_path"]
    
    if os.path.exists(ft_model_path) and os.path.exists(ft_embeddings_path):
        print("Fine-tuned model found, loading...")
        ft_components['sentence_model'] = SentenceTransformer(ft_model_path)
        ft_components['dense_embeddings'] = np.load(ft_embeddings_path)
        ft_retriever = HybridRetriever(ft_components, "FineTunedModel")
        compare_models = True
    else:
        print("Fine-tuned model not found, evaluating base model only...")
        ft_retriever = None
        compare_models = False
    
    print("Models loaded successfully")
    print()
    
    # Store all scores for statistical analysis
    all_scores = {"hybrid_evaluation": {}}
    
    # Evaluate base model
    print("Evaluating base model...")
    base_results, base_times = base_retriever.retrieve(queries, top_k)
    
    # Evaluate fine-tuned model if available
    if compare_models:
        print("Evaluating fine-tuned model...")
        ft_results, ft_times = ft_retriever.retrieve(queries, top_k)
    
    # Calculate BEIR metrics using pytrec_eval
    k_values = [top_k]
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    metrics_strings = {ndcg_string, map_string}
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_strings)
    
    # Evaluate base model
    base_scores = evaluator.evaluate(base_results)
    for qid, scores_for_qid in base_scores.items():
        scores_for_qid["time"] = base_times[qid]
    all_scores["hybrid_evaluation"]["BaseModel"] = base_scores
    
    # Evaluate fine-tuned model if available
    if compare_models:
        ft_scores = evaluator.evaluate(ft_results)
        for qid, scores_for_qid in ft_scores.items():
            scores_for_qid["time"] = ft_times[qid]
        all_scores["hybrid_evaluation"]["FineTunedModel"] = ft_scores
    
    print("Evaluation completed")
    print()
    
    # Print results
    print("RESULTS")
    print("=" * 30)
    
    if compare_models:
        # Statistical comparison
        has_significant_difference = print_scores(all_scores)
    
    # Summary
    print("\nSUMMARY")
    print("-" * 20)
    if has_significant_difference:
        print("Significant performance difference detected")
    else:
        print("No significant performance difference detected")
    
        # Single model results
        base_metrics = list(base_scores[list(base_scores.keys())[0]].keys())
        for metric in base_metrics:
            if metric != "time":
                scores = [base_scores[qid][metric] for qid in base_scores.keys()]
                mean_score = np.mean(scores)
                print(f"{metric}: {mean_score:.4f}")
    
    return all_scores


if __name__ == "__main__":
    args = parse_args()
    
    try:
        all_scores = evaluate_hybrid_retrieval(
            num_queries=args.num_queries,
            top_k=args.top_k
        )
        
        print("\nBEIR-style evaluation completed successfully")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc() 
        sys.exit(1) 