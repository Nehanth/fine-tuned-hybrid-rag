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
import random
from typing import Dict, List, Any
from scipy.stats import permutation_test


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import load_retrieval_components, hybrid_retrieve
from sentence_transformers import SentenceTransformer
from finetune.generate_pairs import extract_key_terms, generate_keyword_queries, generate_question_queries, generate_title_based_queries


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
        
    def retrieve(self, queries: Dict[str, str], user_filters_list: Dict[str, Dict] = None, top_k: int = 10) -> tuple:
        """Retrieve documents for queries and return results with timing."""
        results = {}
        times = {}
        
        if user_filters_list is None:
            user_filters_list = {}
        
        for qid, query in queries.items():
            start_time = time.perf_counter()
            
            # Get user filters for this query
            user_filters = user_filters_list.get(qid, {})
            
            # Use our hybrid retrieval system with metadata filters
            retrieved_docs = hybrid_retrieve(
                query=query,
                components=self.components,
                user_filters=user_filters,
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


def create_realistic_qrels(documents: List[Dict], num_queries: int = 100) -> tuple:
    """Create realistic queries with metadata filters and relevance judgments."""
    
    queries = {}
    qrels = {}
    user_filters_list = {}  # Store user filters for each query
    
    # Sample documents to create queries from
    sampled_indices = random.sample(range(len(documents)), min(num_queries, len(documents)))
    
    def generate_realistic_query(doc: Dict) -> str:
        """Generate a realistic query using the same logic as training data generation."""
        # Generate different types of queries using imported functions
        all_queries = []
        
        # 1. Keyword-based queries (most common in real search)
        all_queries.extend(generate_keyword_queries(doc))
        
        # 2. Question-based queries  
        all_queries.extend(generate_question_queries(doc))
        
        # 3. Title-based queries
        all_queries.extend(generate_title_based_queries(doc))
        
        # Remove duplicates and empty queries
        unique_queries = []
        seen_queries = set()
        for query in all_queries:
            query = query.strip()
            if query and len(query) >= 3 and query not in seen_queries:
                seen_queries.add(query)
                unique_queries.append(query)
        
        # Select query with same balanced distribution as training
        if unique_queries:
            # Get different types of queries
            keyword_queries = [q for q in unique_queries if not q.startswith(('what is', 'how does', 'research on'))]
            question_queries = [q for q in unique_queries if q.startswith(('what is', 'how does', 'research on'))]
            
            # Balanced selection: 70% keywords, 30% questions (matches training)
            if question_queries and random.random() < 0.3:
                return random.choice(question_queries)
            elif keyword_queries:
                return keyword_queries[0]
            else:
                return unique_queries[0]
        
        # Fallback
        return "research paper"
    
    def generate_user_filters(doc: Dict) -> Dict[str, Any]:
        """Generate realistic user filters based on document metadata."""
        metadata = doc.get('metadata', {})
        filters = {}
        
        # Randomly add different types of filters (simulate user preferences)
        filter_probability = random.random()
        
        # 30% chance of field preference
        if filter_probability < 0.3:
            fields = metadata.get('fieldsOfStudy', [])
            if fields:
                filters['field'] = random.choice(fields)
        
        # 20% chance of venue preference
        elif filter_probability < 0.5:
            venue = metadata.get('venue', '')
            if venue and len(venue.strip()) > 0:
                filters['venue'] = venue
        
        # 25% chance of year-based filters
        elif filter_probability < 0.75:
            doc_year = metadata.get('year')
            if doc_year and isinstance(doc_year, (int, float)):
                year = int(doc_year)
                # Recent papers preference
                if random.random() < 0.6:
                    filters['year_after'] = max(1990, year - random.randint(5, 15))
                else:
                    # Historical papers preference
                    filters['year_before'] = min(2023, year + random.randint(2, 10))
        
        # 15% chance of author preference
        elif filter_probability < 0.9:
            authors = metadata.get('authors', [])
            if authors:
                author = random.choice(authors)
                if isinstance(author, dict) and 'name' in author:
                    # Use last name for author search
                    name = author['name']
                    if ' ' in name:
                        filters['author'] = name.split()[-1]
        
        # 10% chance of hard filters
        if random.random() < 0.1:
            # Add some hard filters occasionally
            fields = metadata.get('fieldsOfStudy', [])
            if fields:
                filters['required_fields'] = [random.choice(fields)]
        
        return filters
    
    for i, doc_idx in enumerate(sampled_indices):
        doc = documents[doc_idx]
        qid = f"q{i}"
        doc_id = str(doc_idx)
        
        # Skip if document doesn't have required fields
        if not doc.get('metadata', {}).get('title') or not doc.get('metadata', {}).get('abstract'):
            continue
        
        # Generate realistic query
        query = generate_realistic_query(doc)
        
        # Skip if query is too short or generic
        if len(query.strip()) < 3 or query == "research paper":
            continue
        
        # Generate user filters for this query
        user_filters = generate_user_filters(doc)
        
        queries[qid] = query
        user_filters_list[qid] = user_filters
        
        # Create relevance judgments - the source document is relevant
        # But also find other potentially relevant documents based on similar topics
        qrels[qid] = {doc_id: 1}  # Source document is relevant
        
        # Optional: Add other documents with similar fields of study as partially relevant
        source_fields = set(doc.get('metadata', {}).get('fieldsOfStudy', []))
        if source_fields:
            similar_count = 0
            for other_idx, other_doc in enumerate(documents):
                if other_idx == doc_idx or similar_count >= 3:  # Limit to 3 additional relevant docs
                    continue
                
                other_fields = set(other_doc.get('metadata', {}).get('fieldsOfStudy', []))
                if source_fields & other_fields:  # If fields overlap
                    # Check if titles have common terms (simple similarity check)
                    source_title_terms = set(extract_key_terms(doc.get('metadata', {}).get('title', ''), 5))
                    other_title_terms = set(extract_key_terms(other_doc.get('metadata', {}).get('title', ''), 5))
                    
                    if len(source_title_terms & other_title_terms) >= 1:  # At least 1 common term
                        qrels[qid][str(other_idx)] = 1  # Also relevant
                        similar_count += 1
    
    return queries, qrels, user_filters_list


def permutation_test_for_paired_samples(scores_a, scores_b, iterations=10_000):
    """Performs a permutation test of a given statistic on provided data."""
    
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
    
    print("BEIR-Style Neural Hybrid Retrieval Evaluation")
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
    
    # Create realistic queries and qrels
    print("Creating realistic queries and relevance judgments...")
    queries, qrels, user_filters_list = create_realistic_qrels(documents, num_queries)
    print(f"Created {len(queries)} query-document pairs")
    
    # Show some example filters
    print("\nExample user filters:")
    for i, (qid, filters) in enumerate(list(user_filters_list.items())[:3]):
        print(f"{i+1}. Query '{qid}': {queries[qid]}")
        print(f"   Filters: {filters}")
    
    # Load neural fusion components for base and fine-tuned models
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
    base_results, base_times = base_retriever.retrieve(queries, user_filters_list, top_k)
    
    # Evaluate fine-tuned model if available
    if compare_models:
        print("Evaluating fine-tuned model...")
        ft_results, ft_times = ft_retriever.retrieve(queries, user_filters_list, top_k)
    
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
    print("NEURAL FUSION RESULTS")
    print("=" * 30)
    
    if compare_models:
        # Statistical comparison
        has_significant_difference = print_scores(all_scores)
        
        # Show learned weight insights
        print("\nLEARNED WEIGHT ANALYSIS")
        print("-" * 30)
        
        if "weight_learner" in base_components:
            learned_weights = base_components["weight_learner"].get_weights()
            print(f"Neural weights: dense={learned_weights[0]:.3f}, sparse={learned_weights[1]:.3f}, boost={learned_weights[2]:.3f}")
            
            # Show weight distribution insights
            total_weight = sum(learned_weights)
            print(f"Weight distribution:")
            print(f"   Dense (semantic):  {learned_weights[0]/total_weight*100:.1f}%")
            print(f"   Sparse (keywords): {learned_weights[1]/total_weight*100:.1f}%") 
            print(f"   Boost (metadata):  {learned_weights[2]/total_weight*100:.1f}%")
            
            # Determine key insights from learned weights
            max_weight_idx = np.argmax(learned_weights)
            component_names = ["dense (semantic)", "sparse (TF-IDF)", "boost (metadata)"]
            print(f"Key insight: {component_names[max_weight_idx]} has highest learned importance")
    
    # Summary
    print("\nSUMMARY")
    print("-" * 20)
    if compare_models and has_significant_difference:
        print("🎯 Neural fusion model shows significant performance improvement!")
    elif compare_models:
        print("📊 Neural fusion model performance evaluation complete")
    else:
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
        
        print("\nNeural fusion evaluation completed successfully")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc() 
        sys.exit(1) 