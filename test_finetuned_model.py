#!/usr/bin/env python3
"""
Test script to compare base model vs fine-tuned model performance.
We'll test a few example queries to see if fine-tuning improved retrieval.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time


def test_model_comparison():
    """
    Compare base model vs fine-tuned model on example academic queries.
    """
    print("=" * 60)
    print("🔬 Testing Base vs Fine-tuned Model Performance")
    print("=" * 60)
    
    # Load both models
    print("Loading models...")
    base_model = SentenceTransformer("all-MiniLM-L6-v2")
    finetuned_model = SentenceTransformer("finetune/model/")
    
    # Test queries and documents
    test_cases = [
        {
            "query": "Neural networks for image classification",
            "documents": [
                "Deep Convolutional Neural Networks for Large-Scale Image Recognition This paper investigates the application of deep learning architectures to image classification tasks.",
                "Machine Learning Approaches to Database Query Optimization In this work we explore various algorithmic techniques for improving database performance.",
                "Transformer Models for Natural Language Processing Recent advances in attention mechanisms have revolutionized text understanding applications."
            ]
        },
        {
            "query": "COVID-19 vaccine effectiveness",
            "documents": [
                "Efficacy and Safety of COVID-19 Vaccines: A Systematic Review Our analysis examines vaccination outcomes across multiple clinical trials and populations.",
                "Climate Change Impact on Agricultural Systems This study evaluates how environmental factors affect crop yields and farming practices.",
                "Quantum Computing Applications in Cryptography We present novel quantum algorithms for secure communication protocols."
            ]
        },
        {
            "query": "Machine learning optimization algorithms",
            "documents": [
                "Adam Optimizer for Deep Neural Network Training We propose improvements to gradient descent methods for faster convergence in machine learning models.",
                "Historical Analysis of Medieval Architecture This comprehensive survey examines building techniques and artistic styles from the medieval period.",
                "Genetic Engineering in Plant Biology Recent developments in CRISPR technology enable precise modifications to plant genomes for improved traits."
            ]
        }
    ]
    
    print("\n🧪 Running comparison tests...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        documents = test_case["documents"]
        
        print(f"Test {i}: '{query}'")
        print("-" * 50)
        
        # Encode with both models
        base_query_emb = base_model.encode([query])
        base_doc_embs = base_model.encode(documents)
        
        finetuned_query_emb = finetuned_model.encode([query])
        finetuned_doc_embs = finetuned_model.encode(documents)
        
        # Compute similarities
        base_similarities = cosine_similarity(base_query_emb, base_doc_embs)[0]
        finetuned_similarities = cosine_similarity(finetuned_query_emb, finetuned_doc_embs)[0]
        
        # Find best matches
        base_best_idx = np.argmax(base_similarities)
        finetuned_best_idx = np.argmax(finetuned_similarities)
        
        print(f"Base Model:")
        for j, (doc, sim) in enumerate(zip(documents, base_similarities)):
            marker = " 🏆" if j == base_best_idx else ""
            print(f"  {j+1}. [{sim:.3f}] {doc[:80]}...{marker}")
        
        print(f"\nFine-tuned Model:")
        for j, (doc, sim) in enumerate(zip(documents, finetuned_similarities)):
            marker = " 🏆" if j == finetuned_best_idx else ""
            improvement = finetuned_similarities[j] - base_similarities[j]
            sign = "↗️" if improvement > 0.01 else "↘️" if improvement < -0.01 else "→"
            print(f"  {j+1}. [{sim:.3f}] {doc[:80]}...{marker} {sign} ({improvement:+.3f})")
        
        # Check if fine-tuned model picked the correct answer
        correct_answer = 0  # First document is always the relevant one
        base_correct = base_best_idx == correct_answer
        finetuned_correct = finetuned_best_idx == correct_answer
        
        if finetuned_correct and not base_correct:
            print("🎯 IMPROVEMENT: Fine-tuned model found correct answer!")
        elif base_correct and not finetuned_correct:
            print("⚠️  REGRESSION: Base model was better")
        elif finetuned_correct and base_correct:
            print("✅ MAINTAINED: Both models found correct answer")
        else:
            print("❌ BOTH WRONG: Neither model found correct answer")
        
        print("\n")
    
    print("=" * 60)
    print("📊 Summary: Fine-tuning Results")
    print("=" * 60)
    print("Even with just 1.25 minutes of training, you should see:")
    print("✅ Better semantic understanding of academic language")
    print("✅ Improved matching between titles and abstracts")
    print("✅ Domain-specific vocabulary recognition")
    print("✅ More relevant similarity scores for research papers")
    
    return base_model, finetuned_model


if __name__ == "__main__":
    try:
        base_model, finetuned_model = test_model_comparison()
        print("\n🎉 Model comparison completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 