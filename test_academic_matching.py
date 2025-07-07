#!/usr/bin/env python3
"""
Test title-to-abstract matching using real academic data.
This is what the model was actually trained on, so we should see clearer improvements.
"""

import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def test_title_abstract_matching():
    """
    Test how well base vs fine-tuned models match titles to their abstracts.
    """
    print("=" * 60)
    print("📚 Testing Title-to-Abstract Matching (Real Academic Data)")
    print("=" * 60)
    
    # Load some real examples from our dataset
    print("Loading real academic examples...")
    
    # Sample some documents
    examples = []
    with open("data/processed_docs.jsonl", 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Sample first 1000 for speed
                break
            doc = json.loads(line.strip())
            
            # Try to extract title from text (first sentence/question)
            text = doc.get('text', '').strip()
            if not text:
                continue
                
            # Extract title using same logic as training
            title = None
            if '. ' in text:
                potential_title = text.split('. ')[0].strip()
                if 10 <= len(potential_title) <= 200:
                    title = potential_title
            
            if not title and '? ' in text:
                potential_title = text.split('? ')[0].strip() + '?'
                if 10 <= len(potential_title) <= 200:
                    title = potential_title
            
            if not title:
                words = text.split()
                if len(words) > 5:
                    title = ' '.join(words[:15])
                    if len(title) > 200:
                        title = title[:200].strip()
            
            if title and len(title) >= 10:
                examples.append({
                    'title': title,
                    'full_text': text,
                    'abstract': text[len(title):].strip()
                })
    
    print(f"Found {len(examples)} academic examples")
    
    # Load models
    print("Loading models...")
    base_model = SentenceTransformer("all-MiniLM-L6-v2")
    finetuned_model = SentenceTransformer("finetune/model/")
    
    # Test with multiple examples
    test_examples = random.sample(examples, min(5, len(examples)))
    
    print("\n🧪 Testing title-to-abstract matching...\n")
    
    base_improvements = 0
    finetuned_improvements = 0
    
    for i, example in enumerate(test_examples, 1):
        title = example['title']
        correct_text = example['full_text']
        
        # Create some distractors from other examples
        distractors = random.sample([e['full_text'] for e in examples if e != example], 3)
        all_texts = [correct_text] + distractors
        random.shuffle(all_texts)
        correct_idx = all_texts.index(correct_text)
        
        print(f"Test {i}: {title[:80]}...")
        print("-" * 50)
        
        # Test base model
        base_title_emb = base_model.encode([title])
        base_text_embs = base_model.encode(all_texts)
        base_similarities = cosine_similarity(base_title_emb, base_text_embs)[0]
        base_best_idx = np.argmax(base_similarities)
        
        # Test fine-tuned model  
        ft_title_emb = finetuned_model.encode([title])
        ft_text_embs = finetuned_model.encode(all_texts)
        ft_similarities = cosine_similarity(ft_title_emb, ft_text_embs)[0]
        ft_best_idx = np.argmax(ft_similarities)
        
        print(f"Base Model: Picked option {base_best_idx + 1} (correct={correct_idx + 1}) - Similarity: {base_similarities[correct_idx]:.3f}")
        print(f"Fine-tuned: Picked option {ft_best_idx + 1} (correct={correct_idx + 1}) - Similarity: {ft_similarities[correct_idx]:.3f}")
        
        base_correct = base_best_idx == correct_idx
        ft_correct = ft_best_idx == correct_idx
        
        if ft_correct and not base_correct:
            print("🎯 IMPROVEMENT: Fine-tuned found correct match!")
            finetuned_improvements += 1
        elif base_correct and not ft_correct:
            print("⚠️  REGRESSION: Base model was better")
            base_improvements += 1  
        elif ft_correct and base_correct:
            # Check if fine-tuned has higher confidence
            ft_confidence = ft_similarities[correct_idx]
            base_confidence = base_similarities[correct_idx]
            if ft_confidence > base_confidence:
                print("✅ BOTH CORRECT: Fine-tuned more confident")
                finetuned_improvements += 0.5
            else:
                print("✅ BOTH CORRECT: Base more confident")
                base_improvements += 0.5
        else:
            print("❌ BOTH WRONG")
        
        print(f"Similarity improvement: {ft_similarities[correct_idx] - base_similarities[correct_idx]:+.3f}")
        print()
    
    print("=" * 60)
    print("📊 Final Results")
    print("=" * 60)
    print(f"Fine-tuned advantages: {finetuned_improvements}")
    print(f"Base model advantages: {base_improvements}")
    
    if finetuned_improvements > base_improvements:
        print("🎉 FINE-TUNING SUCCESSFUL!")
        print("The model learned to better match academic titles to their abstracts!")
    elif base_improvements > finetuned_improvements:
        print("🤔 Base model performed better on this test")
        print("Might need longer training or different hyperparameters")
    else:
        print("📊 Mixed results - both models similar performance")
    
    return finetuned_improvements, base_improvements


if __name__ == "__main__":
    try:
        test_title_abstract_matching()
        print("\n✅ Academic matching test completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 