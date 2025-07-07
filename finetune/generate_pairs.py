#!/usr/bin/env python3
"""
Generate training pairs for fine-tuning the MiniLM semantic encoder.

This script loads the original dataset and creates query-document pairs for contrastive learning:
- Query: Clean document title from the original dataset
- Document: Combined title + abstract (same as processed text)
- Output: InputExample objects saved as pickle file

Uses the original dataset to get clean titles instead of extracting from processed text.
"""

import pickle
from typing import List
from sentence_transformers import InputExample
from tqdm import tqdm
from datasets import load_dataset


def load_original_dataset(dataset_name: str = "leminda-ai/s2orc_small"):
    """
    Load the original dataset with separate title and abstract fields.
    
    Args:
        dataset_name (str): Name of the dataset to load
        
    Returns:
        Dataset: Hugging Face dataset with title and paperAbstract fields
    """
    print(f"Loading original dataset: {dataset_name}")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split='train')
    
    # Dataset should support len() for most cases
    print(f"Loaded dataset from {dataset_name}")
    return dataset


def create_training_pairs(dataset) -> List[InputExample]:
    """
    Create training pairs from the original dataset.
    
    For each document:
    - Query: Clean title from dataset
    - Document: Combined title + abstract (same as our processed text)
    
    Args:
        dataset: Hugging Face dataset with title and paperAbstract fields
        
    Returns:
        List[InputExample]: List of training pairs
    """
    print("Creating training pairs from original dataset...")
    pairs = []
    skipped = 0
    
    for doc in tqdm(dataset, desc="Processing documents"):
        # Extract title and abstract from original dataset
        title = doc.get('title', '').strip()
        abstract = doc.get('paperAbstract', '').strip()
        
        # Skip if missing or empty title/abstract
        if not title or not abstract:
            skipped += 1
            continue
        
        # Create combined text (same as our processed data)
        # This matches what we have in processed_docs.jsonl
        combined_text = f"{title} {abstract}"
        
        # Create InputExample for sentence-transformers
        # texts[0] = query (title), texts[1] = document (title + abstract)
        pair = InputExample(texts=[title, combined_text])
        pairs.append(pair)
    
    print(f"Created {len(pairs)} training pairs")
    print(f"Skipped {skipped} documents (missing title or abstract)")
    
    return pairs


def save_pairs(pairs: List[InputExample], output_path: str):
    """
    Save training pairs to pickle file.
    
    Args:
        pairs (List[InputExample]): Training pairs
        output_path (str): Path to save the pairs
    """
    print(f"Saving {len(pairs)} pairs to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(pairs, f)
    
    print(f"Training pairs saved successfully!")


def main():
    """
    Main function to generate training pairs from original dataset.
    """
    # Configuration
    dataset_name = "leminda-ai/s2orc_small"
    output_file = "finetune/pairs.pkl"
    
    print("=" * 60)
    print("Generating Training Pairs from Original Dataset")
    print("=" * 60)
    
    # Load original dataset
    dataset = load_original_dataset(dataset_name)
    
    # Create training pairs
    pairs = create_training_pairs(dataset)
    
    # Save pairs
    save_pairs(pairs, output_file)
    
    print("=" * 60)
    print("Training pair generation completed!")
    print(f"Output: {output_file}")
    print(f"Total pairs: {len(pairs)}")
    print("=" * 60)
    
    # Show a sample pair
    if pairs and len(pairs) > 0:
        print("\nSample pair:")
        sample = pairs[0]
        if sample and sample.texts and len(sample.texts) >= 2:
            print(f"Query: {repr(sample.texts[0])}")
            print(f"Document: {repr(sample.texts[1][:200])}...")


if __name__ == "__main__":
    main() 