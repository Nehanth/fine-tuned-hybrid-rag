"""
Download and Process S2ORC Dataset

This script downloads the S2ORC academic papers dataset and processes it into
a JSON format for use in the hybrid RAG retrieval system.
"""

import json
import yaml
from datasets import load_dataset
from tqdm import tqdm


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_and_process_dataset(config: dict) -> list:
    """
    Download and process the dataset.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        list: List of processed documents
    """
    dataset_name = config["dataset"]["name"]
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    print("Processing documents...")
    processed_docs = []
    skipped = 0
    
    for i, doc in enumerate(tqdm(dataset, desc="Processing documents")):
        # Extract required fields
        title = doc.get('title', '').strip()
        abstract = doc.get('paperAbstract', '').strip()
        
        # Skip documents without title or abstract
        if not title or not abstract:
            skipped += 1
            continue
        
        # Create processed document with metadata schema of the dataset (Current: S2ORC)
        processed_doc = {
            'id': i,
            'text': f"{title} {abstract}",
            'metadata': {
                'title': title,
                'abstract': abstract,
                'year': doc.get('year'),
                'venue': doc.get('venue'),
                'fieldsOfStudy': doc.get('fieldsOfStudy', []),
                'authors': doc.get('authors', [])
            }
        }
        
        processed_docs.append(processed_doc)
    
    print(f"Processed {len(processed_docs)} documents")
    print(f"Skipped {skipped} documents (missing title or abstract)")
    
    return processed_docs


def save_processed_dataset(documents: list, output_path: str):
    """
    Save processed documents to JSONL file.
    
    Args:
        documents (list): List of processed documents
        output_path (str): Path to save the processed dataset
    """
    print(f"Saving processed dataset to {output_path}...")
    
    with open(output_path, 'w') as f:
        for doc in tqdm(documents, desc="Saving documents"):
            json.dump(doc, f)
            f.write('\n')
    
    print(f"Saved {len(documents)} documents to {output_path}")


def main():
    """
    Main function to download, process, and save the dataset.
    """
    print("=" * 60)
    print("S2ORC Dataset Download and Processing")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Download and process dataset
    documents = download_and_process_dataset(config)
    
    # Save processed dataset
    output_path = config["dataset"]["processed_path"]
    save_processed_dataset(documents, output_path)
    
    print("=" * 60)
    print("Dataset processing completed!")
    print(f"Total documents: {len(documents)}")
    print(f"Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
