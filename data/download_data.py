from datasets import load_dataset
import json
import random
import yaml
import os
from tqdm import tqdm

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def download_and_process_dataset(config):
    """
    Download and process the s2orc_small dataset.
    Merges title and paperAbstract into text field.
    Extracts year, venue, fieldsOfStudy, and authors as metadata.
    Skips entries with missing or null paperAbstract.
    """
    dataset_name = config["dataset"]["name"]
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    
    processed_documents = []
    skipped_count = 0
    
    print("Processing documents...")
    for entry in tqdm(dataset, desc="Processing papers", unit="papers"):
        # Skip entries with missing or null paperAbstract
        if not entry.get("paperAbstract"):
            skipped_count += 1
            continue
        
        # Merge title and paperAbstract into text field
        title = entry.get("title", "")
        abstract = entry.get("paperAbstract", "")
        text = f"{title} {abstract}".strip()
        
        # Create processed document, 
        # Meant for metadata boosting, currently only works for the default dataset
        doc = {
            "id": entry.get("paperId"),
            "text": text,
            "metadata": {
                "year": entry.get("year"),
                "venue": entry.get("venue"),
                "fieldsOfStudy": entry.get("fieldsOfStudy", []),
                "authors": entry.get("authors", [])
            }
        }
        
        processed_documents.append(doc)
    
    print(f"Processing complete!")
    print(f"Processed: {len(processed_documents)} documents")
    print(f"Skipped: {skipped_count} documents (missing abstract)")
    
    return processed_documents

def split_dataset(documents, train_ratio=0.8, random_seed=42):
    """
    Split documents into train and test sets.
    
    Args:
        documents (list): List of processed documents
        train_ratio (float): Ratio for training set (default: 0.8)
        random_seed (int): Random seed for reproducible splits
        
    Returns:
        tuple: (train_docs, test_docs)
    """
    print(f"\nSplitting dataset with {train_ratio:.1%} for training...")
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    
    # Shuffle documents
    shuffled_docs = documents.copy()
    random.shuffle(shuffled_docs)
    
    # Calculate split point
    split_point = int(len(shuffled_docs) * train_ratio)
    
    # Split into train and test
    train_docs = shuffled_docs[:split_point]
    test_docs = shuffled_docs[split_point:]
    
    print(f"Train set: {len(train_docs)} documents ({len(train_docs)/len(documents):.1%})")
    print(f"Test set: {len(test_docs)} documents ({len(test_docs)/len(documents):.1%})")
    
    return train_docs, test_docs

def save_split_datasets(train_docs, test_docs, config):
    """
    Save train and test datasets to separate JSONL files.
    
    Args:
        train_docs (list): Training documents
        test_docs (list): Test documents  
        config (dict): Configuration dictionary
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save training set
    train_path = "data/train_docs.jsonl"
    print(f"\nSaving training set to: {train_path}")
    with open(train_path, "w") as f:
        for doc in tqdm(train_docs, desc="Saving train docs", unit="docs"):
            json.dump(doc, f)
            f.write("\n")
    
    # Save test set
    test_path = "data/test_docs.jsonl"
    print(f"Saving test set to: {test_path}")
    with open(test_path, "w") as f:
        for doc in tqdm(test_docs, desc="Saving test docs", unit="docs"):
            json.dump(doc, f)
            f.write("\n")
    
    # Also save full dataset for backward compatibility
    full_path = config["dataset"]["processed_path"]
    print(f"Saving full dataset to: {full_path}")
    with open(full_path, "w") as f:
        for doc in tqdm(train_docs + test_docs, desc="Saving full dataset", unit="docs"):
            json.dump(doc, f)
            f.write("\n")
    
    print("\nDataset splitting complete!")
    print(f"Files created:")
    print(f"  - {train_path} ({len(train_docs)} documents)")
    print(f"  - {test_path} ({len(test_docs)} documents)")
    print(f"  - {full_path} ({len(train_docs + test_docs)} documents)")

def main():
    """Main function to download, process, and split dataset."""
    print("=" * 60)
    print("Dataset Download and Split")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Download and process dataset
    documents = download_and_process_dataset(config)
    
    # Split into train/test
    train_ratio = config["dataset"]["train_ratio"]
    train_docs, test_docs = split_dataset(documents, train_ratio=train_ratio)
    
    # Save split datasets
    save_split_datasets(train_docs, test_docs, config)
    
    print("\n" + "=" * 60)
    print("Ready for next steps:")
    print("1. Generate embeddings: python embeddings/generate_embeddings.py")
    print("2. Run evaluation: python eval/beir_evaluation.py")
    print("3. Fine-tune model: python finetune/train_query_encoder.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
