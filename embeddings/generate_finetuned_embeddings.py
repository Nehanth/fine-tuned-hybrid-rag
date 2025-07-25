#!/usr/bin/env python3
"""
Generate Fine-tuned Dense Embeddings with new model
"""

import json
import numpy as np
import yaml
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_documents(file_path):
    """Load documents from JSONL file."""
    print(f"Loading documents from {file_path}...")
    documents = []
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading documents"):
            doc = json.loads(line.strip())
            documents.append(doc['text'])
    
    print(f"Loaded {len(documents)} documents")
    return documents


def load_finetuned_model(model_path, device):
    """Load the fine-tuned sentence transformer model."""
    print(f"Loading fine-tuned sentence transformer model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at: {model_path}")
    
    # Handle device configuration
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(model_path, device=device)
    print(f"Using device: {model.device}")
    
    return model


def generate_finetuned_dense_embeddings(documents, model, config):
    """Generate embeddings using the fine-tuned model."""
    batch_size = config["embeddings"]["batch_size"]
    normalize = config["embeddings"]["normalize_embeddings"]
    
    print("Generating embeddings with fine-tuned model...")
    print(f"Processing {len(documents)} documents...")
    print(f"Example text length: {len(documents[0])} characters")
    
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def save_finetuned_embeddings(embeddings, output_path):
    """Save fine-tuned embeddings to disk."""
    print(f"Saving embeddings to {output_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as numpy array
    np.save(output_path, embeddings)
    
    # Get file size
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Embeddings saved successfully!")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")


def main():
    """Main function to generate and save fine-tuned embeddings."""
    print("=" * 60)
    print("Fine-tuned Embedding Generation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Get paths from config
    model_path = config["finetune"]["output_path"]
    documents_path = config["dataset"]["processed_path"]
    output_path = config["embeddings"]["dense_finetuned_path"]
    device = config["model"]["device"]
    
    # Load documents
    documents = load_documents(documents_path)
    
    # Load fine-tuned model
    model = load_finetuned_model(model_path, device)
    
    # Generate embeddings
    embeddings = generate_finetuned_dense_embeddings(documents, model, config)
    
    # Save embeddings
    save_finetuned_embeddings(embeddings, output_path)
    
    print("=" * 60)
    print("Fine-tuned embedding generation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
