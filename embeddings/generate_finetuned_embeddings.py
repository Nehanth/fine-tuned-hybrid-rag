#!/usr/bin/env python3
"""
Generate Fine-tuned Embeddings

Re-embed all documents using the fine-tuned model to create a proper
comparison where queries and documents use the same embedding space.
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
            documents.append(doc)
    
    print(f"Loaded {len(documents)} documents")
    return documents


def generate_finetuned_embeddings():
    """Generate embeddings using the fine-tuned model."""
    print("Starting fine-tuned embedding generation...")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Get paths from config
    model_path = config["finetune"]["output_path"]
    documents_path = config["dataset"]["processed_path"]
    output_path = config["embeddings"]["dense_finetuned_path"]
    batch_size = config["embeddings"]["batch_size"]
    normalize = config["embeddings"]["normalize_embeddings"]
    device = config["model"]["device"]
    
    # Handle device configuration
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the fine-tuned model
    print(f"Loading fine-tuned sentence transformer model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at: {model_path}")
    
    model = SentenceTransformer(model_path, device=device)
    print(f"Using device: {model.device}")
    
    # Load documents
    documents = load_documents(documents_path)
    
    # Extract text from documents
    print("Extracting text from documents...")
    texts = [doc['text'] for doc in documents]
    
    print(f"Processing {len(texts)} documents...")
    print(f"Example text length: {len(texts[0])} characters")
    
    # Generate embeddings in batches
    print("Generating embeddings with fine-tuned model...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings
    print(f"Saving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    
    # Get file size
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"Embeddings saved successfully!")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    
    return embeddings


if __name__ == "__main__":
    try:
        embeddings = generate_finetuned_embeddings()
        print("\n" + "=" * 50)
        print("Fine-tuned embedding generation completed successfully!")
        print("You can now use these embeddings for consistent evaluation.")
        
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        import traceback
        traceback.print_exc() 