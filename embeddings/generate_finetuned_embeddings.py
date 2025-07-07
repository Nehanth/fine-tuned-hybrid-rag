#!/usr/bin/env python3
"""
Generate Fine-tuned Embeddings

Re-embed all documents using the fine-tuned model to create a proper
comparison where queries and documents use the same embedding space.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

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
    
    # Load the fine-tuned model
    print("Loading fine-tuned sentence transformer model...")
    model = SentenceTransformer('../finetune/model/')
    print(f"Using device: {model.device}")
    
    # Load documents
    documents = load_documents('../data/processed_docs.jsonl')
    
    # Extract text from documents
    print("Extracting text from documents...")
    texts = [doc['text'] for doc in documents]
    
    print(f"Processing {len(texts)} documents...")
    print(f"Example text length: {len(texts[0])} characters")
    
    # Generate embeddings in batches
    print("Generating embeddings with fine-tuned model...")
    batch_size = 256  # Adjust based on GPU memory
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    output_file = 'dense_finetuned.npy'
    print(f"Saving embeddings to {output_file}...")
    
    # Already in embeddings directory, no need to create it
    
    # Save as numpy array
    np.save(output_file, embeddings)
    
    # Get file size
    file_size = os.path.getsize(output_file)
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