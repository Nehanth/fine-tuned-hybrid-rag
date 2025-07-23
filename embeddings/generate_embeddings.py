import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm

def load_documents(file_path="data/processed_docs.jsonl"):
    """Load documents from JSONL file and return list of texts and IDs."""
    documents = []
    doc_ids = []
    
    print(f"Loading documents from {file_path}...")
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Loading documents"):
            doc = json.loads(line.strip())
            documents.append(doc["text"])
            doc_ids.append(doc["id"])
    
    print(f"Loaded {len(documents)} documents")
    return documents, doc_ids

def generate_dense_embeddings(documents, model_name="all-MiniLM-L6-v2"):
    """Generate dense embeddings using sentence transformers with GPU acceleration."""
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    print("Generating dense embeddings...")
    # Use GPU-optimized settings
    embeddings = model.encode(
        documents, 
        show_progress_bar=True,
        batch_size=32 if device == "cuda" else 16,  # Larger batch for GPU
        convert_to_numpy=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def generate_sparse_embeddings(documents, max_features=10000):
    """Generate sparse TF-IDF embeddings."""
    print(f"Generating TF-IDF embeddings with max_features={max_features}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)  # Include bigrams for better semantic matching
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"Generated TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix

def save_embeddings(dense_embeddings, tfidf_vectorizer, doc_ids):
    """Save embeddings and vectorizer to disk."""
    print("Saving embeddings...")
    
    # Save dense embeddings
    np.save("embeddings/dense.npy", dense_embeddings)
    print("Saved dense embeddings to: embeddings/dense.npy")
    
    # Save TF-IDF vectorizer
    joblib.dump(tfidf_vectorizer, "embeddings/tfidf_vectorizer.pkl")
    print("Saved TF-IDF vectorizer to: embeddings/tfidf_vectorizer.pkl")
    
    print("Embeddings generation complete!")

def main():
    """Main function to generate and save embeddings."""
    # Load documents
    documents, doc_ids = load_documents()
    
    # Generate dense embeddings with GPU acceleration
    dense_embeddings = generate_dense_embeddings(documents)
    
    # Generate sparse embeddings
    tfidf_vectorizer, tfidf_matrix = generate_sparse_embeddings(documents)
    
    # Save embeddings
    save_embeddings(dense_embeddings, tfidf_vectorizer, doc_ids)

if __name__ == "__main__":
    main() 