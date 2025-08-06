import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import random
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.metadata_boosting import compute_boost



class WeightLearner(nn.Module):
    """
    Neural network that learns optimal combination weights for hybrid retrieval.
    
    Instead of fixed weights (0.6 dense, 0.3 sparse, 0.1 boost), this model
    learns the optimal weights from training data.
    """
    
    def __init__(self, initial_weights=[0.6, 0.3, 0.1]):
        super().__init__()
        
        self.raw_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
    
    def forward(self, dense_scores, sparse_scores, boost_scores):
        """
        Combine the three types of scores with learned weights.
        
        Args:
            dense_scores: Semantic similarity scores
            sparse_scores: TF-IDF similarity scores  
            boost_scores: Metadata boost scores
            
        Returns:
            Combined final scores
        """
        # Normalize weights to sum to 1 (softmax ensures this)
        weights = F.softmax(self.raw_weights, dim=0)
        
        # Weighted combination
        combined_scores = (weights[0] * dense_scores + 
                          weights[1] * sparse_scores + 
                          weights[2] * boost_scores)
        
        return combined_scores
    
    def get_weights(self):
        """Get current learned weights as numpy array."""
        with torch.no_grad():
            return F.softmax(self.raw_weights, dim=0).numpy()
    
    def print_weights(self):
        """Print current learned weights in a readable format."""
        weights = self.get_weights()
        print(f"Dense: {weights[0]:.3f}, Sparse: {weights[1]:.3f}, Boost: {weights[2]:.3f}")


class WeightTrainingDataset(Dataset):
    """Dataset for training combination weights."""
    
    def __init__(self, training_examples):
        self.examples = training_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            torch.tensor(example['dense_score'], dtype=torch.float32),
            torch.tensor(example['sparse_score'], dtype=torch.float32), 
            torch.tensor(example['boost_score'], dtype=torch.float32),
            torch.tensor(example['label'], dtype=torch.float32)
        )


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def compute_dense_similarity(query, doc_text, model):
    """Compute dense similarity using frozen fine-tuned model."""
    with torch.no_grad():
        query_emb = model.encode([query])
        doc_emb = model.encode([doc_text])
        return cosine_similarity(query_emb, doc_emb)[0][0]


def compute_sparse_similarity(query, doc_text, vectorizer):
    """Compute TF-IDF similarity using static vectorizer."""
    query_vec = vectorizer.transform([query])
    doc_vec = vectorizer.transform([doc_text])
    return cosine_similarity(query_vec, doc_vec)[0][0]


def generate_training_filters(metadata):
    """Generate realistic user filters for training (similar to evaluation)."""
    filters = {}
    
    # Randomly add different types of filters to simulate user preferences
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
                name = author['name']
                if ' ' in name:
                    filters['author'] = name.split()[-1]
    
    return filters


def create_training_data():
    """
    Create training data for weight learning from existing realistic pairs.
    
    For each query-document pair, we:
    1. Compute individual dense, sparse, and boost scores
    2. Create positive examples (relevant docs, label=1)
    3. Create negative examples (irrelevant docs, label=0)
    
    Returns:
        List of training examples with scores and labels
    """
    print("Creating training data for weight learning...")
    
    # Load config and components
    config = load_config()
    
    # Load trained components (all frozen - we don't retrain these)
    print("Loading frozen dense model...")
    dense_model = SentenceTransformer(config["finetune"]["output_path"])
    dense_model.eval()
    
    print("Loading static TF-IDF vectorizer...")
    tfidf_vectorizer = joblib.load(config["embeddings"]["tfidf_vectorizer_path"])
    
    print("Loading documents...")
    documents = []
    with open(config["dataset"]["processed_path"], 'r') as f:
        for line in f:
            documents.append(json.loads(line.strip()))
    
    print(f"Loaded {len(documents)} documents")
    
    # Create efficient text-to-metadata mapping
    print("Creating text-to-metadata mapping...")
    text_to_metadata = {doc['text']: doc['metadata'] for doc in documents}
    
    # Load existing training pairs
    print("Loading fusion training pairs...")
    with open("finetune/fusion_training_pairs.json", 'r') as f:
        training_pairs = json.load(f)
    
    # Load curated negative examples
    print("Loading fusion negative pairs...")
    with open("finetune/fusion_negative_pairs.json", 'r') as f:
        negative_pairs = json.load(f)
    
    print(f"Loaded {len(training_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
    
    training_examples = []
    
    # Use subset for faster training (adjust as needed)
    max_pairs = min(3000, len(training_pairs))
    selected_pairs = random.sample(training_pairs, max_pairs)
    
    # Create query-to-negative mapping for efficiency
    query_to_negatives = {}
    for neg_pair in negative_pairs:
        query = neg_pair['query']
        if query not in query_to_negatives:
            query_to_negatives[query] = []
        query_to_negatives[query].append(neg_pair['document'])
    
    print(f"Processing {max_pairs} pairs for weight training...")
    
    for pair in tqdm(selected_pairs, desc="Creating weight training data"):
        query = pair['query']
        pos_doc_text = pair['document']
        

        pos_doc_metadata = text_to_metadata.get(pos_doc_text)
        
        if pos_doc_metadata is None:
            continue
            
        # Get curated negative examples for this specific query
        neg_texts = query_to_negatives[query][:2]
        neg_docs = []
        for neg_text in neg_texts:
            neg_metadata = text_to_metadata.get(neg_text)
            if neg_metadata:
                neg_docs.append({'text': neg_text, 'metadata': neg_metadata})
        
        # Process positive example
        try:
            pos_dense = compute_dense_similarity(query, pos_doc_text, dense_model)
            pos_sparse = compute_sparse_similarity(query, pos_doc_text, tfidf_vectorizer)
            # Generate realistic user filters for training (like in evaluation)
            pos_boost_raw = compute_boost(pos_doc_metadata, generate_training_filters(pos_doc_metadata))
            pos_boost = (pos_boost_raw - 1.0) / 0.5  # Normalize 1.0-1.5 → 0.0-1.0
            
            training_examples.append({
                'dense_score': float(pos_dense),
                'sparse_score': float(pos_sparse), 
                'boost_score': float(pos_boost),
                'label': 1.0  # Positive example
            })
            
            # Process negative examples
            for neg_doc in neg_docs:
                neg_dense = compute_dense_similarity(query, neg_doc['text'], dense_model)
                neg_sparse = compute_sparse_similarity(query, neg_doc['text'], tfidf_vectorizer)
                neg_boost_raw = compute_boost(neg_doc['metadata'], generate_training_filters(neg_doc['metadata']))
                neg_boost = (neg_boost_raw - 1.0) / 0.5  # Normalize 1.0-1.5 → 0.0-1.0
                
                training_examples.append({
                    'dense_score': float(neg_dense),
                    'sparse_score': float(neg_sparse),
                    'boost_score': float(neg_boost), 
                    'label': 0.0  # Negative example
                })
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue
    
    print(f"Created {len(training_examples)} training examples")
    
    # Show some stats
    pos_examples = sum(1 for ex in training_examples if ex['label'] == 1.0)
    neg_examples = len(training_examples) - pos_examples
    print(f"Positive examples: {pos_examples}")
    print(f"Negative examples: {neg_examples}")
    
    return training_examples


def train_weights():
    """
    Main training function for weight learning.
    
    This function:
    1. Creates training data from existing realistic pairs
    2. Trains the WeightLearner model
    3. Saves the learned weights
    """
    print("=" * 60)
    print("AUTO-WEIGHTAGE TRAINING")
    print("Learning Optimal Combination Weights")
    print("=" * 60)
    
    # Create training data
    training_examples = create_training_data()
    
    if len(training_examples) == 0:
        print("No training examples created. Check your data.")
        return
    
    # Load config for neural fusion settings
    config = load_config()
    fusion_config = config["neural_fusion"]
    
    dataset = WeightTrainingDataset(training_examples)
    dataloader = DataLoader(dataset, batch_size=fusion_config["batch_size"], shuffle=True)
    
    print(f"Training dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize weight learner with default starting weights
    initial_weights = [0.6, 0.3, 0.1]  # Starting values: dense, sparse, boost
    
    weight_learner = WeightLearner(initial_weights=initial_weights)
    optimizer = torch.optim.Adam(weight_learner.parameters(), lr=fusion_config["learning_rate"], weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Neural Fusion Training Config:")
    print(f"Learning rate: {fusion_config['learning_rate']}")
    print(f"Epochs: {fusion_config['epochs']}")
    print(f"Batch size: {fusion_config['batch_size']}")
    print(f"Starting weights: dense={initial_weights[0]}, sparse={initial_weights[1]}, boost={initial_weights[2]}")
    print()
    
    # Training loop
    num_epochs = fusion_config["epochs"]
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    weight_learner.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            dense_scores, sparse_scores, boost_scores, labels = batch
            
            # Forward pass
            combined_scores = weight_learner(dense_scores, sparse_scores, boost_scores)
            loss = criterion(combined_scores, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(weight_learner.state_dict(), "finetune/learned_weights.pth")
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_weights = weight_learner.get_weights()
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Weights: dense={current_weights[0]:.3f}, sparse={current_weights[1]:.3f}, boost={current_weights[2]:.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best weights
    weight_learner.load_state_dict(torch.load("finetune/learned_weights.pth"))
    weight_learner.eval()
    
    # Print final results
    final_weights = weight_learner.get_weights()
    print("\n" + "=" * 60)
    print("AUTO-WEIGHTAGE TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Original weights: dense={initial_weights[0]:.3f}, sparse={initial_weights[1]:.3f}, boost={initial_weights[2]:.3f}")
    print(f"Learned weights:  dense={final_weights[0]:.3f}, sparse={final_weights[1]:.3f}, boost={final_weights[2]:.3f}")
    
    # Calculate changes
    dense_change = final_weights[0] - initial_weights[0]
    sparse_change = final_weights[1] - initial_weights[1] 
    boost_change = final_weights[2] - initial_weights[2]
    
    print("\nChanges:")
    print(f"   Dense:  {initial_weights[0]:.3f} → {final_weights[0]:.3f} ({dense_change:+.3f})")
    print(f"   Sparse: {initial_weights[1]:.3f} → {final_weights[1]:.3f} ({sparse_change:+.3f})")
    print(f"   Boost:  {initial_weights[2]:.3f} → {final_weights[2]:.3f} ({boost_change:+.3f})")
    
    print(f"\nSaved learned weights to: finetune/learned_weights.pth")
    print("=" * 60)
    
    return weight_learner


if __name__ == "__main__":
    train_weights()