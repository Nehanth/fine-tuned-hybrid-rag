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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.metadata_boosting import compute_boost


class WeightLearner(nn.Module):
    """
    Neural network that learns optimal combination weights for hybrid retrieval using contrastive learning.
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
    

class ContrastiveWeightDataset(Dataset):
    """Dataset for contrastive learning of combination weights."""
    
    def __init__(self, positive_examples):
        self.positive_examples = positive_examples
    
    def __len__(self):
        return len(self.positive_examples)
    
    def __getitem__(self, idx):
        pos_example = self.positive_examples[idx]
        
        return (
            # Positive example scores
            torch.tensor(pos_example['pos_dense'], dtype=torch.float32),
            torch.tensor(pos_example['pos_sparse'], dtype=torch.float32), 
            torch.tensor(pos_example['pos_boost'], dtype=torch.float32),
            # Negative example scores (pre-computed)
            torch.tensor(pos_example['neg_dense'], dtype=torch.float32),
            torch.tensor(pos_example['neg_sparse'], dtype=torch.float32), 
            torch.tensor(pos_example['neg_boost'], dtype=torch.float32)
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


def generate_realistic_filters(metadata):
    """Generate realistic user filters for training based on document metadata."""
    filters = {}
    
    # 80% chance of some kind of filter (more aggressive for training)
    if random.random() < 0.8:
        # Try multiple filter types to increase chances of success
        
        # Field filters (try first)
        if metadata.get('fieldsOfStudy'):
            fields = metadata.get('fieldsOfStudy', [])
            if fields and isinstance(fields, list) and len(fields) > 0:
                field = random.choice(fields)
                if field and isinstance(field, str) and len(field.strip()) > 0:
                    filters['field'] = field.strip()
        
        # Venue filters (try if no field filter)
        if not filters and metadata.get('venue'):
            venue = metadata.get('venue', '')
            if venue and isinstance(venue, str) and len(venue.strip()) > 0:
                filters['venue'] = venue.strip()
        
        # Year filters (try if no other filters)
        if not filters and metadata.get('year'):
            doc_year = metadata.get('year')
            if doc_year and isinstance(doc_year, (int, float)):
                year = int(doc_year)
                if 1950 <= year <= 2023:  # Reasonable year range
                    # Prefer recent papers (80% of time)
                    if random.random() < 0.8:
                        filters['year_after'] = max(1990, year - random.randint(3, 10))
                    else:
                        filters['year_before'] = min(2023, year + random.randint(1, 5))
        
        # Author filters (fallback)
        if not filters and metadata.get('authors'):
            authors = metadata.get('authors', [])
            if authors and isinstance(authors, list):
                for author in authors:
                    if isinstance(author, dict) and 'name' in author:
                        name = author['name']
                        if isinstance(name, str) and ' ' in name:
                            filters['author'] = name.split()[-1].strip()  # Last name
                            break
    
    return filters


def create_contrastive_training_data():
    """
    Create contrastive training data using only positive pairs.
    
    For each positive query-document pair, we:
    1. Compute scores for the positive document
    2. Sample random documents as negatives
    
    Returns:
        List of positive examples with pre-computed negative scores
    """
    print("Creating contrastive training data for weight learning...")
    
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
    text_to_metadata = {doc['text']: doc['metadata'] for doc in documents}
    
    # Load existing training pairs (only positives needed!)
    print("Loading fusion training pairs...")
    with open("finetune/fusion_training_pairs.json", 'r') as f:
        training_pairs = json.load(f)
    
    print(f"Loaded {len(training_pairs)} positive pairs")
    
    positive_examples = []
    
    # Use all pairs for training
    print(f"Processing {len(training_pairs)} pairs for contrastive weight training...")
    
    for pair in tqdm(training_pairs, desc="Creating contrastive training data"):
        query = pair['query']
        pos_doc_text = pair['document']
        
        pos_doc_metadata = text_to_metadata.get(pos_doc_text)
        
        if pos_doc_metadata is None:
            continue
        
        # Pre-compute scores for positive example
        try:
            pos_dense = compute_dense_similarity(query, pos_doc_text, dense_model)
            pos_sparse = compute_sparse_similarity(query, pos_doc_text, tfidf_vectorizer)
            
            # Generate realistic filters and compute boost for positive doc
            pos_filters = generate_realistic_filters(pos_doc_metadata)
            pos_boost_raw = compute_boost(pos_doc_metadata, pos_filters)
            pos_boost = (pos_boost_raw - 1.0) / 0.5  # Normalize 1.0-1.5 → 0.0-1.0
            
            # Pre-compute scores for a random negative (for efficiency)
            neg_doc = random.choice(documents)
            while neg_doc['text'] == pos_doc_text:  # Ensure different document
                neg_doc = random.choice(documents)
                
            neg_dense = compute_dense_similarity(query, neg_doc['text'], dense_model)
            neg_sparse = compute_sparse_similarity(query, neg_doc['text'], tfidf_vectorizer)
            
            # Generate different filters for negative doc (realistic scenario)
            neg_filters = generate_realistic_filters(neg_doc['metadata'])
            neg_boost_raw = compute_boost(neg_doc['metadata'], neg_filters)
            neg_boost = (neg_boost_raw - 1.0) / 0.5  # Normalize 1.0-1.5 → 0.0-1.0, # one hot encoding
            
            positive_examples.append({
                'query': query,
                'pos_dense': float(pos_dense),
                'pos_sparse': float(pos_sparse),
                'pos_boost': float(pos_boost),
                'neg_dense': float(neg_dense),
                'neg_sparse': float(neg_sparse),
                'neg_boost': float(neg_boost)
            })
            
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue
    
    print(f"Created {len(positive_examples)} positive examples for contrastive learning")
    
    return positive_examples


def contrastive_loss(pos_scores, neg_scores, margin=0.2):
    """
    Contrastive loss: maximize positive scores, minimize negative scores.
    
    Args:
        pos_scores: Combined scores for positive examples
        neg_scores: Combined scores for negative examples  
        margin: Margin between positive and negative scores
    """
    # We want pos_scores to be high and neg_scores to be low
    # Loss = max(0, margin - (pos_scores - neg_scores))
    return torch.mean(torch.clamp(margin - (pos_scores - neg_scores), min=0.0))


def train_weights():
    """
    Main training function for contrastive weight learning.
    
    This function:
    1. Creates contrastive training data using only positive pairs
    2. Trains the WeightLearner model with contrastive loss
    3. Saves the learned weights
    """
    print("=" * 60)
    print("CONTRASTIVE AUTO-WEIGHTAGE TRAINING")
    print("Learning Optimal Combination Weights")
    print("=" * 60)
    
    # Create contrastive training data
    positive_examples = create_contrastive_training_data()
    
    if len(positive_examples) == 0:
        print("No training examples created. Check your data.")
        return
    
    # Load config for neural fusion settings
    config = load_config()
    fusion_config = config["neural_fusion"]
    
    dataset = ContrastiveWeightDataset(positive_examples)
    dataloader = DataLoader(dataset, batch_size=fusion_config["batch_size"], shuffle=True)
    
    print(f"Training dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize weight learner with default starting weights
    initial_weights = [0.6, 0.3, 0.1]  # Starting values: dense, sparse, boost
    
    weight_learner = WeightLearner(initial_weights=initial_weights)
    optimizer = torch.optim.Adam(weight_learner.parameters(), lr=fusion_config["learning_rate"], weight_decay=1e-5)
    
    print(f"Contrastive Learning Config:")
    print(f"Learning rate: {fusion_config['learning_rate']}")
    print(f"Epochs: {fusion_config['epochs']}")
    print(f"Batch size: {fusion_config['batch_size']}")
    print(f"Starting weights: dense={initial_weights[0]}, sparse={initial_weights[1]}, boost={initial_weights[2]}")
    print()
    
    # Training loop
    num_epochs = fusion_config["epochs"]
    
    weight_learner.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            pos_dense, pos_sparse, pos_boost, neg_dense, neg_sparse, neg_boost = batch
            
            # Forward pass for positive and negative examples
            pos_scores = weight_learner(pos_dense, pos_sparse, pos_boost)
            neg_scores = weight_learner(neg_dense, neg_sparse, neg_boost)
            
            # Contrastive loss: maximize pos, minimize neg
            loss = contrastive_loss(pos_scores, neg_scores, margin=0.2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_weights = weight_learner.get_weights()
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | "
                  f"Weights: dense={current_weights[0]:.3f}, sparse={current_weights[1]:.3f}, boost={current_weights[2]:.3f}")
    
    # Save final weights
    torch.save(weight_learner.state_dict(), "finetune/learned_weights.pth")
    weight_learner.eval()
    
    # Print final results
    final_weights = weight_learner.get_weights()
    print("\n" + "=" * 60)
    print("CONTRASTIVE AUTO-WEIGHTAGE TRAINING COMPLETE!")
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