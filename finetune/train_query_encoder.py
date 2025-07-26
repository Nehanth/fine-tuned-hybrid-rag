"""
Fine-tune MiniLM semantic encoder using contrastive learning.

This script trains a sentence transformer model using query-document pairs
directly from the dataset to improve retrieval performance.
"""

import os
import yaml
import torch
import torch.nn as nn
import random
from typing import List, Optional
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_training_pairs_from_dataset(config: dict) -> List[InputExample]:
    """
    Create training pairs using text-snippet queries (matches evaluation approach).
    
    Training pairs: text_snippet → (title + abstract)
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        List[InputExample]: List of training pairs
    """
    dataset_name = config["dataset"]["name"]
    max_pairs = config["finetune"]["max_pairs"]
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    print("Creating training pairs using text-snippet queries...")
    
    pairs = []
    skipped = 0
    
    # Process documents with sampling if specified
    total_docs = len(dataset)
    if max_pairs and total_docs > max_pairs:
        print(f"Sampling {max_pairs} documents from {total_docs} for fast training...")
        # Generate random indices for sampling
        sampled_indices = random.sample(range(total_docs), max_pairs)
        sampled_indices_set = set(sampled_indices)
        
        # Process only sampled documents
        for i, doc in enumerate(tqdm(dataset, desc="Processing documents", total=total_docs)):
            if i not in sampled_indices_set:
                continue
            
            # Extract title and abstract
            title = doc.get('title', '').strip()
            abstract = doc.get('paperAbstract', '').strip()
            
            # Skip if missing or empty title/abstract
            if not title or not abstract:
                skipped += 1
                continue
            
            # Create combined document text (same as processed data)
            combined_text = f"{title} {abstract}"
            
            # Create query from document text using middle portion
            text_words = combined_text.split()
            if len(text_words) < 10:
                skipped += 1
                continue
                
            # Use middle portion of text as query to avoid exact matches
            start_idx = len(text_words) // 4
            end_idx = start_idx + min(10, len(text_words) // 2)
            query_words = text_words[start_idx:end_idx]
            query = ' '.join(query_words)
            
            # Add domain-specific terms from ALL available metadata
            metadata_terms = []
            
            # Add field of study
            fields_of_study = doc.get('fieldsOfStudy', [])
            if fields_of_study and len(fields_of_study) > 0:
                main_field = fields_of_study[0] if isinstance(fields_of_study[0], str) else str(fields_of_study[0])
                metadata_terms.append(main_field)
            
            # Add venue information
            venue = doc.get('venue', '')
            if venue and isinstance(venue, str) and len(venue.strip()) > 0:
                # Use venue abbreviation or first word for conciseness
                venue_term = venue.strip().split()[0] if ' ' in venue else venue.strip()
                metadata_terms.append(venue_term)
            
            # Add year information
            year = doc.get('year')
            if year and isinstance(year, (int, float)):
                metadata_terms.append(str(int(year)))
            
            # Add author information (first author's last name)
            authors = doc.get('authors', [])
            if authors and len(authors) > 0:
                first_author = authors[0]
                if isinstance(first_author, dict) and 'name' in first_author:
                    author_name = first_author['name']
                    # Extract last name (assuming "First Last" format)
                    if ' ' in author_name:
                        last_name = author_name.split()[-1]
                        metadata_terms.append(last_name)
                elif isinstance(first_author, str):
                    # Handle string author names
                    if ' ' in first_author:
                        last_name = first_author.split()[-1]
                        metadata_terms.append(last_name)
            
            # Combine metadata terms with query (limit to avoid overly long queries)
            if metadata_terms:
                # Use up to 3 metadata terms to keep queries reasonable
                selected_terms = metadata_terms[:3]
                metadata_prefix = ' '.join(selected_terms)
                query = f"{metadata_prefix} {query}"
            
            # Create training pair: text_snippet → (title + abstract)
            pair = InputExample(texts=[query, combined_text])
            pairs.append(pair)
                    
    else:
        print(f"Processing all {total_docs} documents...")
        for doc in tqdm(dataset, desc="Processing documents", total=total_docs):
            # Extract title and abstract
            title = doc.get('title', '').strip()
            abstract = doc.get('paperAbstract', '').strip()
            
            # Skip if missing or empty title/abstract
            if not title or not abstract:
                skipped += 1
                continue
            
            # Create combined document text (same as processed data)
            combined_text = f"{title} {abstract}"
            
            # Create query from document text using middle portion
            text_words = combined_text.split()
            if len(text_words) < 10:
                skipped += 1
                continue
                
            # Use middle portion of text as query to avoid exact matches
            start_idx = len(text_words) // 4
            end_idx = start_idx + min(10, len(text_words) // 2)
            query_words = text_words[start_idx:end_idx]
            query = ' '.join(query_words)
            
            # Add domain-specific terms from ALL available metadata
            metadata_terms = []
            
            # Add field of study
            fields_of_study = doc.get('fieldsOfStudy', [])
            if fields_of_study and len(fields_of_study) > 0:
                main_field = fields_of_study[0] if isinstance(fields_of_study[0], str) else str(fields_of_study[0])
                metadata_terms.append(main_field)
            
            # Add venue information
            venue = doc.get('venue', '')
            if venue and isinstance(venue, str) and len(venue.strip()) > 0:
                # Use venue abbreviation or first word for conciseness
                venue_term = venue.strip().split()[0] if ' ' in venue else venue.strip()
                metadata_terms.append(venue_term)
            
            # Add year information
            year = doc.get('year')
            if year and isinstance(year, (int, float)):
                metadata_terms.append(str(int(year)))
            
            # Add author information (first author's last name)
            authors = doc.get('authors', [])
            if authors and len(authors) > 0:
                first_author = authors[0]
                if isinstance(first_author, dict) and 'name' in first_author:
                    author_name = first_author['name']
                    # Extract last name (assuming "First Last" format)
                    if ' ' in author_name:
                        last_name = author_name.split()[-1]
                        metadata_terms.append(last_name)
                elif isinstance(first_author, str):
                    # Handle string author names
                    if ' ' in first_author:
                        last_name = first_author.split()[-1]
                        metadata_terms.append(last_name)
            
            # Combine metadata terms with query (limit to avoid overly long queries)
            if metadata_terms:
                # Use up to 3 metadata terms to keep queries reasonable
                selected_terms = metadata_terms[:3]
                metadata_prefix = ' '.join(selected_terms)
                query = f"{metadata_prefix} {query}"
            
            # Create training pair: text_snippet → (title + abstract)
            pair = InputExample(texts=[query, combined_text])
            pairs.append(pair)

    return pairs


def setup_model(config: dict) -> SentenceTransformer:
    """
    Load and setup the sentence transformer model.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        SentenceTransformer: Loaded model
    """
    model_name = config["model"]["encoder"]
    device = config["model"]["device"]
    use_multi_gpu = config["finetune"]["use_multi_gpu"]
    
    print(f"Loading base model: {model_name}")
    
    # Handle device configuration
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SentenceTransformer(model_name, device=device)
    
    # Enable multi-GPU if available
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Enabling multi-GPU training with {torch.cuda.device_count()} GPUs")
        
        # Wrap the internal model with DataParallel
        if hasattr(model, '_modules'):
            for name, module in model._modules.items():
                if hasattr(module, 'auto_model'):
                    print(f"Wrapping {name} module with DataParallel")
                    model._modules[name] = nn.DataParallel(module)
        
        print(f"Multi-GPU setup complete")
    else:
        print(f"Using device: {device}")
    
    print(f"Model max sequence length: {model.get_max_seq_length()}")
    return model


def create_dataloader(pairs: List[InputExample], model: SentenceTransformer, batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader for training pairs.
    
    Args:
        pairs (List[InputExample]): Training pairs
        model (SentenceTransformer): Model for tokenization
        batch_size (int): Batch size
        
    Returns:
        DataLoader: DataLoader for training
    """
    print(f"Creating DataLoader with batch_size={batch_size}")
    
    
    dataset = SentencesDataset(pairs, model)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created DataLoader with {len(dataloader)} batches")
    return dataloader


def setup_loss_function(model: SentenceTransformer):
    """
    Setup the contrastive loss function.
    
    Args:
        model (SentenceTransformer): The model to train
        
    Returns:
        Loss function for contrastive learning
    """
    print("Setting up MultipleNegativesRankingLoss...")
    
    # MultipleNegativesRankingLoss is ideal for query-document pairs
    loss = losses.MultipleNegativesRankingLoss(model)
    
    print("Loss function configured")
    return loss


def train_model(
    model: SentenceTransformer,
    dataloader: DataLoader,
    loss_function,
    config: dict
):
    """
    Train the sentence transformer model.
    
    Args:
        model (SentenceTransformer): Model to train
        dataloader (DataLoader): Training data
        loss_function: Loss function for training
        config (dict): Configuration dictionary
    """
    epochs = config["finetune"]["epochs"]
    warmup_steps = config["finetune"]["warmup_steps"]
    output_path = config["finetune"]["output_path"]
    
    print("=" * 60)
    print("Starting Fine-tuning Training")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    total_steps = len(dataloader) * epochs
    
    print(f"Training configuration:")
    print(f"Epochs: {epochs}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Total steps: {total_steps}")
    print(f"GPUs: {torch.cuda.device_count()}")
    
    # Train the model
    model.fit(
        train_objectives=[(dataloader, loss_function)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        max_grad_norm=1.0,
        use_amp=True,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': 3e-5},
        weight_decay=0.01,

    )
    
    print("=" * 60)
    print("Fine-tuning completed!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


def main():
    """
    Main training function using config.yaml.
    """
    print("=" * 60)
    print("Fine-tuning Semantic Encoder")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Create training pairs directly from dataset
    pairs = create_training_pairs_from_dataset(config)
    
    # Setup model
    model = setup_model(config)
    
    # Create data loader
    batch_size = config["finetune"]["batch_size"]
    dataloader = create_dataloader(pairs, model, batch_size=batch_size)
    
    # Setup loss function
    loss_function = setup_loss_function(model)
    
    # Train the model
    train_model(
        model=model,
        dataloader=dataloader,
        loss_function=loss_function,
        config=config
    )
    
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main() 