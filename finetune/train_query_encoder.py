#!/usr/bin/env python3
"""
Fine-tune MiniLM semantic encoder using contrastive learning with multi-GPU support.

This script trains a sentence transformer model using query-document pairs
to improve retrieval performance. Optimized for fast training with multiple GPUs.
"""

import os
import pickle
import torch
import torch.nn as nn
import random
from typing import List, Optional
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datetime import datetime


def load_training_pairs(pairs_path: str, max_pairs: Optional[int] = None) -> List[InputExample]:
    """
    Load training pairs from pickle file with optional sampling.
    
    Args:
        pairs_path (str): Path to the training pairs pickle file
        max_pairs (int): Maximum number of pairs to load (for fast training)
        
    Returns:
        List[InputExample]: List of training pairs
    """
    print(f"Loading training pairs from {pairs_path}...")
    
    with open(pairs_path, 'rb') as f:
        pairs = pickle.load(f)
    
    # Sample pairs for faster training if specified
    if max_pairs is not None and len(pairs) > max_pairs:
        print(f"Sampling {max_pairs} pairs from {len(pairs)} for fast training...")
        pairs = random.sample(pairs, max_pairs)
    
    print(f"Using {len(pairs)} training pairs")
    return pairs


def setup_model(model_name: str, use_multi_gpu: bool = True) -> SentenceTransformer:
    """
    Load and setup the sentence transformer model with multi-GPU support.
    
    Args:
        model_name (str): Name of the base model to load
        use_multi_gpu (bool): Whether to use multiple GPUs
        
    Returns:
        SentenceTransformer: Loaded model
    """
    print(f"Loading base model: {model_name}")
    
    # Load model on primary GPU first
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    # Enable multi-GPU if available
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"🚀 Enabling multi-GPU training with {torch.cuda.device_count()} GPUs")
        
        # Wrap the internal model with DataParallel
        if hasattr(model, '_modules'):
            # Find the transformer module and wrap it
            for name, module in model._modules.items():
                if hasattr(module, 'auto_model'):
                    print(f"Wrapping {name} module with DataParallel")
                    model._modules[name] = nn.DataParallel(module)
        
        print(f"✅ Multi-GPU setup complete")
    else:
        print(f"Using single GPU: {device}")
    
    print(f"Model max sequence length: {model.get_max_seq_length()}")
    return model


def create_dataloader(pairs: List[InputExample], model: SentenceTransformer, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for training pairs with multi-GPU batch size scaling.
    
    Args:
        pairs (List[InputExample]): Training pairs
        model (SentenceTransformer): Model for tokenization
        batch_size (int): Base batch size (will be scaled for multi-GPU)
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        DataLoader: DataLoader for training
    """
    # Use the provided batch size directly
    effective_batch_size = batch_size
    
    print(f"Creating DataLoader with batch_size={effective_batch_size}, shuffle={shuffle}")
    
    # Use sentence-transformers' SentencesDataset for proper handling
    from sentence_transformers.datasets import SentencesDataset
    
    dataset = SentencesDataset(pairs, model)
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=4,  # Speed up data loading
        pin_memory=True  # Faster GPU transfer
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
    # It treats other documents in the batch as negative examples
    loss = losses.MultipleNegativesRankingLoss(model)
    
    print("Loss function configured")
    return loss


def train_model(
    model: SentenceTransformer,
    dataloader: DataLoader,
    loss_function,
    epochs: int = 1,
    warmup_steps: int = 100,
    output_path: str = "finetune/model/",
    target_minutes: int = 10
):
    """
    Train the sentence transformer model with time-limited training.
    
    Args:
        model (SentenceTransformer): Model to train
        dataloader (DataLoader): Training data
        loss_function: Loss function for training
        epochs (int): Number of training epochs
        warmup_steps (int): Number of warmup steps
        output_path (str): Path to save the trained model
        target_minutes (int): Target training time in minutes
    """
    print("=" * 60)
    print("Starting Fast Multi-GPU Training")
    print("=" * 60)
    
    total_steps = len(dataloader) * epochs
    steps_per_minute = total_steps / target_minutes
    
    print(f"Training configuration:")
    print(f"  🎯 Target time: {target_minutes} minutes")
    print(f"  📊 Epochs: {epochs}")
    print(f"  🔥 Warmup steps: {warmup_steps}")
    print(f"  📦 Batch size: {dataloader.batch_size}")
    print(f"  🔄 Batches per epoch: {len(dataloader)}")
    print(f"  ⚡ Total steps: {total_steps}")
    print(f"  🚀 Required speed: {steps_per_minute:.1f} steps/minute")
    print(f"  💪 GPUs: {torch.cuda.device_count()}")
    
    # Calculate max steps for 10-minute training
    max_steps = int(steps_per_minute * target_minutes)
    if max_steps < total_steps:
        print(f"  ⏱️ Will stop at {max_steps} steps for time limit")
    
    # Train the model
    model.fit(
        train_objectives=[(dataloader, loss_function)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        max_grad_norm=1.0,  # Gradient clipping for stability
        use_amp=True,  # Mixed precision for faster training
        # Optimizer settings for faster convergence
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': 3e-5},  # Slightly higher learning rate
        weight_decay=0.01,
        # Stop early if time limit reached
        steps_per_epoch=min(len(dataloader), max_steps // epochs) if max_steps < total_steps else None,
    )
    
    print("=" * 60)
    print("Fast Training completed!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


def main():
    """
    Main training function optimized for fast single-GPU training.
    """
    # Configuration for 10-minute training
    pairs_path = "finetune/pairs.pkl"
    base_model = "all-MiniLM-L6-v2"
    output_path = "finetune/model/"
    
    # Fast training hyperparameters
    batch_size = 32  # Conservative batch size to avoid OOM
    epochs = 1       # Single epoch for speed
    warmup_steps = 50  # Minimal warmup
    max_pairs = 50000  # Smaller sample for 10-minute training
    target_minutes = 10  # Target training time
    
    print("=" * 60)
    print("🚀 Fast Single-GPU Fine-tuning (10-minute training)")
    print("=" * 60)
    
    # Load training data (with sampling for speed)
    pairs = load_training_pairs(pairs_path, max_pairs=max_pairs)
    
    # Setup model (single GPU)
    model = setup_model(base_model, use_multi_gpu=False)
    
    # Create data loader
    dataloader = create_dataloader(pairs, model, batch_size=batch_size)
    
    # Setup loss function
    loss_function = setup_loss_function(model)
    
    # Train the model
    train_model(
        model=model,
        dataloader=dataloader,
        loss_function=loss_function,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        target_minutes=target_minutes
    )
    
    print("🎉 Fast fine-tuning completed!")
    print(f"Fine-tuned model available at: {output_path}")
    print("\nExpected improvements:")
    print("✅ Better title-to-abstract matching")
    print("✅ Domain-specific vocabulary understanding")
    print("✅ Improved semantic similarity for academic papers")
    print("\nNext steps:")
    print("1. Test the fine-tuned model")
    print("2. Compare performance with base model")
    print("3. Update retrieval pipeline to use fine-tuned model")


if __name__ == "__main__":
    main() 