"""
Fine-tune MiniLM semantic encoder using realistic training pairs.
"""

import os
import json
import yaml
import torch
import torch.nn as nn
from typing import List, Optional
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.datasets import SentencesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_realistic_training_pairs(pairs_file: str) -> List[InputExample]:
    """
    Load realistic training pairs from JSON file.
    
    Args:
        pairs_file (str): Path to the realistic training pairs JSON file
        
    Returns:
        List[InputExample]: List of training pairs
    """
    print(f"Loading realistic training pairs from: {pairs_file}")
    
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Training pairs file not found: {pairs_file}")
    
    with open(pairs_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    pairs = []
    for item in tqdm(training_data, desc="Converting to InputExamples"):
        query = item['query']
        document = item['document']
        
        # Create training pair: query → document
        pair = InputExample(texts=[query, document])
        pairs.append(pair)
    
    print(f"Loaded {len(pairs)} realistic training pairs")
    
    # Show some examples
    print("\nExample training pairs:")
    for i, pair in enumerate(pairs[:3]):
        print(f"{i+1}. Query: '{pair.texts[0]}'")
        print(f"   Document: '{pair.texts[1][:100]}...'")
        print()
    
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
    
    # Enable multi-GPU if available and requested
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
    print("Setting up MultipleNegativesRankingLoss...") #cross entropy loss function?
    
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
    print("Starting Fine-tuning Training with Realistic Pairs")
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
    print(f"Training pairs: Realistic synthetic queries")
    
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
        optimizer_params={'lr': 2e-5},  # Slightly lower LR for better convergence
        weight_decay=0.01,
        scheduler='WarmupLinear'
    )
    
    print("=" * 60)
    print("Fine-tuning completed!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


def main():
    """
    Main training function using realistic synthetic pairs.
    """
    print("=" * 60)
    print("Fine-tuning Semantic Encoder with Realistic Pairs")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load realistic training pairs (generated in previous pipeline step)
    pairs_file = "finetune/realistic_training_pairs.json"
    if not os.path.exists(pairs_file):
        print(f"ERROR: Training pairs not found at {pairs_file}")
        print("Training pairs should have been generated in the previous pipeline step.")
        print("Please ensure 'python finetune/generate_pairs.py' ran successfully.")
        return
    
    print(f"Using training pairs from: {pairs_file}")
    
    # Load realistic training pairs
    pairs = load_realistic_training_pairs(pairs_file)
    
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
    
    print("Fine-tuning with realistic pairs completed!")


if __name__ == "__main__":
    main()