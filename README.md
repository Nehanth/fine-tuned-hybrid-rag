# Fine-Tuned Hybrid RAG System

A retrieval-augmented generation (RAG) system that combines dense semantic search, sparse keyword matching, and metadata boosting with fine-tuning capabilities

## Overview

This system implements a hybrid retrieval approach that:
- Uses dense embeddings (Sentence Transformers) for semantic similarity
- Applies sparse TF-IDF vectors for keyword matching
- Incorporates metadata boosting based on dataset details
- Fine-tunes the base model using contrastive learning
- Evaluates performance using [BEIR methodology](https://github.com/opendatahub-io/rag/blob/main/benchmarks/llama-stack-rag-with-beir/benchmark_beir_ls_vs_no_ls.py)

## Architecture

The pipeline consists of five main stages:

1. **Data Processing**: Downloads and processes academic papers from S2ORC dataset
2. **Base Embedding Generation**: Creates dense embeddings and TF-IDF vectors
3. **Model Fine-tuning**: Trains the encoder using query-document pairs
4. **Fine-tuned Embedding Generation**: Creates embeddings with the improved model
5. **Evaluation**: Compares base vs fine-tuned performance using standard metrics

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python main.py
```

This runs the complete pipeline using settings from `config.yaml`.

### Configuration

You can customize the encoder, scoring weights, and fine-tuning parameters by editing `config.yaml`:

```yaml
model:
  encoder: "sentence-transformers/all-MiniLM-L6-v2"  # Any Sentence Transformers model
  device: "auto"

scoring:
  dense_weight: 0.6    # Adjust semantic search weight
  sparse_weight: 0.3   # Adjust keyword search weight  
  boost_weight: 0.1    # Adjust metadata boost weight

finetune:
  max_pairs: 50000     # Number of training pairs
  epochs: 1            # Training epochs
  batch_size: 32       # Batch size for training
```

**Note**: The dataset configuration is currently fixed to `leminda-ai/s2orc_small` since the metadata boosting schema is not yet dynamic and only supports this specific dataset format.

## Components

### Dense Retrieval

Uses Sentence Transformers to create semantic embeddings that capture the meaning and context of documents. This component excels at finding conceptually similar content even when exact keywords don't match.

### Sparse Retrieval

Applies TF-IDF vectorization for traditional keyword-based matching. This component ensures that documents containing specific terms mentioned in the query are properly ranked and retrieved.

### Metadata Boosting

Enhances retrieval scores based on document metadata such as publication year, venue, authors, and field of study. Documents matching user preferences receive higher rankings in the final results.

### Fine-tuning

The system uses contrastive learning with automatically generated query-document pairs from the dataset. Training queries include metadata context (venue, year, authors, field) to improve retrieval performance.

## File Structure

```
├── data/
│   ├── processed_docs.jsonl          # Processed dataset
│   └── download_data.py               # Dataset preparation
├── embeddings/
│   ├── dense.npy                      # Base model embeddings
│   ├── dense_finetuned.npy            # Fine-tuned embeddings
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── generate_embeddings.py         # Base embedding generation
│   └── generate_finetuned_embeddings.py # Tuned embedding generation
├── retrieval/
│   ├── hybrid_retriever.py            # Main retrieval logic
│   └── metadata_boosting.py           # Metadata filtering and boosting
├── finetune/
│   ├── model/                         # Fine-tuned model checkpoints
│   └── train_query_encoder.py         # Model fine-tuning
├── eval/
│   └── test_models.py                 # Evaluation
├── how_to_use/
│   └── how_to_use.ipynb               # Shows how to use the retrival system
├── results/
│   └── run_YYYYMMDD_HHMMSS/          # Pipeline results
├── config.yaml                       # Configuration file
└── main.py                           # Pipeline orchestrator
```

## Evaluation

The system uses BEIR methodology with standard metrics:

- **MAP@k**: Mean Average Precision
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Statistical significance testing** using permutation tests

## How to use

Check out the `how_to_use/` folder for examples. The notebook shows how to use the retrieval system with code examples.

## Note

This implementation performs very well in evaluation, as you can see in `results/run_20250726_000435`, since the synthetically generated training pairs match the same pattern used in evaluation. However, when it comes to real-world queries, it performs worse because it doesn't know how to handle those types of patterns. Check out the `master` branch for an improved approach that creates more realistic synthetic data and uses the same logic for both training and evaluation pairs.