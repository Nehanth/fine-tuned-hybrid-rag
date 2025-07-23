# fine-tuned-hybrid-rag

A fine-tuned hybrid retrieval-augmented generation (RAG) system for academic paper search. Combines dense semantic search, sparse TF-IDF search, and metadata boosting with optional fine-tuning capabilities.

## 🏗️ Architecture

**Hybrid Retrieval System** with three components:
- **Dense Semantic Search** (60% weight) - MiniLM sentence transformers
- **Sparse TF-IDF Search** (30% weight) - Traditional keyword matching  
- **Metadata Boosting** (10% weight) - Publication year, venue, field preferences

## 📋 Available Scripts

### 1. Data Preparation
```bash
# Download and process S2ORC academic papers dataset
python data/download_data.py
```
- Downloads `leminda-ai/s2orc_small` from HuggingFace
- Processes papers into JSONL format with metadata
- Output: `data/processed_docs.jsonl` (1.2GB)

### 2. Generate Base Embeddings
```bash
# Create dense and sparse embeddings for all documents
python embeddings/generate_embeddings.py
```
- Generates dense embeddings using `all-MiniLM-L6-v2`
- Creates TF-IDF vectorizer for sparse search
- Outputs:
  - `embeddings/dense.npy` (1.3GB)
  - `embeddings/tfidf_vectorizer.pkl` (375KB)

### 3. Test Basic Retrieval
```bash
# Demo the hybrid retrieval system
python test_retrieval.py
```
- Tests basic hybrid search (no metadata filtering)
- Tests metadata-boosted search with filters
- Shows detailed scoring breakdown

### 4. Fine-tuning (Optional)

#### Generate Training Pairs
```bash
# Create title-to-abstract training pairs
python finetune/generate_pairs.py
```
- Creates contrastive learning pairs from original dataset
- Output: `finetune/pairs.pkl` (1.1GB)

#### Train Fine-tuned Model
```bash
# Fine-tune MiniLM encoder (10-minute training)
python finetune/train_query_encoder.py
```
- Fine-tunes model using contrastive learning
- GPU-accelerated with configurable batch sizes
- Output: `finetune/model/` (fine-tuned model)

#### Generate Fine-tuned Embeddings
```bash
# Re-embed documents with fine-tuned model
python embeddings/generate_finetuned_embeddings.py
```
- Creates embeddings using fine-tuned model
- Output: `embeddings/dense_finetuned.npy` (1.3GB)

### 5. Model Evaluation
```bash
# Compare base vs fine-tuned model performance
python eval/test_models.py
```
- Tests both models on 10 evaluation queries
- Shows ranking improvements and score comparisons
- Includes metadata filtering scenarios

## 🚀 Quick Start

### Minimum Setup (Base Model Only)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and process data
python data/download_data.py

# 3. Generate embeddings
python embeddings/generate_embeddings.py

# 4. Test retrieval
python test_retrieval.py
```

### Full Setup (With Fine-tuning)
```bash
# Steps 1-3 from above, then:

# 4. Generate training pairs
python finetune/generate_pairs.py

# 5. Fine-tune model
python finetune/train_query_encoder.py

# 6. Generate fine-tuned embeddings
python embeddings/generate_finetuned_embeddings.py

# 7. Compare models
python eval/test_models.py
```

## 📊 Usage Examples

### Basic Search
```python
from retrieval.hybrid_retriever import load_retrieval_components, hybrid_retrieve

# Load system components
components = load_retrieval_components()

# Search for papers
results = hybrid_retrieve(
    "machine learning neural networks deep learning", 
    components, 
    top_k=5
)

# View results
for i, result in enumerate(results):
    print(f"{i+1}. {result['text'][:100]}...")
    print(f"   Score: {result['scores']['final_score']:.4f}")
    print(f"   Year: {result['metadata']['year']}")
```

### Search with Metadata Filtering
```python
# Define user preferences
user_filters = {
    "field": "Computer Science",
    "year_after": 2018,
    "venue": "NeurIPS"
}

# Search with filtering
results = hybrid_retrieve(
    "transformer attention mechanisms", 
    components, 
    user_filters=user_filters,
    top_k=10
)
```

### Using Fine-tuned Model
```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load fine-tuned components
components = load_retrieval_components()
components['sentence_model'] = SentenceTransformer('finetune/model/')
components['dense_embeddings'] = np.load('embeddings/dense_finetuned.npy')

# Search with fine-tuned model
results = hybrid_retrieve("your query", components)
```

## 📁 File Structure

```
├── data/
│   ├── processed_docs.jsonl      # Processed academic papers (1.2GB)
│   └── download_data.py          # Dataset preparation script
├── embeddings/
│   ├── dense.npy                 # Base model embeddings (1.3GB)
│   ├── dense_finetuned.npy       # Fine-tuned embeddings (1.3GB)
│   ├── tfidf_vectorizer.pkl      # TF-IDF vectorizer (375KB)
│   ├── generate_embeddings.py    # Base embedding generation
│   └── generate_finetuned_embeddings.py
├── retrieval/
│   ├── hybrid_retriever.py       # Main retrieval logic
│   ├── scorer.py                 # Score combination
│   └── metadata_boosting.py      # Metadata filtering/boosting
├── finetune/
│   ├── pairs.pkl                 # Training pairs (1.1GB)
│   ├── model/                    # Fine-tuned model checkpoints
│   ├── generate_pairs.py         # Training pair generation
│   └── train_query_encoder.py    # Model fine-tuning
├── eval/
│   ├── eval_dataset.json         # Evaluation queries (10 queries)
│   └── test_models.py            # Model comparison
├── test_retrieval.py             # Main demo script
└── requirements.txt              # Dependencies
```

## 🛠️ Dependencies

```
datasets                # HuggingFace datasets
sentence-transformers   # Dense embeddings & fine-tuning
scikit-learn           # TF-IDF vectorization
numpy                  # Numerical operations
tqdm                   # Progress bars
joblib                 # Model serialization
```

## 📈 System Stats

- **Documents**: ~1.2GB of academic papers from S2ORC
- **Embeddings**: 384-dimensional (MiniLM-L6-v2)
- **Training**: 50K pairs, 10-minute GPU training
- **Evaluation**: 10 real academic queries with ground truth
- **Storage**: ~4GB total (data + embeddings + models)

## 🎯 Performance

The system provides:
- **Semantic Understanding**: Dense embeddings capture meaning
- **Keyword Matching**: TF-IDF for exact term matches
- **Metadata Filtering**: Year, venue, field, author preferences
- **Fine-tuning**: Domain-specific improvements for academic papers
- **Configurable Scoring**: Adjustable weights for different components
