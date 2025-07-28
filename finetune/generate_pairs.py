"""
Generate realistic training pairs for fine-tuning the semantic encoder.

Got from AI, and modified to fit our needs.
"""

import json
import random
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_processed_documents(file_path: str, max_docs: Optional[int] = None) -> List[Dict]:
    """Load processed documents from JSONL file."""
    print(f"Loading documents from {file_path}...")
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading documents")):
            if max_docs and i >= max_docs:
                break
            try:
                doc = json.loads(line.strip())
                documents.append(doc)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(documents)} documents")
    return documents


def extract_key_terms(text: str, max_terms: int = 5) -> List[str]:
    """Extract key technical terms from text."""
    # Remove common academic words and focus on technical terms
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'from', 'this', 'that', 'these', 'those', 'we', 'they', 'our', 'their',
        'paper', 'study', 'research', 'analysis', 'method', 'approach', 'results', 'conclusion',
        'using', 'based', 'new', 'novel', 'proposed', 'present', 'show', 'demonstrate'
    }
    
    # Extract meaningful terms (2+ characters, not all numbers)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]*\b', text.lower()) #extracts technical terms from academic text
    key_terms = []
    
    for word in words:
        if (len(word) >= 3 and 
            word not in stop_words and 
            not word.isdigit() and
            len([c for c in word if c.isalpha()]) >= 2):  # At least 2 letters
            key_terms.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in key_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms[:max_terms]


def generate_keyword_queries(doc: Dict) -> List[str]:
    """Generate keyword-based queries from document metadata."""
    queries = []
    metadata = doc['metadata']
    title = metadata.get('title', '')
    abstract = metadata.get('abstract', '')
    
    # Extract key terms from title and abstract
    title_terms = extract_key_terms(title, 3)
    abstract_terms = extract_key_terms(abstract, 4)
    
    # Combine terms in different ways
    if title_terms:
        # Use 2-3 key terms from title
        if len(title_terms) >= 2:
            queries.append(' '.join(title_terms[:2]))
        if len(title_terms) >= 3:
            queries.append(' '.join(title_terms[:3]))
    
    # Mix title and abstract terms
    if title_terms and abstract_terms:
        mixed_terms = title_terms[:2] + abstract_terms[:2]
        queries.append(' '.join(mixed_terms[:3]))
    
    # Domain-specific combinations
    fields = metadata.get('fieldsOfStudy', [])
    if fields and title_terms:
        domain = fields[0].lower()
        queries.append(f"{domain} {' '.join(title_terms[:2])}")
    
    return queries


def generate_question_queries(doc: Dict) -> List[str]:
    """Generate question-based queries from document content."""
    queries = []
    metadata = doc['metadata']
    title = metadata.get('title', '')
    abstract = metadata.get('abstract', '')
    
    # Extract key concepts for question generation
    title_terms = extract_key_terms(title, 2)
    
    if title_terms and len(title_terms) > 0:
        main_concept = title_terms[0]
        
        # Only generate questions if main_concept is meaningful (not too short/generic)
        if len(main_concept) >= 3 and main_concept not in ['the', 'and', 'for', 'with']:
            # Generate different question types
            question_templates = [
                f"what is {main_concept}",
                f"how does {main_concept} work",
                f"research on {main_concept}",
                f"{main_concept} methods",
                f"{main_concept} techniques"
            ]
            
            # Add domain context if available
            fields = metadata.get('fieldsOfStudy', [])
            if fields and len(fields) > 0:
                domain = fields[0].lower()
                if len(domain) > 0:
                    question_templates.extend([
                        f"{main_concept} in {domain}",
                        f"{domain} {main_concept}"
                    ])
            
            queries.extend(question_templates[:3])  # Limit to 3 questions
    
    # Generate queries based on abstract patterns
    if abstract and title_terms and len(title_terms) > 0:
        abstract_lower = abstract.lower()
        main_term = title_terms[0]
        
        # Only generate if main_term is meaningful
        if len(main_term) >= 3 and main_term not in ['the', 'and', 'for', 'with']:
            # Look for common research patterns
            if 'improve' in abstract_lower or 'enhancement' in abstract_lower:
                queries.append(f"improve {main_term}")
            
            if 'comparison' in abstract_lower or 'compare' in abstract_lower:
                if len(title_terms) >= 2:
                    second_term = title_terms[1]
                    if len(second_term) >= 3:
                        queries.append(f"compare {main_term} {second_term}")
                else:
                    queries.append(f"compare {main_term}")
            
            if 'analysis' in abstract_lower or 'analyze' in abstract_lower:
                queries.append(f"analysis of {main_term}")
    
    return queries


def generate_title_based_queries(doc: Dict) -> List[str]:
    """Generate queries by paraphrasing and simplifying titles."""
    queries = []
    title = doc['metadata'].get('title', '')
    
    if not title:
        return queries
    
    # Remove special characters and normalize
    clean_title = re.sub(r'[^\w\s-]', '', title)
    
    # Convert to natural search queries
    title_lower = clean_title.lower()
    
    # Remove common academic prefixes/suffixes
    patterns_to_remove = [
        r'^a study of ',
        r'^an analysis of ',
        r'^research on ',
        r'^the effect of ',
        r'^effects of ',
        r': a .*$',
        r': an .*$',
        r' - .*$'
    ]
    
    simplified_title = title_lower
    for pattern in patterns_to_remove:
        simplified_title = re.sub(pattern, '', simplified_title).strip()
    
    # Create variations
    if simplified_title:
        queries.append(simplified_title)
        
        # Add question variations
        if len(simplified_title.split()) <= 4:
            queries.append(f"what is {simplified_title}")
            queries.append(f"research on {simplified_title}")
    
    return queries


def generate_author_venue_queries(doc: Dict) -> List[str]:
    """Generate queries that include author or venue information."""
    queries = []
    metadata = doc['metadata']
    
    # Get key terms from title
    title_terms = extract_key_terms(metadata.get('title', ''), 2)
    
    if not title_terms:
        return queries
    
    # Author-based queries
    authors = metadata.get('authors', [])
    if authors and isinstance(authors[0], dict) and 'name' in authors[0]:
        author_name = authors[0]['name']
        # Extract last name
        if ' ' in author_name:
            last_name = author_name.split()[-1]
            queries.append(f"{last_name} {title_terms[0]}")
    
    # Venue-based queries
    venue = metadata.get('venue', '')
    year = metadata.get('year')
    
    if venue and len(venue.split()) <= 3:  # Short venue names
        queries.append(f"{title_terms[0]} {venue}")
    
    if year and isinstance(year, (int, float)) and year > 1990:
        queries.append(f"{title_terms[0]} {int(year)}")
    
    return queries


def generate_synthetic_pairs(documents: List[Dict], max_pairs: int) -> List[Tuple[str, str]]:
    """Generate diverse synthetic query-document pairs."""
    print(f"Generating {max_pairs} synthetic training pairs...")
    
    pairs = []
    
    # Ensure we don't use more documents than available
    available_docs = len(documents)
    if max_pairs > available_docs:
        print(f"Warning: Requested {max_pairs} pairs but only {available_docs} documents available")
        max_pairs = available_docs
    
    # Sample documents randomly
    sampled_indices = random.sample(range(len(documents)), max_pairs)
    
    for idx in tqdm(sampled_indices, desc="Generating pairs"):
        doc = documents[idx]
        
        # Skip if document doesn't have required fields
        if not doc.get('metadata', {}).get('title') or not doc.get('metadata', {}).get('abstract'):
            continue
        
        # Generate different types of queries
        all_queries = []
        
        # 1. Keyword-based queries (most common in real search)
        all_queries.extend(generate_keyword_queries(doc))
        
        # 2. Question-based queries
        all_queries.extend(generate_question_queries(doc))
        
        # 3. Title-based queries
        all_queries.extend(generate_title_based_queries(doc))
        
        # 4. Author/venue queries (less common but realistic)
        all_queries.extend(generate_author_venue_queries(doc))
        
        # Remove duplicates and empty queries
        unique_queries = []
        seen_queries = set()
        for query in all_queries:
            query = query.strip()
            if query and len(query) >= 3 and query not in seen_queries:
                seen_queries.add(query)
                unique_queries.append(query)
        
        # Select query with balanced distribution between keywords and questions
        if unique_queries:
            # Get different types of queries
            keyword_queries = [q for q in unique_queries if not q.startswith(('what is', 'how does', 'research on'))]
            question_queries = [q for q in unique_queries if q.startswith(('what is', 'how does', 'research on'))]
            
            # Balanced selection: 70% keywords, 30% questions (more realistic distribution)
            if question_queries and random.random() < 0.3:
                selected_query = random.choice(question_queries)
            elif keyword_queries:
                selected_query = keyword_queries[0]
            else:
                selected_query = unique_queries[0]
            
            # Create the pair: query -> document text
            document_text = doc['text']
            pairs.append((selected_query, document_text))
    
    print(f"Generated {len(pairs)} training pairs")
    
    # Show some examples
    print("\nExample training pairs:")
    for i, (query, doc_text) in enumerate(pairs[:5]):
        print(f"{i+1}. Query: '{query}'")
        print(f"   Document: '{doc_text[:100]}...'")
        print()
    
    return pairs


def save_training_pairs(pairs: List[Tuple[str, str]], output_path: str):
    """Save training pairs to a JSON file."""
    print(f"Saving {len(pairs)} training pairs to {output_path}...")
    
    # Convert to the format expected by the training script
    training_data = []
    for query, document in pairs:
        training_data.append({
            'query': query,
            'document': document
        })
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training pairs saved to {output_path}")


def main():
    """Generate realistic training pairs from processed documents."""
    print("=" * 60)
    print("Generating Realistic Training Pairs")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Load processed documents
    data_path = config['dataset']['processed_path']
    max_pairs = config['finetune']['max_pairs']
    
    documents = load_processed_documents(data_path, max_pairs * 2)  # Load extra for sampling
    
    # Generate synthetic pairs
    pairs = generate_synthetic_pairs(documents, max_pairs)
    
    # Save training pairs
    output_path = "finetune/realistic_training_pairs.json"
    save_training_pairs(pairs, output_path)
    
    print("=" * 60)
    print("Training pair generation completed!")
    print(f"Generated: {len(pairs)} pairs")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
