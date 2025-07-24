#!/usr/bin/env python3
"""
Simple test script for hybrid retriever.
Ask questions and get results back with optional metadata filters!
"""

from retrieval.hybrid_retriever import load_retrieval_components, hybrid_retrieve

def parse_filters(query):
    """Parse metadata filters from query string."""
    filters = {}
    original_query = query
    
    # Check for year filters
    if "after:" in query:
        parts = query.split("after:")
        if len(parts) > 1:
            try:
                year = int(parts[1].split()[0])
                filters["year_after"] = year
                query = parts[0].strip()
            except:
                pass
    
    if "before:" in query:
        parts = query.split("before:")
        if len(parts) > 1:
            try:
                year = int(parts[1].split()[0])
                filters["year_before"] = year
                query = parts[0].strip()
            except:
                pass
    
    if "year:" in query:
        parts = query.split("year:")
        if len(parts) > 1:
            try:
                year = int(parts[1].split()[0])
                filters["min_year"] = year
                filters["max_year"] = year
                query = parts[0].strip()
            except:
                pass
    
    # Check for venue filter
    if "venue:" in query:
        parts = query.split("venue:")
        if len(parts) > 1:
            venue = parts[1].split()[0].replace("_", " ")
            filters["venue"] = venue
            query = parts[0].strip()
    
    # Check for field filter
    if "field:" in query:
        parts = query.split("field:")
        if len(parts) > 1:
            field = parts[1].split()[0].replace("_", " ")
            filters["field"] = field
            query = parts[0].strip()
    
    # Check for author filter
    if "author:" in query:
        parts = query.split("author:")
        if len(parts) > 1:
            author = parts[1].split()[0].replace("_", " ")
            filters["author"] = author
            query = parts[0].strip()
    
    return query.strip(), filters

def print_help():
    print("\n📋 Metadata Filter Examples:")
    print("   machine learning after:2020          # Papers after 2020")
    print("   neural networks before:2015          # Papers before 2015") 
    print("   deep learning year:2019              # Papers from exactly 2019")
    print("   NLP venue:ACL                        # Papers from ACL venue")
    print("   transformers field:Computer_Science  # Papers in CS field")
    print("   BERT author:Devlin                   # Papers by author Devlin")
    print("   AI after:2018 venue:NeurIPS          # Multiple filters!")
    print("\n💡 Type 'help' to see this again, 'quit' to exit")

def main():
    print("🔍 Loading hybrid retrieval system...")
    try:
        components = load_retrieval_components()
        print("✅ System loaded successfully!")
        print(f"📚 Ready to search {len(components['documents'])} documents")
        print_help()
        
        while True:
            # Get user query
            query = input("\n💬 Ask a question (or 'help'/'quit'): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print_help()
                continue
                
            if not query:
                continue
            
            # Parse filters from query
            clean_query, user_filters = parse_filters(query)
            
            print(f"\n🔍 Searching for: '{clean_query}'")
            if user_filters:
                print(f"🎯 Filters: {user_filters}")
            print("-" * 30)
            
            # Get results
            results = hybrid_retrieve(clean_query, components, 
                                    user_filters=user_filters, top_k=3)
            
            # Display results
            for i, doc in enumerate(results, 1):
                print(f"\n📄 Result {i}:")
                print(f"   Title: {doc.get('title', 'No title')[:80]}...")
                print(f"   Score: {doc['scores']['final_score']:.4f}")
                print(f"     Dense: {doc['scores']['dense_score']:.3f}")
                print(f"     Sparse: {doc['scores']['sparse_score']:.3f}")
                print(f"     Boost: {doc['scores']['boost_score']:.3f}")
                print(f"   Year: {doc['metadata'].get('year', 'Unknown')}")
                print(f"   Venue: {doc['metadata'].get('venue', 'Unknown')}")
                
                # Show fields if available
                fields = doc['metadata'].get('fieldsOfStudy', [])
                if fields:
                    print(f"   Fields: {', '.join(fields[:3])}...")
                
                # Show first few lines of text
                text = doc.get('text', '')
                if text:
                    preview = ' '.join(text.split()[:30])
                    print(f"   Preview: {preview}...")
                    
    except FileNotFoundError as e:
        print(f"❌ Error: Missing file - {e}")
        print("💡 Make sure you've run:")
        print("   1. python data/download_data.py")
        print("   2. python embeddings/generate_embeddings.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
