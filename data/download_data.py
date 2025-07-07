from datasets import load_dataset
import json

def download_and_process_dataset():
    """
    Download and process the s2orc_small dataset.
    Merges title and paperAbstract into text field.
    Extracts year, venue, fieldsOfStudy, and authors as metadata.
    Skips entries with missing or null paperAbstract.
    """
    print("Loading dataset...")
    dataset = load_dataset("leminda-ai/s2orc_small", split="train")
    
    processed_count = 0
    skipped_count = 0
    
    print("Processing documents...")
    with open("data/processed_docs.jsonl", "w") as f:
        for entry in dataset:
            # Skip entries with missing or null paperAbstract
            if not entry.get("paperAbstract"):
                skipped_count += 1
                continue
            
            # Merge title and paperAbstract into text field
            title = entry.get("title", "")
            abstract = entry.get("paperAbstract", "")
            text = f"{title} {abstract}".strip()
            
            # Create processed document
            doc = {
                "id": entry.get("paperId"),
                "text": text,
                "metadata": {
                    "year": entry.get("year"),
                    "venue": entry.get("venue"),
                    "fieldsOfStudy": entry.get("fieldsOfStudy", []),
                    "authors": entry.get("authors", [])
                }
            }
            
            json.dump(doc, f)
            f.write("\n")
            processed_count += 1
    
    print(f"Processing complete!")
    print(f"Processed: {processed_count} documents")
    print(f"Skipped: {skipped_count} documents (missing abstract)")
    print(f"Output: data/processed_docs.jsonl")

if __name__ == "__main__":
    download_and_process_dataset()
