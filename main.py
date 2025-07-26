"""
Fine-Tuned Hybrid RAG Pipeline

This is the main entry point for the hybrid RAG system that combines:
- Dense semantic search
- Sparse TF-IDF search
- Metadata boosting

The pipeline includes:
1. Data download and processing
2. Base embedding generation
3. Model fine-tuning
4. Fine-tuned embedding generation
5. Eval of Pre and Post-training Retrieval
"""

import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from specified YAML file"""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_step(step_name: str, command: str, skip: bool = False):
    """Run a pipeline step with proper logging."""
    print("\n" + "=" * 80)
    print(f"STEP: {step_name}")
    print("=" * 80)
    
    if skip:
        print(f"Skipping {step_name} (skip flag provided)")
        return True
    
    print(f"Running: {command}")
    result = os.system(command)
    
    if result == 0:
        print(f"{step_name} completed successfully!")
        return True
    else:
        print(f"{step_name} failed with exit code {result}")
        return False


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Found {description}: {filepath} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"Missing {description}: {filepath}")
        return False


def create_output_directory() -> str:
    """Create timestamped output directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


def save_file_paths(output_dir: str, config: dict, steps_completed: list):
    """Save a text file listing all generated file paths."""
    paths_file = os.path.join(output_dir, "generated_files.txt")
    
    with open(paths_file, 'w') as f:
        f.write("Generated Files from Fine-Tuned Hybrid RAG Pipeline\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset files
        f.write("DATASET FILES:\n")
        f.write("-" * 20 + "\n")
        if "data_processing" in steps_completed:
            f.write(f"✓ Processed documents: {config['dataset']['processed_path']}\n")
        else:
            f.write(f"✗ Processed documents: {config['dataset']['processed_path']} (not generated)\n")
        f.write("\n")
        
        # Embedding files
        f.write("EMBEDDING FILES:\n")
        f.write("-" * 20 + "\n")
        if "base_embeddings" in steps_completed:
            f.write(f"✓ Base dense embeddings: {config['embeddings']['dense_path']}\n")
            f.write(f"✓ TF-IDF vectorizer: {config['embeddings']['tfidf_vectorizer_path']}\n")
        else:
            f.write(f"✗ Base dense embeddings: {config['embeddings']['dense_path']} (not generated)\n")
            f.write(f"✗ TF-IDF vectorizer: {config['embeddings']['tfidf_vectorizer_path']} (not generated)\n")
        
        if "finetuned_embeddings" in steps_completed:
            f.write(f"✓ Fine-tuned embeddings: {config['embeddings']['dense_finetuned_path']}\n")
        else:
            f.write(f"✗ Fine-tuned embeddings: {config['embeddings']['dense_finetuned_path']} (not generated)\n")
        f.write("\n")
        
        # Training files
        f.write("TRAINING FILES:\n")
        f.write("-" * 20 + "\n")
        if "pair_generation" in steps_completed:
            f.write(f"✓ Realistic training pairs: finetune/realistic_training_pairs.json\n")
        else:
            f.write(f"✗ Realistic training pairs: finetune/realistic_training_pairs.json (not generated)\n")
        f.write("\n")
        
        # Model files
        f.write("MODEL FILES:\n")
        f.write("-" * 20 + "\n")
        if "fine_tuning" in steps_completed:
            f.write(f"✓ Fine-tuned model: {config['finetune']['output_path']}\n")
        else:
            f.write(f"✗ Fine-tuned model: {config['finetune']['output_path']} (not generated)\n")
        f.write("\n")
        
        # Evaluation files
        f.write("EVALUATION FILES:\n")
        f.write("-" * 20 + "\n")
        if "evaluation" in steps_completed:
            f.write(f"✓ Evaluation output: {output_dir}/evaluation_output.txt\n")
        else:
            f.write(f"✗ Evaluation output: (not generated)\n")
        f.write("\n")
        
        # Summary files
        f.write("SUMMARY FILES:\n")
        f.write("-" * 20 + "\n")
        f.write(f"✓ Run summary: {output_dir}/run_summary.json\n")
        f.write(f"✓ File paths: {output_dir}/generated_files.txt\n")


def save_run_summary(output_dir: str, config: dict, config_path: str, steps_completed: list, 
                    evaluation_results: dict = None):
    """Save a summary of the pipeline run."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config_file": config_path,
        "config": config,
        "steps_completed": steps_completed,
        "evaluation_results": evaluation_results,
        "system_info": {
            "python_version": sys.version,
            "working_directory": os.getcwd()
        }
    }
    
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Run summary saved to: {summary_path}")


def main():
    """Main pipeline orchestrator."""
    print("Fine-Tuned Hybrid RAG Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    print(f"Loaded configuration from config.yaml")
    print(f"   Dataset: {config['dataset']['name']}")
    print(f"   Model: {config['model']['encoder']}")
    print(f"   Device: {config['model']['device']}")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Track completed steps
    steps_completed = []
    
    # Step 1: Data Download and Processing
    success = run_step(
        "Data Download and Processing",
        "python data/download_data.py",
        skip=False
    )
    if success:
        steps_completed.append("data_processing")
    else:
        print("Pipeline failed at data processing step")
        return 1
    
    # Step 2: Base Embedding Generation
    success = run_step(
        "Base Embedding Generation",
        "python embeddings/generate_embeddings.py",
        skip=False
    )
    if success:
        steps_completed.append("base_embeddings")
    else:
        print("Pipeline failed at base embedding generation")
        return 1
    
    # Step 3: Generate Realistic Training Pairs
    success = run_step(
        "Generate Realistic Training Pairs",
        "python finetune/generate_pairs.py",
        skip=False
    )
    if success:
        steps_completed.append("pair_generation")
    else:
        print("Pipeline failed at training pair generation step")
        return 1
    
    # Step 4: Model Fine-tuning with Realistic Pairs
    success = run_step(
        "Model Fine-tuning with Realistic Pairs",
        "python finetune/train_query_encoder.py",
        skip=False
    )
    if success:
        steps_completed.append("fine_tuning")
    else:
        print("Pipeline failed at fine-tuning step")
        return 1
    
    # Step 5: Fine-tuned Embedding Generation
    success = run_step(
        "Fine-tuned Embedding Generation",
        "python embeddings/generate_finetuned_embeddings.py",
        skip=False
    )
    if success:
        steps_completed.append("finetuned_embeddings")
    else:
        print("Pipeline failed at fine-tuned embedding generation")
        return 1
    
    # Step 6: Verify Required Files
    print("\n" + "=" * 80)
    print("VERIFICATION: Checking Required Files")
    print("=" * 80)
    
    required_files = [
        (config["dataset"]["processed_path"], "Processed dataset"),
        (config["embeddings"]["dense_path"], "Base dense embeddings"),
        (config["embeddings"]["tfidf_vectorizer_path"], "TF-IDF vectorizer"),
        (config["finetune"]["output_path"], "Fine-tuned model"),
        (config["embeddings"]["dense_finetuned_path"], "Fine-tuned embeddings"),
    ]
    
    all_files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\nSome required files are missing. Cannot proceed with evaluation.")
        return 1
    
    # Step 7: Evaluation
    print("\n" + "=" * 80)
    print("STEP: Model Evaluation")
    print("=" * 80)
    
    eval_command = f"python eval/test_models.py --num-queries 100"
    print(f"Running: {eval_command}")
    
    # Capture evaluation output
    eval_result = os.system(f"{eval_command} > {output_dir}/evaluation_output.txt 2>&1")
    
    if eval_result == 0:
        print("Evaluation completed successfully!")
        steps_completed.append("evaluation")
        
        # Copy evaluation output to console
        with open(f"{output_dir}/evaluation_output.txt", 'r') as f:
            eval_output = f.read()
            print("\nEVALUATION RESULTS:")
            print("-" * 40)
            print(eval_output)
        
        evaluation_results = {"status": "completed", "results_file": f"{output_dir}/evaluation_output.txt"}
        
    else:
        print("Evaluation failed!")
        evaluation_results = {"status": "failed", "exit_code": eval_result}
    
    # Step 8: Generate Final Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    print(f"Output Directory: {output_dir}")
    print(f"Config File: config.yaml")
    print(f"Completed Steps: {', '.join(steps_completed)}")
    
    if "evaluation" in steps_completed:
        print(f"Evaluation Results: {output_dir}/pipeline_output.txt")
        print("Pipeline completed successfully!")
    else:
        print("Pipeline completed with some issues")
    
    # Save run summary
    save_run_summary(output_dir, config, "config.yaml", steps_completed, evaluation_results)
    
    # Save file paths summary
    save_file_paths(output_dir, config, steps_completed)
    
    print(f"\nComplete run summary: {output_dir}/run_summary.json")
    print(f"Generated file paths: {output_dir}/generated_files.txt")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
