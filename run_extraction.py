#!/usr/bin/env python3
"""
Run all three extraction tasks on a specific dataset instance.

Usage:
    python run_extraction.py <instance_id> [--provider PROVIDER] [--model MODEL]
    
Example:
    python run_extraction.py 0
    python run_extraction.py 0 --provider gemini --model gemini-2.5-flash
    python run_extraction.py 0 --provider huggingface --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path
from llm import LLM
from extract import OCR_to_markdown, extract_JSON_from_image, extract_image_from_document


def load_dataset():
    """Load the dataset.json file."""
    dataset_path = Path("data/dataset.json")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run download_dataset.py first."
        )
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_extraction_on_instance(instance_id: int, provider: str = "gemini", model_name: str = "gemini-2.5-flash"):
    """
    Run all three extraction tasks on a specific dataset instance.
    
    Args:
        instance_id: The ID of the dataset instance to process
        provider: LLM provider to use ('gemini' or 'huggingface')
        model_name: Model name to use for generation
    """
    # Load dataset
    print(f"Loading dataset...")
    dataset = load_dataset()
    
    # Find the instance
    instance = None
    for item in dataset:
        if item['id'] == instance_id:
            instance = item
            break
    
    if instance is None:
        print(f"Error: Instance with id={instance_id} not found in dataset")
        print(f"Available IDs: {[item['id'] for item in dataset[:10]]}...")
        return
    
    # Get image path
    image_path = Path("data") / "img" / f"{instance_id}.png"
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing instance {instance_id}")
    print(f"Image: {image_path}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Initialize LLM
    print("Initializing LLM...")
    llm = LLM()
    
    # Initialize result variables
    markdown_result = None
    json_result = None
    image_extraction_result = None
    
    # Task 1: OCR to Markdown
    print("\n" + "="*60)
    print("TASK 1: OCR to Markdown")
    print("="*60)
    try:
        markdown_result = OCR_to_markdown(llm, str(image_path), provider=provider, model_name=model_name)
        print(markdown_result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Task 2: Extract JSON from image (using a sample schema)
    print("\n" + "="*60)
    print("TASK 2: Extract JSON from Image")
    print("="*60)
    json_schema = """{
        "text": "string",
        "type": "string",
        "confidence": "number"
    }"""
    try:
        json_result = extract_JSON_from_image(llm, str(image_path), json_schema, provider=provider, model_name=model_name)
        print(json_result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Task 3: Extract image description from document
    print("\n" + "="*60)
    print("TASK 3: Extract Image from Document")
    print("="*60)
    try:
        image_extraction_result = extract_image_from_document(llm, str(image_path), provider=provider, model_name=model_name)
        print(image_extraction_result)
    except Exception as e:
        print(f"Error: {e}")
    
    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"instance_{instance_id}_results.json"
    
    results = {
        "instance_id": instance_id,
        "image_path": str(image_path),
        "provider": provider,
        "model_name": model_name,
        "tasks": {
            "ocr_to_markdown": markdown_result,
            "extract_json": json_result,
            "extract_image": image_extraction_result
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run extraction tasks on a dataset instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_extraction.py 0
  python run_extraction.py 0 --provider gemini --model gemini-2.5-flash
  python run_extraction.py 0 --provider huggingface --model meta-llama/Llama-3.1-8B-Instruct
        """
    )
    
    parser.add_argument(
        "instance_id",
        type=int,
        help="ID of the dataset instance to process"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        choices=["gemini", "huggingface"],
        help="LLM provider to use (default: gemini)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name to use (default: gemini-2.5-flash)"
    )
    
    args = parser.parse_args()
    
    run_extraction_on_instance(args.instance_id, args.provider, args.model)
