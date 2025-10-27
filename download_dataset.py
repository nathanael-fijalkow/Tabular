#!/usr/bin/env python3
"""
Download the OCR benchmark dataset from Hugging Face and organize it locally.
Images saved to data/img/ with row ID as filename.
Metadata saved to data/dataset.json
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image


def download_ocr_benchmark(output_dir: str = "data"):
    """
    Download the getomni-ai/ocr-benchmark dataset and save it locally.
    
    Args:
        output_dir: Base directory for saving data (default: "data")
    """
    # Create directories
    base_path = Path(output_dir)
    img_path = base_path / "img"
    img_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("getomni-ai/ocr-benchmark", split="test")
    
    print(f"Dataset loaded with {len(dataset)} rows")
    
    # Collect metadata for JSON
    metadata = []
    
    for idx, row in enumerate(dataset):
        row_id = row.get('id', idx)
        
        # Save image if present
        if 'image' in row and row['image'] is not None:
            img = row['image']
            if isinstance(img, Image.Image):
                # Save as PNG
                img_filename = f"{row_id}.png"
                img_path_full = img_path / img_filename
                img.save(img_path_full)
                print(f"Saved image {idx + 1}/{len(dataset)}: {img_filename}")
            else:
                print(f"Warning: Row {row_id} has non-image data in 'image' field")
        
        # Collect all non-image data for JSON
        row_data = {k: v for k, v in row.items() if k != 'image'}
        row_data['id'] = row_id
        row_data['image_file'] = f"img/{row_id}.png"
        metadata.append(row_data)
    
    # Save metadata to JSON
    json_path = base_path / "dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset downloaded successfully!")
    print(f"- Images: {img_path} ({len(metadata)} files)")
    print(f"- Metadata: {json_path}")


if __name__ == "__main__":
    download_ocr_benchmark()
