#!/usr/bin/env python3
"""
Download APPS dataset from Hugging Face using mirror endpoint.
Handles newer datasets library that doesn't support dataset scripts.

Usage:
    # Standard method (works if datasets version < 2.14)
    python download_apps_v2.py --output-dir ./data/apps --split train --mirror https://hf-mirror.com
    
    # Force Parquet method (works with latest datasets)
    python download_apps_v2.py --output-dir ./data/apps --split train --method parquet
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

print("Loading dependencies...")
try:
    from datasets import load_dataset
    print("? datasets library imported")
except ImportError:
    print("? datasets not found. Install: pip install datasets")
    exit(1)


def setup_hf_mirror(mirror_url: Optional[str] = None) -> None:
    """Setup Hugging Face mirror endpoint."""
    if mirror_url is None:
        mirror_url = os.environ.get("HF_ENDPOINT")
    
    if mirror_url:
        os.environ["HF_ENDPOINT"] = mirror_url
        print(f"? Using HF mirror: {mirror_url}")
    else:
        print("? No HF mirror specified. Using official Hugging Face.")


def download_apps_standard(
    split: str = "train",
    cache_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> bool:
    """
    Try standard dataset loading (works with older datasets versions).
    
    Returns: True if successful, False if dataset scripts are not supported
    """
    print(f"? Attempting standard APPS load (split: {split})...")
    
    try:
        dataset = load_dataset(
            "codeparrot/apps",
            split=split,
            cache_dir=cache_dir,
        )
        print(f"? Successfully loaded {len(dataset)} samples via standard method")
        
        if output_dir:
            _save_to_jsonl(dataset, split, output_dir)
        
        return True
    
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print(f"? Standard method failed: {e}")
            print("  This is expected with newer datasets versions. Trying Parquet method...")
            return False
        raise
    except Exception as e:
        print(f"? Standard method error: {e}")
        raise


def download_apps_parquet(
    split: str = "train",
    output_dir: Optional[str] = None,
) -> bool:
    """
    Download APPS via Parquet files (works with all datasets versions).
    """
    try:
        import pandas as pd
    except ImportError:
        print("? pandas required. Install: pip install pandas")
        return False
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("? huggingface_hub required. Install: pip install huggingface_hub")
        return False
    
    print(f"? Downloading APPS {split} split via Parquet...")
    
    split_file_map = {
        "train": "train-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }
    
    if split not in split_file_map:
        print(f"? Unknown split: {split}. Available: {list(split_file_map.keys())}")
        return False
    
    filename = split_file_map[split]
    
    try:
        print(f"  Downloading {filename}...")
        parquet_path = hf_hub_download(
            repo_id="codeparrot/apps",
            filename=filename,
            repo_type="dataset",
        )
        print(f"? Downloaded: {parquet_path}")
        
        # Load parquet
        print("  Loading Parquet file...")
        df = pd.read_parquet(parquet_path)
        print(f"? Loaded {len(df)} samples")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            jsonl_file = output_path / f"apps_{split}.jsonl"
            print(f"? Converting to JSONL: {jsonl_file}...")
            
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for idx, row in df.iterrows():
                    try:
                        sample = dict(row)
                        
                        # Parse JSON fields safely
                        if "solutions" in sample:
                            sample["solutions"] = _safe_json_loads(sample.get("solutions"), default=[])
                        if "input_output" in sample:
                            sample["input_output"] = _safe_json_loads(sample.get("input_output"), default={})
                        
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"  ? Error processing sample {idx}: {e}. Skipping...")
                        continue
                    
                    if (idx + 1) % 1000 == 0:
                        print(f"  ? Processed {idx + 1} samples")
            
            print(f"? Saved {len(df)} samples to {jsonl_file}")
            
            # Save metadata
            info_file = output_path / f"apps_{split}_info.json"
            info = {
                "split": split,
                "total_samples": len(df),
                "features": list(df.columns),
            }
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            print(f"? Saved metadata to {info_file}")
        
        return True
    
    except Exception as e:
        print(f"? Parquet download failed: {e}")
        return False


def _safe_json_loads(s: any, default=None):
    """Safely parse JSON string with fallback."""
    if s is None or s == "":
        return default if default is not None else {}
    if not isinstance(s, str):
        return s  # Already parsed
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ? JSON decode error: {e}. Using default value.")
        return default if default is not None else {}


def _save_to_jsonl(dataset, split: str, output_dir: str) -> None:
    """Save dataset to JSONL format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    jsonl_file = output_path / f"apps_{split}.jsonl"
    print(f"? Saving to {jsonl_file}...")
    
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            try:
                # Parse JSON strings safely
                if "solutions" in sample:
                    sample["solutions"] = _safe_json_loads(sample.get("solutions"), default=[])
                if "input_output" in sample:
                    sample["input_output"] = _safe_json_loads(sample.get("input_output"), default={})
                
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  ? Error processing sample {idx}: {e}. Skipping...")
                continue
            
            if (idx + 1) % 1000 == 0:
                print(f"  ? Processed {idx + 1} samples")
    
    print(f"? Saved {len(dataset)} samples to {jsonl_file}")
    
    # Save metadata
    info_file = output_path / f"apps_{split}_info.json"
    info = {
        "split": split,
        "total_samples": len(dataset),
        "features": list(dataset.features.keys()),
    }
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"? Saved metadata to {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download APPS dataset with automatic fallback support"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HF cache directory",
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default=None,
        help="HF mirror URL (e.g., https://hf-mirror.com)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "standard", "parquet"],
        default="auto",
        help="Download method: auto (try both), standard (old API), parquet (new API)",
    )
    
    args = parser.parse_args()
    
    # Setup mirror
    setup_hf_mirror(args.mirror)
    
    success = False
    
    if args.method in ["auto", "standard"]:
        print("\n" + "=" * 60)
        print(f"Method 1: Standard load (datasets API)")
        print("=" * 60)
        success = download_apps_standard(
            split=args.split,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
        )
    
    if not success and args.method in ["auto", "parquet"]:
        print("\n" + "=" * 60)
        print(f"Method 2: Parquet download (huggingface_hub)")
        print("=" * 60)
        success = download_apps_parquet(
            split=args.split,
            output_dir=args.output_dir,
        )
    
    if success:
        print("\n" + "=" * 60)
        print("? Download complete!")
        print("=" * 60)
        if args.output_dir:
            print(f"\n? Output files in: {args.output_dir}")
            print(f"   - apps_{args.split}.jsonl")
            print(f"   - apps_{args.split}_info.json")
    else:
        print("\n" + "=" * 60)
        print("? All download methods failed")
        print("=" * 60)
        print("\n? Try these workarounds:")
        print("   1. Downgrade datasets: pip install 'datasets<2.14'")
        print("   2. Then run: python download_apps_v2.py --method standard ...")
        print("   3. Or manually download from: https://huggingface.co/datasets/codeparrot/apps")
        exit(1)


if __name__ == "__main__":
    main()
