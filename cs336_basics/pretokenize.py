"""
Standalone script for tokenizing datasets.
This allows pre-tokenizing data before training to avoid re-tokenization.

Usage:
    python cs336_basics/tokenize.py --dataset roneneldan/TinyStories --context_length 256
    python cs336_basics/tokenize.py --dataset roneneldan/TinyStories --context_length 256 --splits train validation
"""

import os
import argparse
import logging
import hashlib
from typing import List

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize dataset and save to cache")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Hugging Face dataset name")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Tokenizer name (from Hugging Face)")
    parser.add_argument("--context_length", type=int, default=256,
                        help="Context length for tokenization")
    parser.add_argument("--output_dir", type=str, default="tokenized_data",
                        help="Output directory for tokenized data")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation"],
                        help="Dataset splits to tokenize (e.g., train validation)")
    parser.add_argument("--stride_ratio", type=float, default=0.5,
                        help="Stride as ratio of context_length (default 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-tokenization even if cache exists")
    return parser.parse_args()


def get_cache_path(dataset_name: str, tokenizer_name: str, context_length: int,
                   split: str, output_dir: str) -> str:
    """Generate a unique cache filename based on parameters."""
    cache_key = f"{dataset_name}_{tokenizer_name}_{context_length}_{split}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_filename = f"tokens_{split}_{cache_hash}.npy"
    return os.path.join(output_dir, cache_filename)


def tokenize_dataset(dataset_name: str, tokenizer, context_length: int,
                     split: str, output_dir: str, stride_ratio: float = 0.5,
                     force: bool = False) -> str:
    """
    Tokenize a dataset split and save to cache.

    Args:
        dataset_name: Hugging Face dataset name
        tokenizer: Tokenizer instance
        context_length: Maximum sequence length
        split: Dataset split name
        output_dir: Output directory for cached data
        stride_ratio: Stride as ratio of context_length
        force: Force re-tokenization even if cache exists

    Returns:
        Path to the cached tokenized data
    """
    os.makedirs(output_dir, exist_ok=True)

    cache_path = get_cache_path(dataset_name, tokenizer.name_or_path,
                                context_length, split, output_dir)

    # Check if cached data exists
    if os.path.exists(cache_path) and not force:
        logger.info(f"Cached data already exists at: {cache_path}")
        logger.info("Use --force to re-tokenize")
        return cache_path

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name} - {split} split")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Compute stride
    stride = int(context_length * stride_ratio)

    # Define tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            stride=stride,
        )

        # Only keep complete sequences
        input_ids_list = []
        for ids in tokenized["input_ids"]:
            if len(ids) == context_length:
                input_ids_list.append(ids)

        return {"input_ids": input_ids_list}

    # Tokenize dataset
    logger.info(f"Tokenizing with context_length={context_length}, stride={stride}...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Convert to numpy array
    logger.info("Converting to numpy array...")
    all_tokens = []
    for item in tokenized_dataset:
        all_tokens.extend(item["input_ids"])

    tokens_array = np.array(all_tokens, dtype=np.uint16).flatten()

    # Save to cache
    np.save(cache_path, tokens_array)
    logger.info(f"Tokenized data saved to: {cache_path}")
    logger.info(f"Total tokens: {len(tokens_array):,}")
    logger.info(f"Total sequences: {len(all_tokens):,}")

    return cache_path


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Dataset Tokenization")
    logger.info("=" * 50)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Context length: {args.context_length}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 50)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Tokenize each split
    for split in args.splits:
        logger.info(f"\nProcessing split: {split}")
        tokenize_dataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            context_length=args.context_length,
            split=split,
            output_dir=args.output_dir,
            stride_ratio=args.stride_ratio,
            force=args.force,
        )

    logger.info("\n" + "=" * 50)
    logger.info("Tokenization complete!")
    logger.info("=" * 50)

    # Print summary of cached files
    logger.info("\nCached files:")
    for split in args.splits:
        cache_path = get_cache_path(args.dataset, args.tokenizer,
                                     args.context_length, split, args.output_dir)
        if os.path.exists(cache_path):
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            logger.info(f"  {cache_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
