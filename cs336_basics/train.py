import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# ========== Import base components from original model ==========
from cs336_basics.model import (
    Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding,
    softmax, scaled_dot_product_attention, CasualMultiheadSelfAttention,
    TransformerBlock, TransformerLM
)

# ========== Import mHC model ==========
from cs336_basics.mhc_model import (
    mHCTransformerLM, mHCTransformerBlock
)

from cs336_basics.optimizer import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.optimizer import cosine_lr_schedule
from cs336_basics.optimizer import gradient_clipping

from cs336_basics.load import get_batch
from cs336_basics.load import save_checkpoint
from cs336_basics.load import load_checkpoint


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train language model (supports original Transformer and mHC)")

    # ========== Model selection parameters ==========
    parser.add_argument("--model_type", type=str, default="original", choices=["original", "mhc"],
                        help="Model type: original (original Transformer) or mhc (Manifold-Constrained Hyper-Connections)")
    parser.add_argument("--expansion_rate", type=int, default=4,
                        help="mHC residual stream expansion rate n (default 4, only valid when model_type=mhc)")

    parser.add_argument("--train_data", type=str, required=False, default="roneneldan/TinyStories",
                        help="Hugging Face dataset name")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Validation data file path (optional, auto-handled if using HF dataset)")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Vocabulary size")

    parser.add_argument("--n_layer", type=int, default=12,
                        help="Number of Transformer layers")
    parser.add_argument("--n_head", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=768,
                        help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum context length")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--context_length", type=int, default=128,
                        help="Context length")
    parser.add_argument("--max_iters", type=int, default=100000,
                        help="Maximum training iterations")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Evaluation interval")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Checkpoint save interval")
    parser.add_argument("--eval_iters", type=int, default=200,
                        help="Number of iterations for evaluation")

    parser.add_argument("--learning_rate", type=float, default=6e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Learning rate warmup iterations")

    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Training device")

    parser.add_argument("--tokenizer_name", type=str, default="gpt2",
                        help="Tokenizer name (from Hugging Face)")
    parser.add_argument("--tokenized_data_dir", type=str, default="tokenized_data",
                        help="Directory to cache tokenized data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process for tokenization (default: all)")

    return parser.parse_args()

def load_and_prepare_hf_dataset(dataset_name, tokenizer, context_length, tokenized_data_dir, split="train", max_samples=None):
    """
    Load and prepare HuggingFace dataset with caching support.
    If tokenized data already exists in the cache directory, load it directly.
    Otherwise, tokenize and save to cache for future use.
    Uses streaming mode with immediate disk writes to avoid memory issues.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(tokenized_data_dir, exist_ok=True)

    # Generate a unique filename based on dataset name, tokenizer, context_length, and split
    # Use a hash to handle special characters in dataset names
    import hashlib
    cache_key = f"{dataset_name}_{tokenizer.name_or_path}_{context_length}_{split}"
    if max_samples is not None:
        cache_key += f"_max{max_samples}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_filename = f"tokens_{split}_{cache_hash}.npy"
    cache_path = os.path.join(tokenized_data_dir, cache_filename)

    # Check if cached data exists
    if os.path.exists(cache_path):
        logger.info(f"Found cached tokenized data at: {cache_path}")
        logger.info(f"Loading cached data directly (skipping tokenization)...")
        return cache_path

    # No cached data, need to tokenize
    logger.info(f"No cached data found. Loading Hugging Face dataset: {dataset_name} - {split} split")

    # Load dataset with streaming mode to avoid memory issues
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    logger.info(f"Loading dataset in streaming mode...")

    # Tokenize with streaming and immediate writes
    stride = context_length // 2
    temp_path = cache_path + ".tmp"
    sample_count = 0
    total_sequences = 0

    logger.info(f"Tokenizing with context_length={context_length}, stride={stride}...")
    if max_samples:
        logger.info(f"Processing max {max_samples:,} samples")

    from tqdm import tqdm

    # Open file for writing
    f = open(temp_path, 'wb')

    try:
        pbar = tqdm(desc="Tokenizing", unit="samples")

        for sample in dataset:
            if max_samples and sample_count >= max_samples:
                break

            tokenized = tokenizer(
                sample["text"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                stride=stride,
            )

            # Write each sequence immediately to disk
            for ids in tokenized["input_ids"]:
                if len(ids) == context_length:
                    arr = np.array(ids, dtype=np.uint16)
                    f.write(arr.tobytes())
                    total_sequences += 1

            sample_count += 1
            pbar.update(1)

        pbar.close()

    finally:
        f.close()

    logger.info(f"Processed {sample_count:,} samples")

    # Convert to final numpy format
    logger.info("Converting to final format...")
    _convert_to_npy(temp_path, cache_path)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    final_tokens = os.path.getsize(cache_path) // 2  # uint16 = 2 bytes
    logger.info(f"Tokenized data cached to: {cache_path}, total tokens: {final_tokens:,}")

    return cache_path


def _convert_to_npy(temp_path: str, output_path: str):
    """Convert raw binary file to numpy format."""
    with open(temp_path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint16)
    np.save(output_path, arr)

def load_data_mmap(data_path):
    logger.info(f"Loading data: {data_path}")
    if data_path.endswith('.npy'):
        data = np.load(data_path, mmap_mode='r')
    else:
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    logger.info(f"Data loaded, shape: {data.shape}")
    return data

def create_model(args, device):
    """Create model based on arguments"""

    # Compute FFN dimension (shared by both models)
    d_ff = int(4 * args.n_embd)
    d_ff = ((d_ff + 63) // 64) * 64  # Align to multiple of 64

    if args.model_type == "mhc":
        logger.info("=" * 50)
        logger.info("Using mHC architecture (Manifold-Constrained Hyper-Connections)")
        logger.info(f"Expansion rate n = {args.expansion_rate}")
        
        model = mHCTransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.n_embd,
            num_heads=args.n_head,
            d_ff=d_ff,
            num_layers=args.n_layer,
            context_length=args.block_size,
            expansion_rate=args.expansion_rate,
            theta=10000.0,
            dropout=args.dropout,
            device=device
        )
        
        if args.grad_clip == 1.0:
            logger.info("Note: mHC architecture is more stable, you can try increasing --grad_clip to 2.0 or 5.0")

    else:  # original
        logger.info("=" * 50)
        logger.info("Using original Transformer architecture")
        
        model = TransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.n_embd,
            num_heads=args.n_head,
            d_ff=d_ff,
            num_layers=args.n_layer,
            context_length=args.block_size,
            theta=10000.0,
            device=device,
            dtype=torch.float32
        )
    
    return model

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_length, eval_iters, device, model_type):
    """Estimate loss, pass token_positions based on model type"""
    model.eval()
    out = {}

    for split, data in [('train', train_data), ('val', val_data)]:
        if data is None:
            continue

        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)

            if model_type == "mhc":
                # mHC model requires token_positions
                token_positions = torch.arange(context_length, device=device)
                token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
                logits = model(X, token_positions)
            else:
                # Original model doesn't need token_positions
                logits = model(X)
                
            loss = cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    logger.info("=" * 50)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer: {args.tokenizer_name}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    logger.info("=" * 50)
    logger.info("Loading Hugging Face dataset...")
    # Load training data
    train_data_path = load_and_prepare_hf_dataset(
        args.train_data,
        tokenizer,
        args.context_length,
        args.tokenized_data_dir,
        split="train",
        max_samples=args.max_samples
    )
    train_data = load_data_mmap(train_data_path)

    # Load validation data
    val_data = None
    if args.val_data is None:
        # Use default validation set
        try:
            val_data_path = load_and_prepare_hf_dataset(
                args.train_data,
                tokenizer,
                args.context_length,
                args.tokenized_data_dir,
                split="validation",
                max_samples=None  # Don't limit validation samples
            )
            val_data = load_data_mmap(val_data_path)
        except Exception as e:
            logger.warning(f"Cannot load validation set: {e}, will use training set for evaluation only")
            val_data = None
    else:
        val_data = load_data_mmap(args.val_data)

    logger.info("=" * 50)
    logger.info("Initializing model...")
    model = create_model(args, device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    model.to(device)

    logger.info("Initializing optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    start_iter = 0
    if args.resume_from:
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resuming from iteration {start_iter}")

    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info(f"Maximum iterations: {args.max_iters}")
    logger.info(f"Model type: {args.model_type}")
    logger.info("=" * 50)

    model.train()
    t0 = time.time()
    train_losses = []

    for iter in range(start_iter, args.max_iters):
        # Compute learning rate
        lr = cosine_lr_schedule(iter, args.learning_rate, args.warmup_iters, args.max_iters, args.learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass based on model type
        if args.model_type == "mhc":
            token_positions = torch.arange(args.context_length, device=device)
            token_positions = token_positions.unsqueeze(0).expand(args.batch_size, -1)
            logits = model(X, token_positions)
        else:
            logits = model(X)

        loss = cross_entropy(logits, Y)

        optimizer.zero_grad()
        loss.backward()

        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        train_losses.append(loss.item())

        if iter % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            avg_loss = np.mean(train_losses[-args.log_interval:])
            logger.info(
                f"iteration {iter:6d} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"time {dt*1000:.2f}ms/iter"
            )

        if iter % args.eval_interval == 0 and val_data is not None and iter > 0:
            logger.info("Running evaluation...")
            losses = estimate_loss(
                model, train_data, val_data,
                args.batch_size, args.context_length,
                args.eval_iters, device,
                args.model_type  # Pass model type
            )
            logger.info(
                f"Eval - iteration {iter:6d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        if iter % args.save_interval == 0 and iter > 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{iter}.pt")
            logger.info(f"Saving checkpoint: {checkpoint_path}")
            save_checkpoint(model, optimizer, iter, checkpoint_path)

    logger.info("=" * 50)
    logger.info("Training complete!")
    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pt")
    logger.info(f"Saving final model: {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_iters - 1, final_checkpoint_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(0)