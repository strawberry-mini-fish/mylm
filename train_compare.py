#!/usr/bin/env python3
"""
Train original Transformer and mHC models alternately,
logging both losses to the same TensorBoard for comparison.
"""

import os
import sys
import argparse
import time
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

# Try to import TensorBoard, make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Note: TensorBoard not installed, will use file logging only")

from cs336_basics.model import TransformerLM
from cs336_basics.mhc_model import mHCTransformerLM
from cs336_basics.optimizer import cross_entropy, AdamW, cosine_lr_schedule, gradient_clipping
from cs336_basics.load import get_batch, save_checkpoint


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare training of original vs mHC Transformer")

    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=10000, help="Total iterations")
    parser.add_argument("--switch_interval", type=int, default=1000, help="Switch models every N iterations")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--mhc_learning_rate", type=float, default=1e-4, help="Learning rate for mHC")
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # mHC specific
    parser.add_argument("--expansion_rate", type=int, default=2)

    # Data
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--train_data", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--max_samples", type=int, default=50000)

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, default="runs/compare")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def load_data(args, tokenizer):
    """Load and cache tokenized data."""
    from cs336_basics.train import load_and_prepare_hf_dataset, load_data_mmap

    train_data_path = load_and_prepare_hf_dataset(
        args.train_data, tokenizer, args.context_length,
        "tokenized_data", split="train", max_samples=args.max_samples
    )
    return load_data_mmap(train_data_path)


def create_model(args, model_type, device):
    """Create model based on type."""
    d_ff = int(4 * args.n_embd)
    d_ff = ((d_ff + 63) // 64) * 64

    if model_type == "mhc":
        return mHCTransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.n_embd,
            num_heads=args.n_head,
            d_ff=d_ff,
            num_layers=args.n_layer,
            context_length=args.context_length,
            expansion_rate=args.expansion_rate,
            device=device,
            dtype=torch.float32
        )
    else:
        return TransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.n_embd,
            num_heads=args.n_head,
            d_ff=d_ff,
            num_layers=args.n_layer,
            context_length=args.context_length,
            device=device,
            dtype=torch.float32
        )


def train_step(model, optimizer, train_data, args, device, model_type):
    """Single training step, returns loss."""
    X, Y = get_batch(train_data, args.batch_size, args.context_length, device)

    if model_type == "mhc":
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

    return loss.item()


def main():
    args = parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # TensorBoard writer (optional)
    if HAS_TENSORBOARD:
        writer = SummaryWriter(args.tensorboard_dir)
        logger.info(f"TensorBoard logs: {args.tensorboard_dir}")
    else:
        writer = None
        logger.info("TensorBoard not available, using file logging")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data (shared between both models)
    logger.info("Loading data...")
    train_data = load_data(args, tokenizer)

    # Create both models
    logger.info("=" * 60)
    logger.info("Creating models...")

    model_original = create_model(args, "original", device)
    model_mhc = create_model(args, "mhc", device)

    n_params_original = sum(p.numel() for p in model_original.parameters())
    n_params_mhc = sum(p.numel() for p in model_mhc.parameters())
    logger.info(f"Original model: {n_params_original / 1e6:.2f}M parameters")
    logger.info(f"mHC model: {n_params_mhc / 1e6:.2f}M parameters")

    # Create optimizers
    optimizer_original = AdamW(model_original.parameters(), lr=args.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    optimizer_mhc = AdamW(model_mhc.parameters(), lr=args.mhc_learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

    # Track iterations per model
    iter_original = 0
    iter_mhc = 0

    # Loss history
    losses_original = []
    losses_mhc = []

    logger.info("=" * 60)
    logger.info(f"Starting alternating training...")
    logger.info(f"Total iterations: {args.max_iters}")
    logger.info(f"Switch interval: {args.switch_interval}")
    logger.info("=" * 60)

    global_step = 0
    current_model = "original"  # Start with original

    while iter_original < args.max_iters or iter_mhc < args.max_iters:
        # Determine which model to train
        if current_model == "original" and iter_original >= args.max_iters:
            current_model = "mhc"
        elif current_model == "mhc" and iter_mhc >= args.max_iters:
            current_model = "original"

        # Select model and optimizer
        if current_model == "original":
            model = model_original
            optimizer = optimizer_original
            lr = args.learning_rate
            model_iter = iter_original
        else:
            model = model_mhc
            optimizer = optimizer_mhc
            lr = args.mhc_learning_rate
            model_iter = iter_mhc

        # Compute learning rate
        current_lr = cosine_lr_schedule(model_iter, lr, args.warmup_iters, args.max_iters, lr * 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training step
        model.train()
        loss = train_step(model, optimizer, train_data, args, device, current_model)

        # Update counters
        if current_model == "original":
            iter_original += 1
            losses_original.append(loss)
        else:
            iter_mhc += 1
            losses_mhc.append(loss)

        # Logging
        if writer is not None:
            writer.add_scalar(f"loss/{current_model}", loss, global_step)
            writer.add_scalar(f"lr/{current_model}", current_lr, global_step)

        if global_step % args.log_interval == 0:
            logger.info(f"[{current_model:8s}] step {global_step:5d} | "
                       f"original iter {iter_original:5d} | mHC iter {iter_mhc:5d} | "
                       f"loss {loss:.4f} | lr {current_lr:.2e}")

        global_step += 1

        # Switch models every switch_interval
        if global_step % args.switch_interval == 0:
            if current_model == "original" and iter_mhc < args.max_iters:
                current_model = "mhc"
            elif current_model == "mhc" and iter_original < args.max_iters:
                current_model = "original"
            logger.info(f"Switching to {current_model} model")

    # Save final models
    logger.info("=" * 60)
    logger.info("Training complete!")

    save_dir = os.path.join(args.output_dir, "original")
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model_original, optimizer_original, iter_original, os.path.join(save_dir, "final_model.pt"))

    save_dir = os.path.join(args.output_dir, "mhc")
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model_mhc, optimizer_mhc, iter_mhc, os.path.join(save_dir, "final_model.pt"))

    logger.info(f"Original model final loss: {np.mean(losses_original[-100:]):.4f}")
    logger.info(f"mHC model final loss: {np.mean(losses_mhc[-100:]):.4f}")

    if writer is not None:
        writer.close()

    if HAS_TENSORBOARD:
        logger.info(f"View TensorBoard: tensorboard --logdir {args.tensorboard_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
