#!/usr/bin/env python3
"""
Train both original Transformer and mHC models sequentially,
logging both losses to the same TensorBoard for comparison.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
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
    logger.warning("TensorBoard not installed, will use file logging only")

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
    parser.add_argument("--max_iters", type=int, default=10000, help="Iterations per model")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--mhc_learning_rate", type=float, default=1e-4, help="Learning rate for mHC")
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=500)
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


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, model_type, eval_iters=100):
    """Evaluate model loss."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device)
        if model_type == "mhc":
            token_positions = torch.arange(context_length, device=device)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
            logits = model(X, token_positions)
        else:
            logits = model(X)
        loss = cross_entropy(logits, Y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_model(args, model_type, train_data, tokenizer, writer, global_step_offset, device):
    """Train a single model and return final loss."""
    logger.info("=" * 60)
    logger.info(f"Training {model_type.upper()} model")
    logger.info("=" * 60)

    # Create model
    model = create_model(args, model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.2f}M")

    # Learning rate
    lr = args.learning_rate if model_type == "original" else args.mhc_learning_rate

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    # Training loop
    model.train()
    train_losses = []

    for iter in range(args.max_iters):
        # Learning rate schedule
        current_lr = cosine_lr_schedule(iter, lr, args.warmup_iters, args.max_iters, lr * 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Get batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        if model_type == "mhc":
            token_positions = torch.arange(args.context_length, device=device)
            token_positions = token_positions.unsqueeze(0).expand(args.batch_size, -1)
            logits = model(X, token_positions)
        else:
            logits = model(X)

        loss = cross_entropy(logits, Y)
        train_losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging
        global_step = global_step_offset + iter
        if writer is not None:
            writer.add_scalar(f"loss/{model_type}", loss.item(), global_step)
            writer.add_scalar(f"lr/{model_type}", current_lr, global_step)

        if iter % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:])
            logger.info(f"[{model_type}] iter {iter:5d}/{args.max_iters} | loss {avg_loss:.4f} | lr {current_lr:.2e}")

        # Evaluation
        if iter % args.eval_interval == 0 and iter > 0:
            eval_loss = evaluate(model, train_data, args.batch_size, args.context_length, device, model_type)
            if writer is not None:
                writer.add_scalar(f"eval_loss/{model_type}", eval_loss, global_step)
            logger.info(f"[{model_type}] eval loss: {eval_loss:.4f}")

    # Save final model
    save_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(save_dir, "final_model.pt"))

    final_loss = np.mean(train_losses[-100:])
    logger.info(f"[{model_type}] Training complete! Final loss: {final_loss:.4f}")

    return global_step_offset + args.max_iters


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

    # Train original model first
    global_step = 0
    global_step = train_model(args, "original", train_data, tokenizer, writer, global_step, device)

    # Train mHC model
    global_step = train_model(args, "mhc", train_data, tokenizer, writer, global_step, device)

    if writer is not None:
        writer.close()
    logger.info("=" * 60)
    logger.info("Comparison training complete!")
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
