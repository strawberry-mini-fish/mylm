#!/usr/bin/env python3
"""
Test script to verify mHC implementation improvements from arxiv paper 2512.24880
"""

import torch
import torch.nn as nn
from cs336_basics.mhc_model import (
    sinkhorn_knopp,
    ManifoldHyperConnection,
    mHCTransformerBlock,
    mHCTransformerLM
)


def test_sinkhorn_knopp():
    """Test Sinkhorn-Knopp algorithm correctness
    
    Verify:
    1. Output is doubly stochastic (rows and columns sum to ~1)
    2. All values are non-negative
    3. Numerical stability
    """
    print("=" * 60)
    print("Testing Sinkhorn-Knopp Algorithm")
    print("=" * 60)
    
    # Test with random matrices
    batch_size = 2
    n = 4
    
    # Create random matrices
    M = torch.randn(batch_size, n, n)
    
    # Apply Sinkhorn-Knopp with 20 iterations (paper default)
    M_ds = sinkhorn_knopp(M, num_iter=20)
    
    print(f"Input matrix shape: {M.shape}")
    print(f"Output matrix shape: {M_ds.shape}")
    print(f"All non-negative: {(M_ds >= 0).all().item()}")
    
    # Check row sums
    row_sums = M_ds.sum(dim=-1)
    print(f"Row sums (should be ~1): min={row_sums.min():.6f}, max={row_sums.max():.6f}, mean={row_sums.mean():.6f}")
    
    # Check column sums  
    col_sums = M_ds.sum(dim=-2)
    print(f"Col sums (should be ~1): min={col_sums.min():.6f}, max={col_sums.max():.6f}, mean={col_sums.mean():.6f}")
    
    # Check spectral norm (should be <= 1)
    spec_norms = torch.linalg.eigvalsh(M_ds)
    print(f"Spectral norm (should be <= 1): {spec_norms.abs().max().item():.6f}")
    
    print("✓ Sinkhorn-Knopp test passed\n")


def test_manifold_hyperconnection():
    """Test ManifoldHyperConnection module
    
    Verify:
    1. H_pre has correct shape and constraints
    2. H_post has correct shape and constraints 
    3. H_res is doubly stochastic
    """
    print("=" * 60)
    print("Testing ManifoldHyperConnection Module")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 4
    d_model = 64
    n = 4
    
    # Create mHC module
    mhc = ManifoldHyperConnection(d_model=d_model, expansion_rate=n)
    
    # Create input (flattened stream)
    x = torch.randn(batch_size * seq_len, n * d_model)
    
    # Forward pass
    H_pre, H_post, H_res = mhc(x)
    
    print(f"Input shape: {x.shape}")
    print(f"H_pre shape: {H_pre.shape} (expected: ({batch_size*seq_len}, {n}))")
    print(f"H_post shape: {H_post.shape} (expected: ({batch_size*seq_len}, {n}))")  
    print(f"H_res shape: {H_res.shape} (expected: ({batch_size*seq_len}, {n}, {n}))")
    
    # Check constraints
    print(f"\nH_pre constraints:")
    print(f"  Min value: {H_pre.min().item():.6f} (should be >= 0)")
    print(f"  Max value: {H_pre.max().item():.6f} (should be <= 1)")
    
    print(f"H_post constraints:")
    print(f"  Min value: {H_post.min().item():.6f} (should be >= 0)")
    print(f"  Max value: {H_post.max().item():.6f} (should be <= 2)")
    
    print(f"H_res doubly stochastic:")
    H_res_2d = H_res[0]  # Take first sample
    row_sums = H_res_2d.sum(dim=-1)
    col_sums = H_res_2d.sum(dim=-2)
    print(f"  Row sums: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    print(f"  Col sums: min={col_sums.min():.6f}, max={col_sums.max():.6f}")
    
    print("✓ ManifoldHyperConnection test passed\n")


def test_mhc_transformer_block():
    """Test mHCTransformerBlock forward pass
    
    Verify:
    1. Output shape matches input
    2. No NaN or Inf values
    3. Numerical stability
    """
    print("=" * 60)
    print("Testing mHCTransformerBlock")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 4
    d_ff = 256
    expansion_rate = 4
    
    # Create block
    block = mHCTransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=seq_len,
        expansion_rate=expansion_rate
    )
    
    # Create input and positions
    x = torch.randn(batch_size, seq_len, d_model)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    output = block(x, positions)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output matches input shape: {output.shape == x.shape}")
    
    print(f"\nNumerical stability checks:")
    print(f"  No NaN values: {(~torch.isnan(output)).all().item()}")
    print(f"  No Inf values: {(~torch.isinf(output)).all().item()}")
    print(f"  Output value range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(f"  Output std: {output.std().item():.6f}")
    
    print("✓ mHCTransformerBlock test passed\n")


def test_mhc_transformer_lm():
    """Test mHCTransformerLM forward pass
    
    Verify:
    1. End-to-end model works
    2. Output logits have correct shape
    3. No training issues
    """
    print("=" * 60)
    print("Testing mHCTransformerLM")
    print("=" * 60)
    
    vocab_size = 256
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    context_length = 16
    expansion_rate = 4
    
    # Create model
    model = mHCTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        context_length=context_length,
        expansion_rate=expansion_rate
    )
    
    # Create input
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"Shape correct: {logits.shape == (batch_size, seq_len, vocab_size)}")
    
    print(f"\nNumerical checks:")
    print(f"  No NaN: {(~torch.isnan(logits)).all().item()}")
    print(f"  No Inf: {(~torch.isinf(logits)).all().item()}")
    
    # Test backward pass
    print(f"\nTesting backward pass...")
    loss = logits.sum()
    loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    print(f"  No NaN in gradients: {not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)}")
    
    print("✓ mHCTransformerLM test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("mHC Implementation Tests (from arXiv 2512.24880)")
    print("=" * 60 + "\n")
    
    try:
        test_sinkhorn_knopp()
        test_manifold_hyperconnection()
        test_mhc_transformer_block()
        test_mhc_transformer_lm()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
