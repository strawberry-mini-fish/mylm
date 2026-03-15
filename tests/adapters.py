from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.model import Linear, Embedding


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    linear_layer=Linear(d_in,d_out)
    linear_layer.load_state_dict({'W':weights})
    with torch.no_grad():
        output=linear_layer(in_features)
    return output



def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_modle"]:
    embedding_layer=Embedding(vocab_size, d_model)
    embedding_layer.load_state_dict({'weight':weights})
    with torch.no_grad():
        output=embedding_layer(token_ids)
    return output

    


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    print("="*50)
    print("run_swiglu 被调用:")
    print(f"  d_model: {d_model}")
    print(f"  w1.shape: {w1_weight.shape}")
    print(f"  w2.shape: {w2_weight.shape}")
    print(f"  w3.shape: {w3_weight.shape}")
    print(f"  in_features.shape: {in_features.shape}")
    d_ff = w1_weight.shape[0]
    print(f"  推断 d_ff = {d_ff}")
    from cs336_basics.model import SwiGLU
    swiglu=SwiGLU(d_model,d_ff)
    state_dict = {
        'W1.W': w1_weight,
        'W2.W': w2_weight,
        'W3.W': w3_weight,
    }
    swiglu.load_state_dict(state_dict)
    swiglu.eval()
    with torch.no_grad():
        output=swiglu(in_features)
    return output



def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    from cs336_basics.model import scaled_dot_product_attention 
    return scaled_dot_product_attention(Q,K,V,mask)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError
    

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    from cs336_basics.model import CasualMultiheadSelfAttention,RotaryPositionalEmbedding
    from einops import rearrange
    multihead_attention = CasualMultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
    multihead_attention.load_state_dict({
        "linearQ.W": q_proj_weight,
        "linearK.W": k_proj_weight,
        "linearV.W": v_proj_weight,
        "linearO.W": o_proj_weight,
    })
    return multihead_attention(in_features, token_positions)
    

   
   
   


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    from cs336_basics.model import RotaryPositionalEmbedding
    rope=RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device
    )
    with torch.no_grad():
        output=rope(in_query_or_key,token_positions)
    return output

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from cs336_basics.model import TransformerBlock, CasualMultiheadSelfAttention, RMSNorm, SwiGLU
    from einops import rearrange
    import torch
    
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta
    )
    
    state_dict={}
    state_dict["norm1.weight"] = weights["ln1.weight"]
    state_dict["norm2.weight"] = weights["ln2.weight"]
    state_dict["self_attention.linearQ.W"] = weights["attn.q_proj.weight"]
    state_dict["self_attention.linearK.W"] = weights["attn.k_proj.weight"]
    state_dict["self_attention.linearV.W"] = weights["attn.v_proj.weight"]
    state_dict["self_attention.linearO.W"] = weights["attn.output_proj.weight"]
    state_dict["ffn.W1.W"] = weights["ffn.w1.weight"]  # 键名大写，值从小写获取
    state_dict["ffn.W3.W"] = weights["ffn.w3.weight"]  # 键名大写，值从小写获取
    state_dict["ffn.W2.W"] = weights["ffn.w2.weight"]  # 键名大写，值从小写获取

    
    transformer_block.load_state_dict(state_dict)

    

    batch_size, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=in_features.device)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    with torch.no_grad():
        output = transformer_block(in_features, token_positions)
    
    return output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from cs336_basics.model import TransformerLM
    print("\n" + "="*60)
    print("TRANSFORMER LM DEBUG INFO")
    print("="*60)
    
    print(f"vocab_size: {vocab_size}")
    print(f"context_length: {context_length}")
    print(f"d_model: {d_model}")
    print(f"num_layers: {num_layers}")
    print(f"num_heads: {num_heads}")
    print(f"d_ff: {d_ff}")
    print(f"rope_theta: {rope_theta}")
    print(f"in_indices shape: {in_indices.shape}")

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        context_length=context_length,
        theta=rope_theta,
    )
    model.eval() 
    state_dict = {}
    state_dict["token_embedding.weight"] = weights["token_embeddings.weight"]
    for i in range(num_layers):
        prefix = f"blocks.{i}."
        
        # 第一个RMSNorm
        state_dict[f"{prefix}norm1.weight"] = weights[f"layers.{i}.ln1.weight"]
        
        # 第二个RMSNorm
        state_dict[f"{prefix}norm2.weight"] = weights[f"layers.{i}.ln2.weight"]
        
        # 自注意力层的权重
        state_dict[f"{prefix}self_attention.linearQ.W"] = weights[f"layers.{i}.attn.q_proj.weight"]
        state_dict[f"{prefix}self_attention.linearK.W"] = weights[f"layers.{i}.attn.k_proj.weight"]
        state_dict[f"{prefix}self_attention.linearV.W"] = weights[f"layers.{i}.attn.v_proj.weight"]
        state_dict[f"{prefix}self_attention.linearO.W"] = weights[f"layers.{i}.attn.output_proj.weight"]

        w1 = weights[f"layers.{i}.ffn.w1.weight"]  # [d_model, d_ff]
        w3 = weights[f"layers.{i}.ffn.w3.weight"]  # [d_model, d_ff]
        w2 = weights[f"layers.{i}.ffn.w2.weight"]  # [d_ff, d_model]
        
        # 根据你的Linear类实现决定是否需要转置
        # 如果你的Linear类期望 [out_features, in_features]，则需要转置
        state_dict[f"{prefix}ffn.W1.W"] = w1.T  # [d_ff, d_model]
        state_dict[f"{prefix}ffn.W3.W"] = w3.T  # [d_ff, d_model]
        state_dict[f"{prefix}ffn.W2.W"] = w2.T  # [d_model, d_ff]

    state_dict["ln_final.weight"] = weights["ln_final.weight"]
    
    # 3.4 输出投影层 (lm_head)
    state_dict["output_projection.W"] = weights["lm_head.weight"]
    
    # 4. 打印权重形状以便调试
    print("\n--- Weight shapes after mapping ---")
    for i in range(min(num_layers, 1)):  # 只打印第一层
        print(f"\nLayer {i}:")
        print(f"  W1: {state_dict[f'blocks.{i}.ffn.W1.W'].shape}")
        print(f"  W3: {state_dict[f'blocks.{i}.ffn.W3.W'].shape}")
        print(f"  W2: {state_dict[f'blocks.{i}.ffn.W2.W'].shape}")

    try:
        model.load_state_dict(state_dict)
        print("\n✓ Weights loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading weights: {e}")
        # 如果转置后还不行，尝试不转置
        print("\nTrying without transpose...")
        state_dict_no_transpose = {}
        for k, v in state_dict.items():
            if 'ffn' in k and 'W' in k:
                state_dict_no_transpose[k] = v.T  # 再转置回去
            else:
                state_dict_no_transpose[k] = v
        model.load_state_dict(state_dict_no_transpose)
        print("✓ Weights loaded successfully (without transpose)")
        batch_size, seq_len = in_indices.shape
    token_positions = torch.arange(seq_len, device=in_indices.device)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    
    # 7. 前向传播
    with torch.no_grad():
        logits = model(in_indices, token_positions)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Output stats: mean={logits.mean():.4f}, std={logits.std():.4f}")
    
    return logits
    



def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.model import RMSNorm
    rmsnorm=RMSNorm(d_model,eps=eps)
    rmsnorm.load_state_dict({'weight':weights})
    rmsnorm.eval()
    with torch.no_grad():
        output=rmsnorm(in_features)
    return output

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from cs336_basics.load import get_batch
    return get_batch(dataset,batch_size,context_length,device)
    


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from cs336_basics.model import softmax
    return softmax(in_features,dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
   from cs336_basics.train import cross_entropy
   return cross_entropy(inputs,targets)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    from cs336_basics.train import gradient_clipping
    params = parameters
    max_norm_value=max_l2_norm
    return gradient_clipping(params, max_norm_value)

def get_adamw_cls() -> Any:
    from cs336_basics.train import AdamW
    
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    from cs336_basics.train import cosine_lr_schedule
    return cosine_lr_schedule(
        t=it,
        alpha_max=max_learning_rate,
        alpha_min=min_learning_rate,
        T_w=warmup_iters,
        T_c=cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    from cs336_basics.load import save_checkpoint
    save_checkpoint(model,optimizer,iteration,out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    from cs336_basics.load import load_checkpoint
    return load_checkpoint(src,model,optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
