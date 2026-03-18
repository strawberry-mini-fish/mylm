import torch
import torch.nn as nn
import math
from einops import einsum, reduce, rearrange, repeat
import torch.nn.functional as F
from typing import Optional

# ========== Basic components (reused from model.py) ==========
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (..., d_model) or (..., n*d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute RMS of the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms

        # Ensure weight dimension matches
        if x_norm.shape[-1] == self.d_model:
            output = x_norm * self.weight
        else:
            # For flattened input, repeat weight
            n = x_norm.shape[-1] // self.d_model
            weight_expanded = self.weight.unsqueeze(0).repeat(1, n)  # (1, n*d_model)
            output = x_norm * weight_expanded

        return output.to(in_dtype)

    @staticmethod
    def apply(x: torch.Tensor, dim: int):
        """Static method for RMSNorm in mHC"""
        eps = 1e-6
        x_float = x.to(torch.float32)
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
        result = x_float / rms
        # Clamp to prevent extreme values that cause NaN
        result = torch.clamp(result, min=-10, max=10)
        return result.to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            theoretical_ff = int(8/3 * d_model)
            self.d_ff = ((theoretical_ff + 63) // 64) * 64
        else:
            self.d_ff = d_ff
        
        print(f"d_model={d_model}, d_ff={self.d_ff} (theoretical={int(8/3*d_model)})")

        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        
    def forward(self, x):
        input_shape = x.shape
        act = self.W1(x)        # activation path
        act = F.silu(act)        # activation

        gate = self.W3(x)        # gate path (no activation)

        gated = act * gate       # GLU multiplication

        output = self.W2(gated)  # output projection

        assert output.shape == input_shape, \
            f"output shape {output.shape} should match input shape {input_shape}"
        return output
    
    def extra_repr(self):
        return f"d_model={self.d_model}, d_ff={self.d_ff}"

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert d_k % 2 == 0, "d_k must be even for rotation"
        self._build_cache(device)
        
    def _build_cache(self, device):
        positions = torch.arange(self.max_seq_len, device=device).float()
        dim_pairs = torch.arange(0, self.d_k, 2, device=device).float()
        freqs = 1.0 / self.theta ** (dim_pairs / self.d_k)
        angles = positions[:, None] * freqs[None, :]
        cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)

        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        seq_len = x.shape[-2]
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos_pair = cos[..., 0::2]
        sin_pair = sin[..., 0::2]

        # Add extra dimension for broadcasting with multi-head attention
        # x has shape (batch, num_heads, seq_len, d_k/2)
        # cos_pair/sin_pair have shape (batch, seq_len, d_k/2)
        # Need to unsqueeze to (batch, 1, seq_len, d_k/2) for broadcasting
        while cos_pair.dim() < x_even.dim():
            cos_pair = cos_pair.unsqueeze(1)
            sin_pair = sin_pair.unsqueeze(1)

        x_rotated_even = x_even * cos_pair - x_odd * sin_pair
        x_rotated_odd = x_even * sin_pair + x_odd * cos_pair

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x_softmax = x_exp / x_sum
    return x_softmax

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attention_weights = softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output

class CasualMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, 
                 RoPE: RotaryPositionalEmbedding | None = None, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.hd_k = self.hd_v = d_model
        self.d_k = self.hd_k // num_heads
        self.d_v = self.hd_v // num_heads
        if RoPE is not None:
            self.RoPE = RoPE
        else:
            self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        self.linearQ = Linear(d_model, self.hd_k, device, dtype)
        self.linearK = Linear(d_model, self.hd_k, device, dtype)
        self.linearV = Linear(d_model, self.hd_v, device, dtype)
        self.linearO = Linear(self.hd_v, d_model, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = self.linearQ(x)
        K = self.linearK(x)
        V = self.linearV(x)
        Q = rearrange(Q, "... l (h d_k) -> ... h l d_k", h=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "... l (h d_k) -> ... h l d_k", h=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "... l (h d_v) -> ... h l d_v", h=self.num_heads, d_v=self.d_v)

        seq_len = token_positions.shape[-1]
        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... h l d_v -> ... l (h d_v)")
        return self.linearO(attention)


# ========== mHC Core Components ==========
def sinkhorn_knopp(M: torch.Tensor, num_iter: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """
    Project matrix onto doubly stochastic matrix manifold using Sinkhorn-Knopp algorithm.
    From Section 4.2 of mHC paper: transforms input matrix to doubly stochastic matrix
    where all rows and columns sum to 1.
    
    M: input matrix, shape (..., n, n)
    num_iter: number of iterations (paper uses 20)
    eps: numerical stability epsilon
    Returns: doubly stochastic matrix
    """
    # Paper Eq. (9): Start with M^(0) = exp(tilde{H}^res)
    # Clamp M to prevent overflow in exp
    M_clamped = torch.clamp(M, min=-20, max=20)
    # Ensure all elements are positive via exponentiation
    M_pos = torch.exp(M_clamped)
    
    # Alternating row and column normalization
    # Paper: M^(t) = T_r(T_c(M^(t-1)))
    for _ in range(num_iter):
        # Row normalization: T_r makes row sums equal to 1
        row_sum = M_pos.sum(dim=-1, keepdim=True)
        M_pos = M_pos / (row_sum + eps)
        
        # Column normalization: T_c makes column sums equal to 1  
        col_sum = M_pos.sum(dim=-2, keepdim=True)
        M_pos = M_pos / (col_sum + eps)
    
    return M_pos

class ManifoldHyperConnection(nn.Module):
    """
    mHC: Manifold-Constrained Hyper-Connections
    Paper Section 4.2: Parameterization and Manifold Projection

    Each layer (Attention or FFN sub-layer) has independent H_pre, H_post, H_res parameters.
    This follows paper's design where each transformer layer has its own set of mappings.
    """
    def __init__(
        self,
        d_model: int,           # Original dimension C
        expansion_rate: int,     # Expansion rate n (n=4 in paper)
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n = expansion_rate
        self.stream_dim = self.n * d_model  # Residual stream dimension n*C

        # Linear projection in paper formula (7)
        # phi_pre, phi_post: R^{nC x n}
        self.phi_pre = nn.Parameter(
            torch.empty((self.stream_dim, self.n), device=device, dtype=dtype)
        )
        self.phi_post = nn.Parameter(
            torch.empty((self.stream_dim, self.n), device=device, dtype=dtype)
        )
        # phi_res: R^{nC x n^2}
        self.phi_res = nn.Parameter(
            torch.empty((self.stream_dim, self.n * self.n), device=device, dtype=dtype)
        )

        # Learnable gating factors alpha (scalar)
        # Paper: initialized to small values for stability
        self.alpha_pre = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))
        self.alpha_post = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))
        self.alpha_res = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))

        # Learnable biases b (static mappings)
        # Paper Eq. (7): b_pre, b_post ∈ R^{1×n}, b_res ∈ R^{n×n}
        self.b_pre = nn.Parameter(torch.zeros((1, self.n), device=device, dtype=dtype))
        self.b_post = nn.Parameter(torch.zeros((1, self.n), device=device, dtype=dtype))
        self.b_res = nn.Parameter(torch.zeros((self.n, self.n), device=device, dtype=dtype))

        # Initialize phi matrices
        self._init_weights()

    def _init_weights(self):
        # Use smaller std for numerical stability
        std = 0.02
        nn.init.trunc_normal_(self.phi_pre, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.phi_post, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.phi_res, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for mHC mapping computation.
        Paper Section 4.2: Parameterization and Manifold Projection

        x: input, shape (..., nC) - flattened residual stream vector x_vec_l
        Returns: (H_pre, H_post, H_res) - three constrained mappings
        """
        # Paper Eq. (7): Compute raw mappings with RMSNorm and dynamic+static components
        # x'_l = RMSNorm(vec(x_l))

        # RMSNorm on flattened input (operates on last dimension)
        x_norm = RMSNorm.apply(x, x.shape[-1])  # (..., nC)

        # Paper Eq. (7): Compute H_tilde (pre-constraint) matrices
        # H_pre_tilde = alpha_pre * (x_norm @ phi_pre) + b_pre
        H_pre_tilde = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre  # (..., n)

        # H_post_tilde = alpha_post * (x_norm @ phi_post) + b_post
        H_post_tilde = self.alpha_post * (x_norm @ self.phi_post) + self.b_post  # (..., n)

        # H_res_tilde = alpha_res * mat(x_norm @ phi_res) + b_res
        # where mat(...) reshapes R^{1×n^2} to R^{n×n}
        H_res_tilde_flat = self.alpha_res * (x_norm @ self.phi_res)  # (..., n^2)
        H_res_tilde = rearrange(H_res_tilde_flat, '... (n m) -> ... n m', n=self.n, m=self.n)  # (..., n, n)
        H_res_tilde = H_res_tilde + self.b_res

        # Paper Eq. (8): Apply manifold constraint projections
        # H_pre = sigma(H_pre_tilde) - sigmoid constraint (non-negative, range [0,1])
        H_pre = torch.sigmoid(H_pre_tilde)  # (..., n)

        # H_post = 2 * sigma(H_post_tilde) - sigmoid constraint scaled to [0,2]
        # Paper Section 4.1: "we impose non-negativity constraints on H_pre and H_post"
        # H_post scaled to [0,2] for expressivity
        H_post = 2 * torch.sigmoid(H_post_tilde)  # (..., n)

        # H_res = Sinkhorn-Knopp(H_res_tilde) - doubly stochastic constraint
        # Paper Section 4.1: "constrain H_res to be a doubly stochastic matrix"
        # Project onto Birkhoff polytope to ensure row and column sums = 1
        H_res = sinkhorn_knopp(H_res_tilde, num_iter=20)  # (..., n, n)

        return H_pre, H_post, H_res


class mHCTransformerBlock(nn.Module):
    """
    mHC-based Transformer Block

    Paper architecture:
    - Each Transformer block contains Attention and FFN as two separate "layers"
    - Each layer has its own independent mHC parameters (H_pre, H_post, H_res)
    - Pre-norm structure: RMSNorm is applied before each sub-layer

    Paper Eq. (3): x_{l+1} = H_res * x_l + (H_post)^T * F(H_pre * x_l, W_l)

    The residual stream dimension is n*C, maintained across layers.
    For interface with standard Transformer, we expand C -> n*C at input and aggregate n*C -> C at output.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        expansion_rate: int = 2,  # n from paper, using 2 for small models
        theta: float = 10000.0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.n = expansion_rate
        self.stream_dim = self.n * d_model

        # Pre-norm layers (applied before each sub-layer)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        # mHC for Attention sub-layer (each sub-layer has independent parameters)
        self.mhc_attn = ManifoldHyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            device=device,
            dtype=dtype
        )

        # mHC for FFN sub-layer (independent parameters)
        self.mhc_ffn = ManifoldHyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            device=device,
            dtype=dtype
        )

        # Attention layer: input/output dimension is d_model (C)
        self.self_attention = CasualMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )

        # FFN layer: input/output dimension is d_model (C)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _compute_mhc_mappings(self, x_stream: torch.Tensor, mhc_module: ManifoldHyperConnection):
        """
        Compute mHC mappings for a sub-layer.

        Args:
            x_stream: residual stream, shape (batch, seq_len, n, C)
            mhc_module: ManifoldHyperConnection module (attention or FFN specific)

        Returns:
            H_pre: shape (batch, seq_len, n)
            H_post: shape (batch, seq_len, n)
            H_res: shape (batch, seq_len, n, n)
        """
        batch_size, seq_len = x_stream.shape[:2]

        # Flatten to (b*s, nC) to compute H mappings
        x_flat = rearrange(x_stream, 'b s n c -> (b s) (n c)')  # (b*s, n*C)

        # Compute constrained mappings H_pre, H_post, H_res
        H_pre, H_post, H_res = mhc_module(x_flat)  # Shapes: (b*s, n), (b*s, n), (b*s, n, n)

        # Restore batch and seq_len dimensions
        H_pre = rearrange(H_pre, '(b s) n -> b s n', b=batch_size, s=seq_len)  # (b, s, n)
        H_post = rearrange(H_post, '(b s) n -> b s n', b=batch_size, s=seq_len)  # (b, s, n)
        H_res = rearrange(H_res, '(b s) n1 n2 -> b s n1 n2',
                         b=batch_size, s=seq_len, n1=self.n, n2=self.n)  # (b, s, n, n)

        return H_pre, H_post, H_res

    def _apply_mhc_sublayer(
        self,
        x_stream: torch.Tensor,
        H_pre: torch.Tensor,
        H_post: torch.Tensor,
        H_res: torch.Tensor,
        sublayer_fn
    ) -> torch.Tensor:
        """
        Apply mHC transformation for a sub-layer.

        Paper Eq. (3): x_{l+1} = H_res * x_l + (H_post)^T * F(H_pre * x_l, W_l)

        Args:
            x_stream: residual stream, shape (batch, seq_len, n, C)
            H_pre: pre-aggregation weights, shape (batch, seq_len, n)
            H_post: post-expansion weights, shape (batch, seq_len, n)
            H_res: residual mixing matrix, shape (batch, seq_len, n, n)
            sublayer_fn: function F (attention or FFN)

        Returns:
            Updated residual stream, shape (batch, seq_len, n, C)
        """
        # Step 1: H_pre * x_l - Aggregate n*C stream to C dimension
        # Paper: H_pre aggregates features from the n*C-dim stream into a C-dim layer input
        x_sublayer_in = einsum(H_pre, x_stream, 'b s n, b s n c -> b s c')  # (b, s, C)

        # Step 2: Apply sub-layer function F
        sublayer_out = sublayer_fn(x_sublayer_in)  # (b, s, C)

        # Step 3: (H_post)^T * F(...) - Expand C output back to n*C stream
        # Paper: H_post maps the layer output back onto the stream
        sublayer_stream = einsum(H_post, sublayer_out, 'b s n, b s c -> b s n c')  # (b, s, n, C)

        # Step 4: H_res * x_l - Mix the residual stream
        # Paper: H_res is a learnable mapping that mixes features within the residual stream
        x_res_mixed = einsum(H_res, x_stream, 'b s n1 n2, b s n2 c -> b s n1 c')  # (b, s, n, C)

        # Step 5: x_{l+1} = H_res * x_l + (H_post)^T * F(...)
        x_stream_new = x_res_mixed + sublayer_stream  # (b, s, n, C)

        return x_stream_new

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mHC Transformer block.

        Args:
            x: input, shape (batch_size, seq_len, C) - layer input dimension
            token_positions: position indices for RoPE

        Returns:
            output, shape (batch_size, seq_len, C)
        """
        batch_size, seq_len, C = x.shape
        assert C == self.d_model, f"Input dimension {C} should match d_model {self.d_model}"

        # ==== Step 1: Expand input to n*C dimensional residual stream ====
        # Paper Section 3: "x_l = (x_{l,0}^T, ..., x_{l,n-1}^T)^T ∈ R^{n×C}"
        # Initial expansion: replicate input across n streams
        x_stream = repeat(x, 'b s c -> b s n c', n=self.n)  # (b, s, n, C)

        # ==== Step 2: Attention sub-layer with mHC ====
        # Pre-norm: Apply RMSNorm before attention
        # We need to apply norm to each stream independently
        x_stream_normed = torch.stack([
            self.ln1(x_stream[:, :, i, :]) for i in range(self.n)
        ], dim=2)  # (b, s, n, C)

        # Compute mHC mappings for attention
        H_pre_attn, H_post_attn, H_res_attn = self._compute_mhc_mappings(x_stream_normed, self.mhc_attn)

        # Apply attention with mHC
        # Note: attention applies its own RoPE internally
        def attn_fn(x_in):
            return self.self_attention(x_in, token_positions)

        x_stream = self._apply_mhc_sublayer(x_stream, H_pre_attn, H_post_attn, H_res_attn, attn_fn)

        # ==== Step 3: FFN sub-layer with mHC ====
        # Pre-norm: Apply RMSNorm before FFN
        x_stream_normed = torch.stack([
            self.ln2(x_stream[:, :, i, :]) for i in range(self.n)
        ], dim=2)  # (b, s, n, C)

        # Compute mHC mappings for FFN (independent from attention)
        H_pre_ffn, H_post_ffn, H_res_ffn = self._compute_mhc_mappings(x_stream_normed, self.mhc_ffn)

        # Apply FFN with mHC
        x_stream = self._apply_mhc_sublayer(x_stream, H_pre_ffn, H_post_ffn, H_res_ffn, self.ffn)

        # ==== Step 4: Aggregate n*C stream back to C dimension for output ====
        # Mean aggregation across streams for final output
        # This maintains compatibility with standard Transformer interface
        x_output = x_stream.mean(dim=2)  # (b, s, C)

        # Apply dropout
        x_output = self.dropout(x_output)

        return x_output


class mHCTransformerLM(nn.Module):
    """
    Complete mHC Transformer Language Model

    Paper Section 5: "mHC exhibits exceptional stability and scalability
    while maintaining the performance advantages of HC."

    Architecture:
    - Token embedding: vocab_size -> d_model
    - N mHC Transformer blocks (each with Attention + FFN)
    - Final RMSNorm
    - Output projection: d_model -> vocab_size
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        context_length: int,
        expansion_rate: int = 2,  # n from paper, using 2 for small models (paper uses 4)
        theta: float = 10000.0,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.context_length = context_length
        self.expansion_rate = expansion_rate
        self.theta = theta

        # Token embedding: outputs d_model dimensions
        self.token_embedding = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype
        )

        # Stack of mHC Transformer Blocks
        # Each block contains Attention and FFN sub-layers with independent mHC
        self.blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                expansion_rate=expansion_rate,
                theta=theta,
                dropout=dropout,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # Output projection (language model head)
        self.output_projection = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for mHC Transformer Language Model.

        Args:
            input_ids: token ids, shape (batch_size, seq_len)
            token_positions: position indices for RoPE, shape (batch_size, seq_len)

        Returns:
            logits: shape (batch_size, seq_len, vocab_size)
        """
        # Token embedding: (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embedding(input_ids)

        # If positions not provided, use default sequential positions
        if token_positions is None:
            seq_len = input_ids.shape[1]
            token_positions = torch.arange(seq_len, device=input_ids.device)
            token_positions = token_positions.unsqueeze(0).expand(input_ids.shape[0], -1)

        # Pass through mHC blocks
        for block in self.blocks:
            x = block(x, token_positions)

        # Final layer norm
        x = self.ln_final(x)

        # Output projection to vocabulary
        logits = self.output_projection(x)

        return logits

    def get_num_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())


# ========== For comparison, also keep original Transformer import ==========
# Note: Your original TransformerBlock and TransformerLM are in another file
# Commented out here for completeness, you can import as needed
# from cs336_basics.model import TransformerBlock as OriginalTransformerBlock
# from cs336_basics.model import TransformerLM as OriginalTransformerLM