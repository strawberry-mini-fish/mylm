import torch
import torch.nn as nn
import math
from einops import einsum, reduce, rearrange, repeat
import torch.nn.functional as F
from typing import Optional

# ========== 基础组件（复用您的实现）==========
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
        # x 可以是 (..., d_model) 或 (..., n*d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # 计算最后一维的 RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        
        # 确保 weight 的维度匹配
        if x_norm.shape[-1] == self.d_model:
            output = x_norm * self.weight
        else:
            # 对于展平后的输入，需要重复 weight
            n = x_norm.shape[-1] // self.d_model
            weight_expanded = self.weight.unsqueeze(0).repeat(1, n)  # (1, n*d_model)
            output = x_norm * weight_expanded
            
        return output.to(in_dtype)
    
    @staticmethod
    def apply(x: torch.Tensor, dim: int):
        """静态方法，用于 mHC 中的 RMSNorm"""
        eps = 1e-5
        x_float = x.to(torch.float32)
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x_float / rms).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            theoretical_ff = int(8/3 * d_model)
            self.d_ff = ((theoretical_ff + 63) // 64) * 64
        else:
            self.d_ff = d_ff
        
        print(f"d_model={d_model}, d_ff={self.d_ff} (理论值={int(8/3*d_model)})")

        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        
    def forward(self, x):
        input_shape = x.shape
        act = self.W1(x)        # 激活路径
        act = F.silu(act)        # 激活
    
        gate = self.W3(x)        # 门控路径（不激活）
    
        gated = act * gate       # GLU 相乘
    
        output = self.W2(gated)  # 输出投影
    
        assert output.shape == input_shape, \
            f"输出形状{output.shape}应与输入形状{input_shape}相同"
        return output
    
    def extra_repr(self):
        return f"d_model={self.d_model}, d_ff={self.d_ff}"

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        assert d_k % 2 == 0, "d_k必须是偶数才能旋转"
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


# ========== mHC 核心组件 ==========
def sinkhorn_knopp(M: torch.Tensor, num_iter: int = 20) -> torch.Tensor:
    """
    将矩阵投影到双随机矩阵流形上
    M: 输入矩阵，形状为 (..., n, n)
    num_iter: 迭代次数
    返回：双随机矩阵
    """
    # 确保所有元素为正
    M_pos = torch.exp(M)  # 论文中先取指数
    # 交替行归一化和列归一化
    for _ in range(num_iter):
        # 行归一化
        M_pos = M_pos / (M_pos.sum(dim=-1, keepdim=True) + 1e-8)
        # 列归一化
        M_pos = M_pos / (M_pos.sum(dim=-2, keepdim=True) + 1e-8)
    return M_pos

class ManifoldHyperConnection(nn.Module):
    """
    mHC: Manifold-Constrained Hyper-Connections
    将残差流从 C 维扩展到 n*C 维，并施加流形约束
    """
    def __init__(
        self,
        d_model: int,           # 原始维度 C
        expansion_rate: int,     # 扩展率 n (论文中 n=4)
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n = expansion_rate
        self.stream_dim = self.n * d_model  # 残差流维度 n*C
        
        # 论文公式(7)中的线性投影
        # φ_pre, φ_post: R^{nC × n}
        self.phi_pre = nn.Parameter(
            torch.empty((self.stream_dim, self.n), device=device, dtype=dtype)
        )
        self.phi_post = nn.Parameter(
            torch.empty((self.stream_dim, self.n), device=device, dtype=dtype)
        )
        # φ_res: R^{nC × n^2}
        self.phi_res = nn.Parameter(
            torch.empty((self.stream_dim, self.n * self.n), device=device, dtype=dtype)
        )
        
        # 可学习的门控因子 α (标量)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))
        self.alpha_post = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))
        self.alpha_res = nn.Parameter(torch.tensor(0.01, device=device, dtype=dtype))
        
        # 可学习的偏置 b
        self.b_pre = nn.Parameter(torch.zeros((1, self.n), device=device, dtype=dtype))
        self.b_post = nn.Parameter(torch.zeros((1, self.n), device=device, dtype=dtype))
        self.b_res = nn.Parameter(torch.zeros((self.n, self.n), device=device, dtype=dtype))
        
        # 初始化 phi 矩阵
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2 / (self.stream_dim + self.n))
        nn.init.trunc_normal_(self.phi_pre, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.phi_post, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.phi_res, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: 输入，形状 (..., n*C) - 展平后的残差流
        返回: (H_pre, H_post, H_res) - 三个约束后的映射
        """
        # x 已经是展平后的向量，形状 (..., n*C)
        
        # 论文公式(7)中的 RMSNorm
        # 注意：RMSNorm 作用在最后一维
        x_norm = RMSNorm.apply(x, self.stream_dim)
        
        # 论文公式(7): 计算原始映射
        # H_pre_tilde = α_pre * (x_norm @ φ_pre) + b_pre
        H_pre_tilde = self.alpha_pre * (x_norm @ self.phi_pre) + self.b_pre  # (..., n)
        
        # H_post_tilde = α_post * (x_norm @ φ_post) + b_post
        H_post_tilde = self.alpha_post * (x_norm @ self.phi_post) + self.b_post  # (..., n)
        
        # H_res_tilde = α_res * (x_norm @ φ_res) 重塑 + b_res
        H_res_tilde_flat = self.alpha_res * (x_norm @ self.phi_res)  # (..., n^2)
        H_res_tilde = rearrange(H_res_tilde_flat, '... (n n2) -> ... n n2', n=self.n, n2=self.n)  # (..., n, n)
        H_res_tilde = H_res_tilde + self.b_res
        
        # 论文公式(8): 应用约束
        # H_pre = sigmoid(H_pre_tilde)
        H_pre = torch.sigmoid(H_pre_tilde)  # 非负约束
        
        # H_post = 2 * sigmoid(H_post_tilde)
        H_post = 2 * torch.sigmoid(H_post_tilde)  # 非负约束，且范围 [0,2]
        
        # H_res = Sinkhorn-Knopp(H_res_tilde)
        H_res = sinkhorn_knopp(H_res_tilde)  # 双随机矩阵约束
        
        return H_pre, H_post, H_res


class mHCTransformerBlock(nn.Module):
    """
    基于 mHC 的 Transformer Block
    将残差流从 C 维扩展到 n*C 维
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        expansion_rate: int = 4,  # 论文中的 n
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
        
        # mHC 的核心模块
        self.mhc = ManifoldHyperConnection(
            d_model=d_model,
            expansion_rate=expansion_rate,
            device=device,
            dtype=dtype
        )
        
        # 注意：注意力层和FFN的输入维度还是 d_model，不是 stream_dim
        self.self_attention = CasualMultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        
        # 可选的 dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: 输入，形状 (batch_size, seq_len, C) - 原始维度
        返回: 输出，形状 (batch_size, seq_len, C)
        """
        batch_size, seq_len, C = x.shape
        assert C == self.d_model, f"输入维度 {C} 应与 d_model {self.d_model} 匹配"
        
        # ==== 1. 将输入扩展到 n*C 维的残差流 ====
        # 通过重复扩展：x_stream (batch, seq_len, n*C)
        x_stream = repeat(x, 'b s c -> b s (n c)', n=self.n)
        
        # ==== 2. 展平以计算映射系数 ====
        # 为了计算 H_pre, H_post, H_res，需要将 batch 和 seq_len 合并
        x_flat = rearrange(x_stream, 'b s d -> (b s) d')  # (b*s, n*C)
        
        # 计算三个映射
        H_pre, H_post, H_res = self.mhc(x_flat)  # 每个形状: (b*s, n) 或 (b*s, n, n)
        
        # 恢复 batch 和 seq_len 维度
        H_pre = rearrange(H_pre, '(b s) n -> b s n', b=batch_size, s=seq_len)
        H_post = rearrange(H_post, '(b s) n -> b s n', b=batch_size, s=seq_len)
        H_res = rearrange(H_res, '(b s) n1 n2 -> b s n1 n2', 
                         b=batch_size, s=seq_len, n1=self.n, n2=self.n)
        
        # ==== 3. 应用 H_pre 聚合到 C 维输入 ====
        # 将 x_stream 分成 n 组，每组 C 维
        x_split = rearrange(x_stream, 'b s (n c) -> b s n c', n=self.n, c=C)
        
        # 使用 H_pre 加权聚合: x_attn_input (b, s, C)
        # H_pre: (b, s, n) 作为权重
        x_attn_input = einsum(H_pre, x_split, 'b s n, b s n c -> b s c')
        
        # ==== 4. 通过注意力层 ====
        attn_output = self.self_attention(x_attn_input, token_positions)
        # attn_output: (b, s, C)
        
        # ==== 5. 应用 H_post 将注意力输出投影回流 ====
        # 首先将 attn_output 扩展为 n 个副本，然后用 H_post 加权
        attn_output_expanded = repeat(attn_output, 'b s c -> b s n c', n=self.n)
        
        # 应用 H_post 作为权重
        # H_post: (b, s, n) -> (b, s, n, 1)
        attn_stream = attn_output_expanded * H_post.unsqueeze(-1)
        
        # ==== 6. 应用 H_res 进行流内混合 ====
        # x_stream_current = H_res @ x_stream_prev + attn_stream
        # 这里 x_stream 是上一层的残差流
        
        # 将 x_stream 重塑为 (b, s, n, C) 以便于矩阵乘法
        x_stream_reshaped = rearrange(x_stream, 'b s (n c) -> b s n c', n=self.n, c=C)
        
        # H_res: (b, s, n, n) 对 n 个流进行混合
        # 对每个位置和批次，做 n×n 矩阵乘 n×C
        x_stream_mixed = einsum(H_res, x_stream_reshaped, 'b s n1 n2, b s n2 c -> b s n1 c')
        
        # 加上投影后的注意力输出
        x_stream_updated = x_stream_mixed + attn_stream
        
        # ==== 7. 通过 FFN（需要先聚合回 C 维） ====
        # 对更新后的流再次聚合，得到 FFN 的输入
        x_ffn_input = einsum(H_pre, x_stream_updated, 'b s n, b s n c -> b s c')
        
        ffn_output = self.ffn(x_ffn_input)
        
        # ==== 8. 再次应用 H_post 投影回流 ====
        ffn_output_expanded = repeat(ffn_output, 'b s c -> b s n c', n=self.n)
        ffn_stream = ffn_output_expanded * H_post.unsqueeze(-1)
        
        # ==== 9. 最终残差流更新 ====
        x_stream_final = x_stream_updated + ffn_stream
        
        # ==== 10. 聚合回 C 维输出 ====
        # 最终输出是 H_pre 加权的流的和
        x_output = einsum(H_pre, x_stream_final, 'b s n, b s n c -> b s c')
        
        # 应用 dropout
        x_output = self.dropout(x_output)
        
        return x_output


class mHCTransformerLM(nn.Module):
    """
    完整的 mHC Transformer 语言模型
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        context_length: int,
        expansion_rate: int = 4,
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
        
        # token embedding 不变，输出 d_model 维
        self.token_embedding = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype
        )
        
        # 使用 mHC 的 Transformer Block
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
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
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
        # token embedding: (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embedding(input_ids)
        
        # 如果未提供位置信息，使用默认位置
        if token_positions is None:
            seq_len = input_ids.shape[1]
            token_positions = torch.arange(seq_len, device=input_ids.device)
            token_positions = token_positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        
        # 通过 mHC 块
        for block in self.blocks:
            x = block(x, token_positions)
            
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits


# ========== 为了方便对比，也保留原始 Transformer 的导入 ==========
# 注意：您原来的 TransformerBlock 和 TransformerLM 在另一个文件中
# 这里为了完整性，注释掉，您可以根据需要导入
# from cs336_basics.model import TransformerBlock as OriginalTransformerBlock
# from cs336_basics.model import TransformerLM as OriginalTransformerLM