import torch
import torch.nn as nn
import math
from einops import einsum,reduce, rearrange
import torch.nn.functional as F
from typing import Optional

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
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        self.weight=nn.Parameter(
            torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1,a=-3,b=3)
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        x_norm=x/rms
        output=x_norm*self.weight
        return output.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff=None,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        if d_ff is None:
            theoretical_ff=int(8/3*d_model)
            self.d_ff=((theoretical_ff+63)//64)*64
        else:
            self.d_ff=d_ff
        
        print(f"d_model={d_model},d_ff={self.d_ff}(理论值={int(8/3*d_model)})")

        self.W1=Linear(d_model,self.d_ff,device=device,dtype=dtype)
        self.W3=Linear(d_model,self.d_ff,device=device,dtype=dtype)
        self.W2=Linear(self.d_ff,d_model,device=device,dtype=dtype)
        
    def forward(self,x):
        input_shape=x.shape
        act = self.W1(x)        # 激活路径
        act = F.silu(act)        # 激活
    
        gate = self.W3(x)        # 门控路径（不激活）
    
        gated = act * gate       # GLU 相乘
    
        output = self.W2(gated)  # 输出投影
    
        assert output.shape == input_shape, \
        f"输出形状{output.shape}应与输入形状{input_shape}相同"
        return output
    
    def extra_repr(self):
        return f"d_model={self.d_model},d_ff={self.d_ff}"

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        assert d_k%2==0,"d_k必须是偶数才能旋转"
        self._build_cache(device)
    def _build_cache(self,device):
        positions=torch.arange(self.max_seq_len,device=device).float()
        dim_pairs=torch.arange(0,self.d_k,2,device=device).float()
        freqs=1.0/self.theta**(dim_pairs/self.d_k)
        angles=positions[:,None]*freqs[None,:]
        cos_cache=torch.cos(angles).repeat_interleave(2,dim=-1)
        sin_cache=torch.sin(angles).repeat_interleave(2,dim=-1)

        self.register_buffer('cos_cache', cos_cache, persistent=False)
        self.register_buffer('sin_cache', sin_cache, persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        original_shape=x.shape
        seq_len=x.shape[-2]
        cos=self.cos_cache[token_positions]
        sin=self.sin_cache[token_positions]
        x_even=x[...,0::2]
        x_odd=x[...,1::2]
        
        cos_pair=cos[...,0::2]
        sin_pair=sin[...,0::2]

        x_rotated_even = x_even * cos_pair - x_odd * sin_pair
        x_rotated_odd = x_even * sin_pair + x_odd * cos_pair

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd

        return x_rotated
    
    def rotate_pair(self,x:torch.Tensor,pos:torch.Tensor)->torch.Tensor:
        cos = self.cos_cache[pos]
        sin = self.sin_cache[pos]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
        x_rot_odd = x_even * sin[..., 0::2] + x_odd * cos[..., 0::2]
        
        result = torch.zeros_like(x)
        result[..., 0::2] = x_rot_even
        result[..., 1::2] = x_rot_odd
        
        return result

def softmax(x:torch.Tensor,dim:int=-1)->torch.Tensor:
    x_max=torch.max(x,dim=dim,keepdim=True)[0]
    x_stable=x-x_max
    x_exp=torch.exp(x_stable)
    x_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    x_softmax=x_exp/x_sum
    return x_softmax

def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:Optional[torch.Tensor]=None
)->torch.Tensor:
    d_k=Q.shape[-1]
    scores=torch.matmul(Q,K.transpose(-2,-1))
    scores=scores/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(~mask,float('-inf'))
    attention_weights=softmax(scores,dim=-1)
    output=torch.matmul(attention_weights,V)
    return output

class CasualMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, RoPE: RotaryPositionalEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.hd_k = self.hd_v = d_model
        self.d_k = self.hd_k // num_heads
        self.d_v = self.hd_v // num_heads
        if RoPE != None:
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
        # pos = torch.arange(0, seq_len)
        # token_positions = repeat(token_positions, "... l -> ... h l", h=self.num_heads)
        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = rearrange(attention, "... h l d_v -> ... l (h d_v)")
        return self.linearO(attention)

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,theta:float=10000.0,dropout:float=0.1,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm1=RMSNorm(d_model,device=device,dtype=dtype)
        self.self_attention=CasualMultiheadSelfAttention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,theta=theta,device=device,dtype=dtype)
        

        self.norm2=RMSNorm(d_model,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
        

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        norm_x=self.norm1(x)
        attn_output=self.self_attention(norm_x,token_positions)
        x=x+attn_output

        norm_x=self.norm2(x)
        ffn_output=self.ffn(norm_x)
        x=x+ffn_output

        return x

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size:int,
            d_model:int,
            num_heads:int,
            d_ff:int,
            num_layers:int,
            context_length:int,
            theta:float=10000.0,
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
        self.theta = theta
        self.token_embedding=Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype
        )
        self.blocks=nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,  
                theta=theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final=RMSNorm(d_model,device=device,dtype=dtype)
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
        x=self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits


