# mHC 代码改进总结  
根据 arXiv 论文 2512.24880 "mHC: Manifold-Constrained Hyper-Connections"

## 论文关键内容
- **标题**: mHC: Manifold-Constrained Hyper-Connections
- **问题**: HC（Hyper-Connections）存在数值不稳定性和系统开销问题
- **解决方案**: 使用Sinkhorn-Knopp算法将残差映射投影到Birkhoff polytope（doubly stochastic matrices流形）

## 代码改进清单

### 1. **改进Sinkhorn-Knopp算法** ✓
**文件**: `cs336_basics/mhc_model.py`

**改动**:
- 更新epsilon参数为1e-6（论文推荐）
- 改进数值稳定性，clamping范围扩大到[-20, 20]
- 添加详细的文档注释
- 明确说明论文Eq.(9)的实现

**关键代码**:
```python
def sinkhorn_knopp(M: torch.Tensor, num_iter: int = 20, eps: float = 1e-6) -> torch.Tensor:
    # Paper Eq. (9): M^(0) = exp(tilde{H}^res)
    M_clamped = torch.clamp(M, min=-20, max=20)
    M_pos = torch.exp(M_clamped)
    
    # Alternating row and column normalization
    for _ in range(num_iter):
        row_sum = M_pos.sum(dim=-1, keepdim=True)
        M_pos = M_pos / (row_sum + eps)
        col_sum = M_pos.sum(dim=-2, keepdim=True)
        M_pos = M_pos / (col_sum + eps)
    
    return M_pos
```

### 2. **改进ManifoldHyperConnection forward方法** ✓
**文件**: `cs336_basics/mhc_model.py`

**改动**:
- 完整实现论文Eq.(7-8)的约束投影
- 确保H_pre使用sigmoid约束（范围[0,1]）
- 确保H_post使用2*sigmoid约束（范围[0,2]）
- 对H_res使用Sinkhorn-Knopp投影到doubly stochastic矩阵
- 添加清晰的注释指向论文公式

**关键改动**:
- 论文Eq.(7): 计算raw mappings $\tilde{H}$
- 论文Eq.(8): 应用约束投影转换为最终的映射$H$
```python
# H_pre = sigmoid(H_pre_tilde) - range [0,1]
H_pre = torch.sigmoid(H_pre_tilde)

# H_post = 2 * sigmoid(H_post_tilde) - range [0,2]
H_post = 2 * torch.sigmoid(H_post_tilde)

# H_res = Sinkhorn-Knopp(H_res_tilde) - doubly stochastic
H_res = sinkhorn_knopp(H_res_tilde, num_iter=20)
```

### 3. **改进mHCTransformerBlock forward方法** ✓
**文件**: `cs336_basics/mhc_model.py`

**改动**:
- 完整实现论文公式(3)的流架构
- 正确处理n*C维的残差流
- 按照论文设计应用H_pre、H_post、H_res的操作
- 简化并标准化流的处理

**论文Eq.(3)**: $x_{l+1} = H_{res} x_l + (H_{post})^T F(H_{pre} x_l, W_l)$

**关键改动**:
```python
# 1. 聚合流到C维输入: x_attn_in = H_pre * x_l
x_attn_in = einsum(H_pre, x_stream, 'b s n, b s n c -> b s c')

# 2. 通过Attention层
attn_out = self.self_attention(x_attn_in, token_positions)

# 3. 投影回n*C维流: attn_stream = (H_post)^T * attn_out
attn_stream = einsum(H_post, attn_out, 'b s n, b s c -> b s n c')

# 4. 残差连接和H_res混合: x'_l = H_res * x_l + (H_post)^T * attn_out
x_res_mixed = einsum(H_res, x_stream, 'b s n1 n2, b s n2 c -> b s n1 c')
x_stream_after_attn = x_res_mixed + attn_stream

# 5. FFN类似处理
```

## 论文核心理论总结

### 问题分析
- **HC的不稳定性**: 无约束的$H_{res}$导致信号爆炸或消失
- **内存开销**: n倍的流维度导致内存访问成本增加n倍

### mHC的解决方案

#### 1. **Manifold Projection** (论文Section 4.1)
- 约束$H_{res} \in \mathcal{M}_{res}$（Birkhoff polytope）
- 使用doubly stochastic矩阵（行列和都为1）
- 保证谱范数$\|\mathcal{H}_{res}\|_2 \leq 1$
- 乘法封闭性确保组合映射也是doubly stochastic

#### 2. **参数化与投影** (论文Section 4.2)  
- 动态映射：依赖于输入$x_l$的部分
- 静态映射：可学习的偏置项
- 非负约束：$H_{pre}, H_{post} \geq 0$防止信号抵消

#### 3. **系统级优化** (论文Section 4.3)
- Kernel Fusion：融合多个操作以减少内存访问
- Selective Recomputation：梯度检查点优化内存
- DualPipe通信重叠：管道并行优化

## 实现特点

### 数值稳定性
- Sinkhorn-Knopp迭代次数：20（论文推荐）
- 防止数值溢出的clamp操作
- 适当的epsilon参数（1e-6）

### 约束实现
- 非负约束通过sigmoid激活函数
- Doubly stochastic通过Sinkhorn-Knopp投影
- 保证所有约束的数学正确性

## 论文性能指标
- 相比HC：在27B模型上loss减少0.021
- 相比baseline：多个下游任务获得2-3%的性能提升
- 系统开销：n=4时仅增加6.7%训练时间
- 稳定性：可将复合映射的Amax Gain从~3000降至~1.6

## 测试文件
- `test_mhc_improvements.py`: 完整的单元测试套件
  - 验证Sinkhorn-Knopp算法的doubly stochastic性质
  - 验证约束的正确应用
  - 端到端的模型测试

## 参考
论文标题: mHC: Manifold-Constrained Hyper-Connections
arXiv: 2512.24880
