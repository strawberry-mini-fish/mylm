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

# ========== 导入原始模型的基础组件 ==========
from cs336_basics.model import (
    Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding,
    softmax, scaled_dot_product_attention, CasualMultiheadSelfAttention,
    TransformerBlock, TransformerLM
)

# ========== 导入 mHC 模型 ==========
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
    parser = argparse.ArgumentParser(description="训练语言模型（支持原始Transformer和mHC）")
    
    # ========== 新增模型选择参数 ==========
    parser.add_argument("--model_type", type=str, default="original", choices=["original", "mhc"],
                        help="模型类型：original（原始Transformer）或 mhc（流形约束超连接）")
    parser.add_argument("--expansion_rate", type=int, default=4,
                        help="mHC 残差流扩展率 n (默认4，仅当 model_type=mhc 时有效)")
    
    parser.add_argument("--train_data", type=str, required=False, default="roneneldan/TinyStories",
                        help="Hugging Face数据集名称")
    parser.add_argument("--val_data", type=str, default=None,
                        help="验证数据文件路径（可选，如果使用HF数据集则自动处理）")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="词汇表大小")
    
    parser.add_argument("--n_layer", type=int, default=12,
                        help="Transformer层数")
    parser.add_argument("--n_head", type=int, default=12,
                        help="注意力头数量")
    parser.add_argument("--n_embd", type=int, default=768,
                        help="嵌入维度")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="最大上下文长度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout概率")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--context_length", type=int, default=128,
                        help="上下文长度")
    parser.add_argument("--max_iters", type=int, default=100000,
                        help="最大训练迭代次数")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="评估间隔")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="检查点保存间隔")
    parser.add_argument("--eval_iters", type=int, default=200,
                        help="评估时使用的迭代次数")
    
    parser.add_argument("--learning_rate", type=float, default=6e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="权重衰减")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="学习率预热迭代次数")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="检查点保存目录")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从哪个检查点恢复训练")
    
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    
    parser.add_argument("--tokenizer_name", type=str, default="gpt2",
                        help="使用的tokenizer名称（来自Hugging Face）")
    
    return parser.parse_args()

def load_and_prepare_hf_dataset(dataset_name, tokenizer, context_length, split="train"):
    logger.info(f"正在加载Hugging Face数据集：{dataset_name} - {split} split")
    
    # 加载数据集
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"数据集大小：{len(dataset)} 个样本")
    
    # 定义tokenization函数
    def tokenize_function(examples):
        # Tokenize文本
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            stride=context_length // 2,  # 添加重叠以增加数据
        )
        
        # 处理overflowing tokens生成多个序列
        input_ids_list = []
        for ids in tokenized["input_ids"]:
            if len(ids) == context_length:  # 只保留完整长度的序列
                input_ids_list.append(ids)
        
        return {"input_ids": input_ids_list}
    
    # 对数据集进行tokenization
    logger.info("正在进行tokenization...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # 将数据转换为numpy数组以便mmap使用
    logger.info("正在准备内存映射数据...")
    all_tokens = []
    for item in tokenized_dataset:
        all_tokens.extend(item["input_ids"])
    
    tokens_array = np.array(all_tokens, dtype=np.uint16).flatten()
    
    # 保存为临时文件以便使用mmap
    data_path = f"/tmp/tinystories_{split}_tokens.npy"
    np.save(data_path, tokens_array)
    logger.info(f"数据已保存到：{data_path}，总token数：{len(tokens_array)}")
    
    return data_path

def load_data_mmap(data_path):
    logger.info(f"正在加载数据：{data_path}")
    if data_path.endswith('.npy'):
        data = np.load(data_path, mmap_mode='r')
    else:
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    logger.info(f"数据加载完成，形状：{data.shape}")
    return data

def create_model(args, device):
    """根据参数创建模型"""
    
    # 计算 FFN 维度（两个模型共用）
    d_ff = int(4 * args.n_embd)
    d_ff = ((d_ff + 63) // 64) * 64  # 对齐到 64 的倍数
    
    if args.model_type == "mhc":
        logger.info("=" * 50)
        logger.info("使用 mHC 架构 (Manifold-Constrained Hyper-Connections)")
        logger.info(f"扩展率 n = {args.expansion_rate}")
        
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
            logger.info("提示：mHC 架构更稳定，可以尝试增大 --grad_clip 到 2.0 或 5.0")
        
    else:  # original
        logger.info("=" * 50)
        logger.info("使用原始 Transformer 架构")
        
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
    """评估损失，根据模型类型决定是否传入 token_positions"""
    model.eval()
    out = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        if data is None:
            continue
        
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, context_length, device)
            
            if model_type == "mhc":
                # mHC 模型需要 token_positions
                token_positions = torch.arange(context_length, device=device)
                token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
                logits = model(X, token_positions)
            else:
                # 原始模型不需要 token_positions
                logits = model(X)
                
            loss = cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录：{args.output_dir}")
    
    device = torch.device(args.device)
    logger.info(f"使用设备：{device}")
    
    logger.info("=" * 50)
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer: {args.tokenizer_name}")
    logger.info(f"词汇表大小: {tokenizer.vocab_size}")
    
    logger.info("=" * 50)
    logger.info("加载Hugging Face数据集...")
    # 加载训练数据
    train_data_path = load_and_prepare_hf_dataset(
        args.train_data, 
        tokenizer, 
        args.context_length, 
        split="train"
    )
    train_data = load_data_mmap(train_data_path)
    
    # 加载验证数据
    val_data = None
    if args.val_data is None:
        # 使用默认的验证集
        try:
            val_data_path = load_and_prepare_hf_dataset(
                args.train_data,
                tokenizer,
                args.context_length,
                split="validation"
            )
            val_data = load_data_mmap(val_data_path)
        except Exception as e:
            logger.warning(f"无法加载验证集: {e}，将只使用训练集评估")
            val_data = None
    else:
        val_data = load_data_mmap(args.val_data)
    
    logger.info("=" * 50)
    logger.info("初始化模型...")
    model = create_model(args, device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量：{n_params / 1e6:.2f}M")
    
    model.to(device)
    
    logger.info("初始化优化器...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    
    start_iter = 0
    if args.resume_from:
        logger.info(f"从检查点恢复训练：{args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"从迭代 {start_iter} 恢复训练")
    
    logger.info("=" * 50)
    logger.info("开始训练...")
    logger.info(f"最大迭代次数：{args.max_iters}")
    logger.info(f"模型类型：{args.model_type}")
    logger.info("=" * 50)
    
    model.train()
    t0 = time.time()
    train_losses = []
    
    for iter in range(start_iter, args.max_iters):
        # 计算学习率
        lr = cosine_lr_schedule(iter, args.learning_rate, args.warmup_iters, args.max_iters, args.learning_rate * 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 获取batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        # 根据模型类型前向传播
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
                f"迭代 {iter:6d} | "
                f"损失 {avg_loss:.4f} | "
                f"学习率 {lr:.2e} | "
                f"时间 {dt*1000:.2f}ms/iter"
            )
        
        if iter % args.eval_interval == 0 and val_data is not None and iter > 0:
            logger.info("执行评估...")
            losses = estimate_loss(
                model, train_data, val_data, 
                args.batch_size, args.context_length, 
                args.eval_iters, device,
                args.model_type  # 传入模型类型
            )
            logger.info(
                f"评估 - 迭代 {iter:6d} | "
                f"训练损失 {losses['train']:.4f} | "
                f"验证损失 {losses['val']:.4f}"
            )
        
        if iter % args.save_interval == 0 and iter > 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{iter}.pt")
            logger.info(f"保存检查点：{checkpoint_path}")
            save_checkpoint(model, optimizer, iter, checkpoint_path)
    
    logger.info("=" * 50)
    logger.info("训练完成！")
    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pt")
    logger.info(f"保存最终模型：{final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_iters - 1, final_checkpoint_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"训练失败：{e}")
        sys.exit(0)