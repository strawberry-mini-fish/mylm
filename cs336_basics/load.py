import numpy as np
import torch
import os 
import typing

def get_batch(x,batch_size,context_length,device):
    assert x.ndim==1,"x应该是一维数组"
    max_start_isx=len(x)-context_length
    start_indices=torch.randint(0,max_start_isx,size=(batch_size,))
    inputs_list=[]
    targets_list=[]
    for i in range(batch_size):
        idx = start_indices[i].item()
        input_seq = torch.tensor(x[idx:idx + context_length], dtype=torch.long)
        target_seq = torch.tensor(x[idx + 1:idx + context_length + 1], dtype=torch.long)
        inputs_list.append(input_seq)
        targets_list.append(target_seq)
    inputs = torch.stack(inputs_list)
    targets = torch.stack(targets_list)
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets

def save_checkpoint(model,optimizer,iteration,out):
    model_state=model.state_dict()
    optimizer_state=optimizer.state_dict()
    checkpoint = {
        'model_state_dict': model_state,      # 模型权重
        'optimizer_state_dict': optimizer_state,  # 优化器状态
        'iteration': iteration                 # 当前迭代次数
    }
    torch.save(checkpoint, out)

def load_checkpoint(src,model,optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
