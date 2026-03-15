import torch
import torch.nn.functional as F
from typing import Any
import numpy as np
import math
from typing import Callable,Optional,List


def cross_entropy(logits,targets):
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]  
    logits_stable = logits - logits_max
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    targets = targets.long()
    target_logits = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = -(target_logits - log_sum_exp)
    return losses.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr:float=1e-3,
            betas:tuple=(0.9,0.999),
            eps:float=1e-8,
            weight_decay:float=0.01,
    ):
        if lr<0.0:
            raise ValueError(f"Invalid learning rate:{lr}")
        
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params,defaults)

    def step(self,closure:Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad.data
                state=self.state[p]
                if len(state)==0:
                     state["step"] = 0
                     state["exp_avg"] = torch.zeros_like(p.data)    # m
                     state["exp_avg_sq"] = torch.zeros_like(p.data) # v
                exp_avg = state["exp_avg"]      # m
                exp_avg_sq = state["exp_avg_sq"] # v
                state["step"] += 1   
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss
        
def cosine_lr_schedule(
    t: int,           
    alpha_max: float, 
    alpha_min: float, 
    T_w: int,         
    T_c: int   
    )->float :
        if t < T_w:
            alpha_t = (t / T_w) * alpha_max
        elif t <= T_c:
            cos_input = math.pi * (t - T_w) / (T_c - T_w)
        
            cos_val = math.cos(cos_input)
       
            decay_factor = 0.5 * (1 + cos_val)
        
            alpha_t = alpha_min + decay_factor * (alpha_max - alpha_min)
        else:
            alpha_t = alpha_min
    
        return alpha_t    

def gradient_clipping(
        params:List[torch.nn.Parameter],
        max_norm_value:float,
        eps:float=1e-6
):
    gradients = []
    for p in params:
        if p.grad is not None:
            gradients.append(p.grad)

    if len(gradients) == 0:
        return
    
    total_norm_sq=0.0
    for grad in gradients:
        total_norm_sq+=grad.norm(2).item()**2
    total_norm=math.sqrt(total_norm_sq)
    clip_coef=max_norm_value/(total_norm+eps)
    if clip_coef<1.0:
        for grad in gradients:
            grad.mul_(clip_coef)
