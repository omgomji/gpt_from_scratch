"""
lr_scheduler.py
---------------
Cosine annealing with linear warmup.

Satisfies: V2 Upgrade (Training Improvements)
"""

import math

def get_lr(step: int, max_iters: int, max_lr: float, min_lr: float, warmup_iters: int) -> float:
    """Compute the learning rate at a given step.
    
    1. Linear warmup for warmup_iters steps.
    2. Cosine decay down to min_lr over remaining steps.
    3. If step > max_iters, return min_lr.
    
    Parameters
    ----------
    step : int
        Current training step.
    max_iters : int
        Total number of training steps.
    max_lr : float
        Maximum learning rate.
    min_lr : float
        Minimum learning rate.
    warmup_iters : int
        Number of steps to linearly warmup to max_lr.
        
    Returns
    -------
    float
        The computed learning rate.
    """
    # 1. Linear warmup
    if step < warmup_iters:
        return max_lr * (step + 1) / warmup_iters
        
    # 3. Past max_iters
    if step > max_iters:
        return min_lr
        
    # 2. Cosine decay
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)
