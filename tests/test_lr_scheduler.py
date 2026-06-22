"""
tests/test_lr_scheduler.py
--------------------------
Test suite for the learning rate scheduler.
"""

import math
from src.lr_scheduler import get_lr

def test_lr_warmup():
    max_iters = 1000
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 100
    
    # At step 0, lr should be exactly max_lr * (1 / warmup_iters)
    assert get_lr(0, max_iters, max_lr, min_lr, warmup_iters) == max_lr / warmup_iters
    
    # At step warmup_iters - 1, lr should be max_lr
    assert get_lr(warmup_iters - 1, max_iters, max_lr, min_lr, warmup_iters) == max_lr

def test_lr_cosine_decay():
    max_iters = 1000
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 100
    
    # At max_iters, it should hit min_lr
    assert math.isclose(get_lr(max_iters, max_iters, max_lr, min_lr, warmup_iters), min_lr)
    
    # In between, it should be between min_lr and max_lr
    mid_step = warmup_iters + (max_iters - warmup_iters) // 2
    mid_lr = get_lr(mid_step, max_iters, max_lr, min_lr, warmup_iters)
    assert min_lr < mid_lr < max_lr

def test_lr_past_max_iters():
    max_iters = 1000
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 100
    
    assert get_lr(1050, max_iters, max_lr, min_lr, warmup_iters) == min_lr
