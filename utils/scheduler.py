#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/1 15:12
# @File    : scheduler.py
# @Description :
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineLR(LambdaLR):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0):
        """
        预热余弦退火调度器

        参数:
        optimizer (torch.optim.Optimizer): 优化器
        T_max (int): 总训练步数
        eta_min (float, optional): 余弦退火的最小学习率. 默认为0
        warmup_steps (int, optional): 预热步数. 默认为0
        last_epoch (int, optional): 上一个epoch. 默认为-1
        """
        self.T_max = T_max  # 10 * 9999
        self.eta_min = eta_min  # 0.01
        self.warmup_steps = warmup_steps  # 1 * 9999
        self.initial_lr = optimizer.param_groups[0]['lr']
        if T_max <= warmup_steps:
            raise ValueError("T_max must be greater than warmup_steps")

        def lr_lambda(step):
            if step < self.warmup_steps:
                #
                return (1 - self.eta_min) * (step / self.warmup_steps) ** 0.5 + self.eta_min
            else:
                return self.eta_min / self.initial_lr + (1 - self.eta_min / self.initial_lr) * 0.5 * (
                            1 + math.cos(math.pi * (step - self.warmup_steps) / (self.T_max - self.warmup_steps)))

        super(WarmupCosineLR, self).__init__(optimizer, lr_lambda)
