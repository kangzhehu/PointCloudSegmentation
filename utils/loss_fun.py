#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/4 10:24
# @File    : loss_fun.py
# @Description : 损失函数优化
# CrossEntropyLoss + lovasz_softmax_flat加权组合
# 要找到最合适的权重
import torch
import torch.nn as nn
from .lovasz_losses import lovasz_softmax_flat


class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_lovasz=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_lovasz = weight_lovasz
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # logits: [B*N, C]
        # labels: [B*N]

        # Calculate CrossEntropyLoss
        ce_loss = self.ce_loss(logits, labels)

        # Calculate Lovasz-Softmax Loss
        probas = torch.nn.functional.softmax(logits, dim=1)
        lovasz_loss = lovasz_softmax_flat(probas, labels)

        # Weighted combination of losses
        total_loss = self.weight_ce * ce_loss + self.weight_lovasz * lovasz_loss
        return total_loss


