import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

from .lovasz_losses import lovasz_softmax_flat


def fast_hist(pred, label, n):
    """
    通常用于计算预测和标签之间的快速直方图
    :param pred: 模型的预测结果，通常是一个向量或张量
    :param label: 实际标签
    :param n: 分类任务中类别的数量
    :return:完整的混淆矩阵，其中每个元素表示实际类别和预测类别的特定组合出现的次数。
            混淆矩阵的行表示实际的类别，列表示预测的类别。
    """
    assert torch.all(label > -1) & torch.all(pred > -1)
    assert torch.all(label < n) & torch.all(pred < n)
    # bincount()计算整数张量中每一个值的出现次数
    return torch.bincount(n * label + pred, minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """
    主要用来计算每个类别的IoU分数
    :param hist: 混淆矩阵
    :return:
    """
    # 使用np.errstate(divide="ignore", invalid="ignore")忽略可能出现的除数为零或无效操作的警告
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        return iou


def overall_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist).sum() / hist.sum()


def per_class_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / hist.sum(1)


# CrossEntropyLoss + lovasz_softmax_flat加权组合
class SemSegLoss(nn.Module):
    def __init__(self, nb_class, lovasz_weight=1.0, ignore_index=255):
        super().__init__()
        self.nb_class = nb_class
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, pred, true_label):
        """
        :param pred:(N, C, d1, d2, ..., dK) C是类别总数， d1, d2是其他维度
        :param true_label:(N, d1, d2, ..., dK)
        :return:
        """
        loss = self.ce(pred, true_label)

        pred = pred.transpose(1, 2)
        if self.lovasz_weight > 0:
            where = true_label != self.ignore_index
            if where.sum() > 0:
                loss += self.lovasz_weight * lovasz_softmax_flat(
                    softmax(pred[where], dim=1),
                    true_label[where]
                )
        return loss


# 封装交叉熵函数
class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction="mean",
                 label_smoothing=0.0,
                 loss_weight=1.0,
                 ignore_index=255,
                 ):
        super().__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss






















