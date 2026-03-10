"""
损失函数：有序分类损失和评估指标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def true_metric_loss(true, no_of_classes, scale=1):
    """
    生成有序分类的软标签
    """
    batch_size = true.size(0)
    true = true.view(batch_size, 1)
    
    if not isinstance(true, torch.cuda.LongTensor):
        true = true.long().to(true.device)
    true_labels = true.repeat(1, no_of_classes).float()
    
    class_labels = torch.arange(no_of_classes, dtype=torch.float32, device=true.device)
    phi = (scale * torch.abs(class_labels - true_labels))
    y = nn.Softmax(dim=1)(-phi)
    return y


def loss_function(output, labels, loss_type="ordered", expt_type=5, scale=2.5):
    """
    支持多种损失函数类型，当前主要使用有序分类损失
    """
    if loss_type == "ordered":
        targets = true_metric_loss(labels, expt_type, scale)
        return torch.sum(-targets * F.log_softmax(output, -1), -1).mean()
    else:
        return F.cross_entropy(output, labels)


def gr_metrics(op, t):
    """
    计算分级精度 GP、召回 GR、F-score FS 和过估计错误率 OE
    """
    op = np.array(op)
    t = np.array(t)
    
    TP = (op == t).sum()
    FN = (t > op).sum()
    FP = (t < op).sum()
    
    GP = TP / (TP + FP)
    GR = TP / (TP + FN)
    FS = 2 * GP * GR / (GP + GR) if (GP + GR) > 0 else 0.0
    OE = (np.abs(t - op) > 1).sum() / op.size
    
    return GP, GR, FS, OE
