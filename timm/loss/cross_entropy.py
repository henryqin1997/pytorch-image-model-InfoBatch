""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction=='mean':
            return loss.mean()
        if self.reduction=='none':
            return loss


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

class SoftTargetCrossEntropyNoReduction(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropyNoReduction, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss


class SoftTargetCrossEntropyInfoV2(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropyInfoV2, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, lam: Union[float,torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(lam,torch.Tensor):
                p = F.softmax(x,dim=-1)
                scores = torch.max(torch.abs(target-p),dim=-1)[0]
                scores = (scores + scores.flip(0))/(lam+(1-lam).flip(0))
            else:
                p = F.softmax(x,dim=-1)
                scores = torch.max(torch.abs(target-p),dim=-1)[0]
                scores = scores + scores.flip(0)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean(), scores