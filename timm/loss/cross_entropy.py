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
                if lam>0.5:
                    original_targets = torch.max(target,dim=-1)[1]
                else:
                    original_targets = torch.max(target,dim=-1)[1].flip(0)
                p = F.softmax(x,dim=-1)
                selfscores = torch.abs(target[range(len(target)),original_targets]-p[range(len(target)),original_targets])
                mixscores = torch.abs(target[range(len(target)),original_targets.flip(0)]-p[range(len(target)),original_targets.flip(0)])
                scores = selfscores + mixscores.flip(0)


        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss, scores


class SoftTargetCrossEntropyInfoV3(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropyInfoV2, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, lam: Union[float,torch.Tensor], res_weights: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            rescaled_samples = torch.where(res_weights>1)[0]
            unrescaled_samples = torch.where(res_weights<=1)[0]
            batch_permuted = torch.zeros(x.shape[0], dtype=torch.int64, x.device)
            batch_permuted[rescaled_samples] = rescaled_samples.flip(0)
            batch_permuted[unrescaled_samples] = unrescaled_samples.flip(0)
            if isinstance(lam,torch.Tensor):
                p = F.softmax(x,dim=-1)
                scores = torch.max(torch.abs(target-p),dim=-1)[0]
                scores = (scores + scores[batch_permuted])/(lam+(1-lam)[batch_permuted]))
            else:
                if lam>0.5:
                    original_targets = torch.max(target,dim=-1)[1]
                else:
                    original_targets = torch.max(target,dim=-1)[1][[batch_permuted]]
                p = F.softmax(x,dim=-1)
                selfscores = torch.abs(target[range(len(target)),original_targets]-p[range(len(target)),original_targets])
                mixscores = torch.abs(target[range(len(target)),original_targets[[batch_permuted]]]-p[range(len(target)),original_targets[[batch_permuted]]])
                scores = selfscores + mixscores[[batch_permuted]]

        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss, scores