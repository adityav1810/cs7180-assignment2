"""
File: Loss.py

Description: 
    Classes to define all Loss functions and track them .

Authors:
    Author 1 (Aditya Varshney,varshney.ad@northeastern.edu, Northeastern University)
    Author 2 (Luv Verma, verma.lu@northeastern.edu , Northeastern University)

Citations and References:
    - Reference 1: https://github.com/matteo-rizzo/fc4-pytorch
    
"""

import math
import torch
from torch import Tensor
from torch.nn.functional import normalize


class Loss:
    def __init__(self, device: torch.device):
        self._device = device

    def _compute(self, *args, **kwargs) -> Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self._compute(*args).to(self._device)

class AngularLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def _compute(self, pred: Tensor, label: Tensor, safe_v: float = 0.999999) -> Tensor:
        dot = torch.clamp(torch.sum(normalize(pred, dim=1) * normalize(label, dim=1), dim=1), -safe_v, safe_v)
        angle = torch.acos(dot) * (180 / math.pi)
        return torch.mean(angle).to(self._device)

class LossTracker(object):

    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_loss(self) -> float:
        return self.avg
