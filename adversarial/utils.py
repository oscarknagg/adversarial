from typing import Union, Tuple
from torch.nn import Module
import numpy as np
import torch


def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float):
    """Projects x_adv into the l_norm ball around x"""
    if x.shape != x_adv.shape:
        raise ValueError('Input Tensors must have the same shape')

    if norm == 'inf':
        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x
        if delta.norm(norm) > eps:
            delta *= eps / delta.norm(norm)

        x_adv = x + delta

    return x_adv


def generate_misclassified_sample(model: Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Generates an arbitrary misclassified sample"""
    while True:
        initial = torch.empty_like(x).uniform_(*clamp)

        if model(initial).argmax(dim=1) != y:
            return initial
