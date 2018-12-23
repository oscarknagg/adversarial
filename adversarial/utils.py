from typing import Union, Tuple
from torch.nn import Module
import numpy as np
import torch


def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Projects x_adv into the l_norm ball around x

    Assumes x and x_adv are 4D Tensors representing batches of images

    Args:
        x:
        x_adv:
        norm:
        eps:

    Returns:
        x_adv:
    """
    if x.shape != x_adv.shape:
        raise ValueError('Input Tensors must have the same shape')

    if norm == 'inf':
        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x

        # Assume x and x_adv are batched tensors where the first dimension is
        # a batch dimension
        mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= eps

        scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
        scaling_factor[mask] = eps

        # .view() assumes batched images as a 4D Tensor
        delta *= eps / scaling_factor.view(-1, 1, 1, 1)

        x_adv = x + delta

    return x_adv


def random_perturbation(x: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Applies a random l_norm bounded perturbation to x

    Assumes x is a 4D Tensor representing a batch of images

    Args:
        x: Batch of images
        norm:
        eps:

    Returns:
        x_perturbed:
    """
    perturbation = torch.normal(torch.zeros_like(x), torch.ones_like(x))
    if norm == 'inf':
        perturbation = torch.sign(perturbation) * eps
    else:
        perturbation = project(torch.zeros_like(x), perturbation, norm, eps)

    return x + perturbation


def generate_misclassified_sample(model: Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """Generates an arbitrary misclassified sample"""
    while True:
        initial = torch.empty_like(x).uniform_(*clamp)

        if model(initial).argmax(dim=1) != y:
            return initial
