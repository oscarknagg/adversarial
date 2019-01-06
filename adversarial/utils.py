from typing import Union, Tuple
from torch.nn import Module
import numpy as np
import torch


def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float) -> torch.Tensor:
    """Projects x_adv into the l_norm ball around x

    Assumes x and x_adv are 4D Tensors representing batches of images

    Args:
        x: Batch of natural images
        x_adv: Batch of adversarial images
        norm: Norm of ball around x
        eps: Radius of ball

    Returns:
        x_adv: Adversarial examples projected to be at most eps
            distance from x under a certain norm
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
        norm: Norm to measure size of perturbation
        eps: Size of perturbation

    Returns:
        x_perturbed: Randomly perturbed version of x
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
    """Generates an arbitrary misclassified sample

    Args:
        model: Model that must misclassify
        x: Batch of image data
        y: Corresponding labels
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_misclassified: A sample for the model that is not classified correctly
    """
    while True:
        x_misclassified = torch.empty_like(x).uniform_(*clamp)

        if model(x_misclassified).argmax(dim=1) != y:
            return x_misclassified
