from abc import ABC, abstractmethod
from typing import Union, Callable
from torch.nn import Module
import torch


class Attack(ABC):
    """Base class for adversarial attack methods"""

    @abstractmethod
    def create_adversarial_sample(self, model: Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable, **kwargs):
        raise NotImplementedError


class FGSM(Attack):
    """Implements the Fast Gradient-Sign Method (FGSM).

    FGSM is a white box attack.
    """
    def __init__(self, eps: float):
        self.eps = eps

    def create_adversarial_sample(self, model: Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable,
                                  clamp=(0, 1)):
        """Creates an adversarial sample

        Args:
            model: Model
            x: Batch of samples
            y: Corresponding labels
            loss_fn: Loss function to maximise
            clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

        Returns:
            x_adv: Adversarially perturbed version of x
        """
        x.requires_grad = True
        model.train()
        prediction = model(x)
        loss = -loss_fn(prediction, y)
        loss.backward()

        x_adv = (x-torch.sign(x.grad)*self.eps).clamp(*clamp).detach()

        return x_adv


class FGSM_k(Attack):
    """Implements the iterated Fast Gradient-Sign Method"""
    def __init__(self, eps: float, norm: Union[int, str], k: int):
        self.eps = eps
        self.norm = norm
        self.k = k

    def create_adversarial_sample(self, model, x, y, loss_fn, **kwargs):
        raise NotImplementedError


class PGD(Attack):
    """Implements the Projected Gradient Descent attack method."""
    def __init__(self):
        pass

    def create_adversarial_sample(self, model, x, y, loss_fn, **kwargs):
        raise NotImplementedError
