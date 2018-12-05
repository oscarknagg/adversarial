from abc import ABC, abstractmethod
from typing import Union, Callable
from torch.nn import Module
import torch


class Attack(ABC):
    """Base class for adversarial attack methods"""
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_adversarial_sample(self, model: Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable, **kwargs):
        raise NotImplementedError


def project(x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float):
    """Projects x_adv into the l_norm ball around x"""
    if norm == 'inf':
        upper = x + eps
        lower = eps

        x_adv[x_adv > upper] = x[x_adv > upper] + eps
        x_adv[x_adv < lower] = x[x_adv < lower] - eps
    else:
        delta = x_adv - x

        if delta.norm(norm) > eps:
            delta *= eps / delta.norm(norm)

        x_adv = x + delta

    return x_adv


class FGSM(Attack):
    """Implements the Fast Gradient-Sign Method (FGSM).

    FGSM is a white box attack.
    """
    def __init__(self, eps: float, **kwargs):
        super(FGSM, self).__init__()
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
    """Implements the iterated Fast Gradient-Sign Method

    This is a form of Projected Gradient Descent (PGD).
    """
    def __init__(self, eps: float, k: int, step: float, norm: Union[str, int] = 'inf', **kwargs):
        super(FGSM_k, self).__init__()
        self.eps = eps
        self.step = step
        self.k = k
        self.norm = norm

    def project(self, x: torch.Tensor, x_adv: torch.Tensor):
        """Projects x_adv into the l_norm ball around x"""
        return project(x, x_adv, self.norm, self.eps)

    def create_adversarial_sample(self, model: Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable,
                                  clamp=(0, 1)):
        x_adv = torch.Tensor(x.data)
        x_adv.requires_grad = True

        for _ in range(self.k):
            # Gradient descent
            prediction = model(x_adv)
            loss = -loss_fn(prediction, y)
            loss.backward()
            # Simply take a step in the direction of the gradient, ignoring
            #
            x_adv = (x_adv - torch.sign(x_adv.grad) * self.step).clamp(*clamp).detach()

            # Project back into l_norm ball
            x_adv = self.project(x, x_adv)

        return x_adv


class PGD(Attack):
    """Implements the Projected Gradient Descent attack"""
    def __init__(self, eps: float, k: int, step: float, norm: Union[str, int] = 'inf', **kwargs):
        super(PGD, self).__init__()
        self.eps = eps
        self.step = step
        self.k = k
        self.norm = norm

    def project(self, x: torch.Tensor, x_adv: torch.Tensor):
        """Projects x_adv into the l_norm ball around x"""
        return project(x, x_adv, self.norm, self.eps)

    def create_adversarial_sample(self, model: Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable,
                                  clamp=(0, 1)):
        x_adv = torch.Tensor(x.data)
        x_adv.requires_grad = True

        for _ in range(self.k):
            # Gradient descent
            prediction = model(x_adv)
            loss = -loss_fn(prediction, y)
            loss.backward()
            x_adv = (x_adv - x_adv.grad * self.step).clamp(*clamp).detach()

            # Project back into l_norm ball
            x_adv = self.project(x, x_adv)

        return x_adv
