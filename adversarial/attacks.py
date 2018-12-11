from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple
from torch.nn import Module
from collections import deque
import torch


class Attack(ABC):
    """Base class for adversarial attack methods"""
    @abstractmethod
    def create_adversarial_sample(self, *args, **kwargs):
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
                                  clamp: Tuple[float, float] = (0, 1)):
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


class IteratedFGSM(Attack):
    """Implements the iterated Fast Gradient-Sign Method"""
    def __init__(self, eps: float, k: int, step: float, norm: Union[str, int] = 'inf', **kwargs):
        super(IteratedFGSM, self).__init__()
        self.eps = eps
        self.step = step
        self.k = k
        self.norm = norm

    def project(self, x: torch.Tensor, x_adv: torch.Tensor):
        """Projects x_adv into the l_norm ball around x"""
        return project(x, x_adv, self.norm, self.eps)

    def create_adversarial_sample(self, model: Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable,
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
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
                                  clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        x_adv = torch.tensor(x.data, device=x.device)
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


class Boundary(Attack):
    """Implements the boundary attack

    This is a black box attack that doesn't require knowledge of the model
    structure. It only requires knowledge of

    https://arxiv.org/pdf/1712.04248.pdf

    Args:
        delta: orthogonal step size (delta in paper)
        eps: perpendicular step size (epsilon in paper)
    """
    def __init__(self, orthogonal_step: float, perpendicular_step: float):
        super(Boundary, self).__init__()
        self.orthogonal_step = orthogonal_step
        self.perpendicular_step = perpendicular_step

        # Deques to keep track of the success rate of the perturbations
        # This is used to dynamically adjust the orthogonal and
        # perpendicular step_sizes
        self.orth_step_stats = deque(maxlen=30)
        self.perp_step_stats = deque(maxlen=30)
        # Factors to adjust step sizes by
        self.orth_step_factor = 0.97
        self.perp_step_factor = 0.97

    def generate_initial(self, model: Module, x: torch.Tensor, y: torch.Tensor,
                         clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
        while True:
            initial = torch.empty_like(x).uniform_(*clamp)

            if model(initial).argmax(dim=1) != y:
                return initial

    def propose(self,
                x: torch.Tensor,
                x_adv: torch.Tensor,
                y: torch.Tensor,
                model: Module,
                clamp: Tuple[float, float] = (0, 1)
                ) -> torch.Tensor:
        """Generate proposal perturbed sample

        Args:
            x: Original sample
            x_adv: Adversarial sample
            y: Label of original sample
            clamp: Domain (i.e. max/min) of samples
        """
        # Sample from unit Normal distribution with same shape as input
        perturbation = torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv))

        # Rescale perturbation so l2 norm is delta
        perturbation = project(torch.zeros_like(perturbation), perturbation, norm=2, eps=self.orthogonal_step)

        # Apply perturbation and project onto sphere around original sample such that the distance
        # between the perturbed adversarial sample and the original sample is the same as the
        # distance between the unperturbed adversarial sample and the original sample
        # i.e. d(x_adv, x) = d(x_adv + perturbation, x)
        perturbed = x_adv + perturbation
        perturbed = project(x, perturbed, 2, torch.norm(x_adv - x, 2)).clamp(*clamp)

        # Record success/failure of orthogonal step
        self.orth_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Make step towards original sample
        step_towards_original = project(torch.zeros_like(perturbation), x - perturbed, norm=2, eps=self.perpendicular_step)
        perturbed = (perturbed + step_towards_original).clamp(*clamp)

        # Record success/failure of perpendicular step
        self.perp_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Clamp to domain of sample
        perturbed = perturbed.clamp(*clamp)

        return perturbed

    def create_adversarial_sample(self, model: Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  loss_fn: Callable,
                                  k: int,
                                  clamp: Tuple[float, float] = (0, 1),
                                  initial: torch.Tensor = None) -> torch.Tensor:
        if x.size(0) != 1:
            # TODO: Attack a whole batch in parallel
            raise NotImplementedError

        if initial is not None:
            x_adv = initial
        else:
            # Generate initial adversarial sample from uniform distribution
            x_adv = self.generate_initial(model, x, y)

        total_stats = torch.zeros(k)

        for i in range(k):
            # Propose perturbation
            perturbed = self.propose(x, x_adv, y, model, clamp)

            # Check if perturbed input is adversarial i.e. gives the wrong prediction
            perturbed_prediction = model(perturbed).argmax(dim=1)
            total_stats[i] = perturbed_prediction != y
            if perturbed_prediction != y:
                x_adv = perturbed

            # Check statistics and adjust step sizes
            if len(self.perp_step_stats) == self.perp_step_stats.maxlen:
                if torch.Tensor(self.perp_step_stats).mean() > 0.5:
                    self.perpendicular_step /= self.perp_step_factor
                    self.orthogonal_step /= self.orth_step_factor
                elif torch.Tensor(self.perp_step_stats).mean() < 0.2:
                    self.perpendicular_step *= self.perp_step_factor
                    self.orthogonal_step *= self.orth_step_factor

            if len(self.orth_step_stats) == self.orth_step_stats.maxlen:
                if torch.Tensor(self.orth_step_stats).mean() > 0.5:
                    self.orthogonal_step /= self.orth_step_factor
                elif torch.Tensor(self.orth_step_stats).mean() < 0.2:
                    self.orthogonal_step *= self.orth_step_factor

        return x_adv
