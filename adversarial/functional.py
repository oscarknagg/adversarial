from typing import Union, Callable, Tuple
from collections import deque
from torch.nn import Module
import torch

from adversarial.utils import project, generate_misclassified_sample


def fgsm(model: Module,
         x: torch.Tensor,
         y: torch.Tensor,
         loss_fn: Callable,
         eps: float,
         clamp: Tuple[float, float] = (0, 1)):
    """Creates an adversarial sample using the Fast Gradient-Sign Method (FGSM)

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        eps: Size of adversarial perturbation
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    x.requires_grad = True
    model.train()
    prediction = model(x)
    loss = -loss_fn(prediction, y)
    loss.backward()

    x_adv = (x - torch.sign(x.grad) * eps).clamp(*clamp).detach()

    return x_adv


def iterated_fgsm(model: Module,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  loss_fn: Callable,
                  k: int,
                  step: float,
                  eps: float,
                  norm: Union[str, float],
                  random: bool = False,
                  clamp: Tuple[float, float] = (0, 1)):
    """Creates an adversarial sample using the iterated Fast Gradient-Sign Method

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        random: Whether to start Iterated FGSM within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    x_adv = torch.tensor(x.data, device=x.device, requires_grad=True)
    if random:
        x_adv = x_adv + torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv)) * eps

    for i in range(k):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        # Gradient descent
        prediction = model(_x_adv)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            x_adv += _x_adv.grad.sign() * step

        # Project back into l_norm ball and correct range
        x_adv = project(x, x_adv, norm, eps).clamp(*clamp)

    return x_adv.detach()


def pgd(model: Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Callable,
        k: int,
        step: float,
        eps: float,
        norm: Union[str, float],
        random: bool = False,
        clamp: Tuple[float, float] = (0, 1)):
    """Creates an adversarial sample using the Projected Gradient Descent Method

    This is a white-box attack.

    Args:
        model: Model
        x: Batch of samples
        y: Corresponding labels
        loss_fn: Loss function to maximise
        k: Number of iterations to make
        step: Size of step to make at each iteration
        eps: Maximum size of adversarial perturbation, larger perturbations will be projected back into the
            L_norm ball
        norm: Type of norm
        random: Whether to start PGD within a random point in the l_norm ball
        clamp: Max and minimum values of elements in the samples i.e. (0, 1) for MNIST

    Returns:
        x_adv: Adversarially perturbed version of x
    """
    x_adv = torch.tensor(x.data, device=x.device, requires_grad=True)
    if random:
        x_adv = x_adv + torch.normal(torch.zeros_like(x_adv), torch.ones_like(x_adv)) * eps

    for i in range(k):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        # Gradient descent
        prediction = model(_x_adv)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            x_adv += _x_adv.grad * step

        # Project back into l_norm ball and correct range
        x_adv = project(x, x_adv, norm, eps).clamp(*clamp)

    return x_adv.detach()


def boundary(model: Module,
             x: torch.Tensor,
             y: torch.Tensor,
             k: int,
             orthogonal_step: float = 1e-2,
             perpendicular_step: float = 1e-2,
             initial: torch.Tensor = None,
             clamp: Tuple[float, float] = (0, 1)):
    """Implements the boundary attack

    This is a black box attack that doesn't require knowledge of the model
    structure. It only requires knowledge of

    https://arxiv.org/pdf/1712.04248.pdf

    Args:
        model:
        x:
        y:
        k: Number of steps to take
        orthogonal_step: orthogonal step size (delta in paper)
        perpendicular_step: perpendicular step size (epsilon in paper)
        initial:
        clamp:

    Returns:
        x_adv:
    """
    orth_step_stats = deque(maxlen=30)
    perp_step_stats = deque(maxlen=30)
    # Factors to adjust step sizes by
    orth_step_factor = 0.97
    perp_step_factor = 0.97

    def _propose(x: torch.Tensor,
                 x_adv: torch.Tensor,
                 y: torch.Tensor,
                 model: Module,
                 clamp: Tuple[float, float] = (0, 1)) -> torch.Tensor:
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
        perturbation = project(torch.zeros_like(perturbation), perturbation, norm=2, eps=orthogonal_step)

        # Apply perturbation and project onto sphere around original sample such that the distance
        # between the perturbed adversarial sample and the original sample is the same as the
        # distance between the unperturbed adversarial sample and the original sample
        # i.e. d(x_adv, x) = d(x_adv + perturbation, x)
        perturbed = x_adv + perturbation
        perturbed = project(x, perturbed, 2, torch.norm(x_adv - x, 2)).clamp(*clamp)

        # Record success/failure of orthogonal step
        orth_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Make step towards original sample
        step_towards_original = project(torch.zeros_like(perturbation), x - perturbed, norm=2, eps=perpendicular_step)
        perturbed = (perturbed + step_towards_original).clamp(*clamp)

        # Record success/failure of perpendicular step
        perp_step_stats.append(model(perturbed).argmax(dim=1) != y)

        # Clamp to domain of sample
        perturbed = perturbed.clamp(*clamp)

        return perturbed

    if x.size(0) != 1:
        # TODO: Attack a whole batch in parallel
        raise NotImplementedError

    if initial is not None:
        x_adv = initial
    else:
        # Generate initial adversarial sample from uniform distribution
        x_adv = generate_misclassified_sample(model, x, y)

    total_stats = torch.zeros(k)

    for i in range(k):
        # Propose perturbation
        perturbed = _propose(x, x_adv, y, model, clamp)

        # Check if perturbed input is adversarial i.e. gives the wrong prediction
        perturbed_prediction = model(perturbed).argmax(dim=1)
        total_stats[i] = perturbed_prediction != y
        if perturbed_prediction != y:
            x_adv = perturbed

        # Check statistics and adjust step sizes
        if len(perp_step_stats) == perp_step_stats.maxlen:
            if torch.Tensor(perp_step_stats).mean() > 0.5:
                perpendicular_step /= perp_step_factor
                orthogonal_step /= orth_step_factor
            elif torch.Tensor(perp_step_stats).mean() < 0.2:
                perpendicular_step *= perp_step_factor
                orthogonal_step *= orth_step_factor

        if len(orth_step_stats) == orth_step_stats.maxlen:
            if torch.Tensor(orth_step_stats).mean() > 0.5:
                orthogonal_step /= orth_step_factor
            elif torch.Tensor(orth_step_stats).mean() < 0.2:
                orthogonal_step *= orth_step_factor

    return x_adv
