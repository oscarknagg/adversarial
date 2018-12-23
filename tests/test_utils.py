import unittest
import numpy as np
import torch

from adversarial.utils import *


shape = (1, 28, 28)


class TestProject(unittest.TestCase):
    def test_2_norm(self):
        norm = 2
        eps = 0.5

        # Test batchwise projection
        x = torch.zeros((2, ) + shape)
        y = torch.cat([
            torch.ones((1, ) + shape),
            torch.zeros((1,) + shape)
        ])

        y_proj = project(x, y, norm, eps)

        within_norm_balls = y_proj.view(y_proj.shape[0], -1).norm(norm, dim=-1) <= eps
        self.assertTrue(torch.all(within_norm_balls))

    def test_inf_norm(self):
        norm = 'inf'
        eps = 0.5

        # Test batchwise projection
        x = torch.zeros((2,) + shape)
        y = torch.cat([
            torch.ones((1,) + shape),
            torch.zeros((1,) + shape)
        ])

        y_proj = project(x, y, norm, eps)

        within_norm_balls = y_proj.view(y_proj.shape[0], -1).norm(float(norm), dim=-1) <= eps
        self.assertTrue(torch.all(within_norm_balls))


class TestRandomPerturbation(unittest.TestCase):
    def test_2_norm(self):
        norm = 2
        eps = 0.5

        x = torch.zeros((2,) + shape)
        x_ = random_perturbation(x, norm, eps)
        within_norm_balls = x_.view(x_.shape[0], -1).norm(norm, dim=-1) <= eps
        self.assertTrue(torch.all(within_norm_balls))

    def test_inf_norm(self):
        norm = 'inf'
        eps = 0.5

        x = torch.zeros((2,) + shape)
        x_ = random_perturbation(x, norm, eps)
        within_norm_balls = x_.view(x_.shape[0], -1).norm(float(norm), dim=-1) <= eps
        self.assertTrue(torch.all(within_norm_balls))
