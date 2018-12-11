import unittest
import numpy as np
import torch

from adversarial.utils import *


class TestProject(unittest.TestCase):
    def test_2_norm(self):
        norm = 2
        eps = 0.5
        x = torch.Tensor([0, 0])
        y = torch.Tensor([1, 1])

        y_proj = project(x, y, norm, eps)

        self.assertTrue(np.isclose(y_proj.norm(norm).item(), eps))

    def test_inf_norm(self):
        norm = 'inf'
        eps = 0.5
        x = torch.Tensor([0, 0])
        y = torch.Tensor([1, -1])

        y_proj = project(x, y, norm, eps)

        self.assertEqual(y_proj.norm(float(norm)).item(), eps)
