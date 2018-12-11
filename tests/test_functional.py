import unittest
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from adversarial.functional import *
from adversarial.models import MNISTClassifier
from config import PATH


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TestAttacks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MNISTClassifier()
        cls.model.load_state_dict(torch.load(f'{PATH}/models/mnist_natural.pt'))
        cls.model.to(DEVICE)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        cls.val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)
        cls.val_loader = DataLoader(cls.val, batch_size=1, num_workers=0)

        x, y = cls.val_loader.__iter__().__next__()
        cls.x = x.to(DEVICE)
        cls.y = y.to(DEVICE)

    def test_fgsm(self):
        eps = 0.25

        # x, y = self.val_loader.__iter__().__next__()
        # x = x.to(device)
        # y = y.to(device)

        x_adv = fgsm(self.model, self.x, self.y, torch.nn.CrossEntropyLoss(), eps)

        # Check that adversarial example is misclassified
        self.assertNotEqual(self.model(x_adv).argmax(dim=1).item(), self.y.item())

        # Check that l_inf distance is as expected
        self.assertTrue(
            np.isclose(torch.norm(self.x - x_adv, float("inf")).item(), eps)
        )

    def test_iterated_fgsm_untargeted(self):
        k = 100
        step = 1
        eps = 0.3
        norm = 'inf'

        x_adv = iterated_fgsm(self.model, self.x, self.y, torch.nn.CrossEntropyLoss(), k=k, step=step, eps=eps, norm=norm)

        # Check that adversarial example is misclassified
        self.assertNotEqual(self.model(x_adv).argmax(dim=1).item(), self.y.item())

        # Assert that distance between adversarial and natural sample is
        # less than specified amount
        adversarial_distance = (self.x - x_adv).norm(float(norm)).item()
        self.assertTrue(
            np.isclose(adversarial_distance, eps) or adversarial_distance < eps
        )

    def test_iterated_fgsm_targeted(self):
        k = 100
        step = 0.1
        eps = 0.3
        norm = 'inf'
        target = torch.Tensor([0]).long().to(DEVICE)

        x_adv = iterated_fgsm(self.model, self.x, self.y, torch.nn.CrossEntropyLoss(), y_target=target, k=k, step=step,
                              eps=eps, norm=norm)

        # Check that adversarial example is classified as the target class
        self.assertEqual(self.model(x_adv).argmax(dim=1).item(), target.item())

        # Assert that distance between adversarial and natural sample is
        # less than specified amount
        adversarial_distance = (self.x - x_adv).norm(float(norm)).item()
        self.assertTrue(
            np.isclose(adversarial_distance, eps) or adversarial_distance < eps
        )

    def test_pgd_untargeted(self):
        k = 100
        eps = 2.0
        step = 0.1
        norm = 2

        x_adv = pgd(self.model, self.x, self.y, torch.nn.CrossEntropyLoss(), k, step, eps=eps, norm=norm)

        # Check that adversarial example is misclassified
        self.assertNotEqual(self.model(x_adv).argmax(dim=1).item(), self.y.item())

        # Assert that distance between adversarial and natural sample is small
        self.assertLess(
            (self.x - x_adv).norm(norm).item(),
            eps
        )

    def test_pgd_targeted(self):
        k = 10
        eps = 5
        step = 1
        norm = 2
        target = torch.Tensor([0]).long().to(DEVICE)

        x_adv = pgd(self.model, self.x, self.y, torch.nn.CrossEntropyLoss(), k, step,
                    y_target=target, eps=eps, norm=norm)

        # Check that adversarial example is misclassified
        self.assertNotEqual(self.model(x_adv).argmax(dim=1).item(), self.y.item())

        # Assert that distance between adversarial and natural sample is small
        self.assertLess(
            (self.x - x_adv).norm(norm).item(),
            eps
        )

    def test_boundary_untargeted(self):
        x_adv = boundary(self.model, self.x, self.y, 500)

        # Check that adversarial example is misclassified
        self.assertNotEqual(self.model(x_adv).argmax(dim=1).item(), self.y.item())

        # Assert that distance between adversarial and natural sample is small
        self.assertLess(
            (torch.norm(self.x - x_adv, 2).pow(2) / self.x.numel()).item(),
            0.1
        )
