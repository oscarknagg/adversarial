import unittest
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from adversarial.functional import iterated_fgsm
from adversarial.models import MNISTClassifier
from config import PATH


class TestIteratedFGSMAttack(unittest.TestCase):
    def test_attack(self):
        device = 'cuda'
        k = 100
        step = 1
        eps = 0.3
        norm = 'inf'

        model = MNISTClassifier()
        model.load_state_dict(torch.load(f'{PATH}/models/mnist_natural.pt'))
        model.to(device)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)
        val_loader = DataLoader(val, batch_size=1, num_workers=0)

        x, y = val_loader.__iter__().__next__()
        x = x.to(device)
        y = y.to(device)

        x_adv = iterated_fgsm(model, x, y, torch.nn.CrossEntropyLoss(), k=k, step=step, eps=eps, norm=norm)

        # Check that adversarial example is misclassified
        self.assertNotEqual(model(x_adv).argmax(dim=1).item(), y.item())

        # Assert that distance between adversarial and natural sample is small
        self.assertLess(
            (torch.norm(x - x_adv, 2).pow(2) / x.numel()).item(),
            eps
        )
