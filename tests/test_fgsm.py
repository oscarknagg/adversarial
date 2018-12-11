import unittest
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

from adversarial.functional import fgsm
from adversarial.models import MNISTClassifier
from config import PATH


class TestFGSMAttack(unittest.TestCase):
    def test_attack(self):
        device = 'cuda'
        eps = 0.25

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

        x_adv = fgsm(model, x, y, torch.nn.CrossEntropyLoss(), eps)

        # Check that adversarial example is misclassified
        self.assertNotEqual(model(x_adv).argmax(dim=1).item(), y.item())

        # Check that l_inf distance is as expected
        self.assertTrue(
            np.isclose(torch.norm(x - x_adv, float("inf")).item(), eps)
        )
