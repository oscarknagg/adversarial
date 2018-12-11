import unittest
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from adversarial.attacks import FGSM, Boundary
from adversarial.models import MNISTClassifier
from config import PATH


class TestBoundaryAttack(unittest.TestCase):
    def test_attack(self):
        device = 'cuda'

        model = MNISTClassifier()
        model.load_state_dict(torch.load(f'{PATH}/models/mnist_natural.pt'))
        model.to(device)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)
        val_loader = DataLoader(val, batch_size=1, num_workers=0)

        boundary_attacker = Boundary(1e-2, 1e-2)

        x, y = val_loader.__iter__().__next__()
        x = x.to(device)
        y = y.to(device)

        x_adv = boundary_attacker.create_adversarial_sample(model, x, y, torch.nn.CrossEntropyLoss(), 500)

        self.assertLess(
            (torch.norm(x - x_adv, 2).pow(2) / x.numel()).item(),
            0.1
        )
