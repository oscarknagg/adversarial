import unittest
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from adversarial.datasets import RestrictedImageNet


class TestRestrictedImageNet(unittest.TestCase):
    def test_dataset(self):
        data = RestrictedImageNet()

        dataloader = DataLoader(data, batch_size=128, num_workers=cpu_count())

        for batch in dataloader:
            print(batch[0].shape)
            break
