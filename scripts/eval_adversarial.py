from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torchvision import transforms, datasets
from olympic import evaluate
from olympic.metrics import accuracy
from torch.nn.functional import cross_entropy
import torch
import argparse

from adversarial.models import MNISTClassifier
from adversarial.attacks import FGSM
from config import PATH


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--attack')
parser.add_argument('--model')
parser.add_argument('--device')
args = parser.parse_args()

if args.dataset != 'mnist':
    raise NotImplementedError


# Get model
model = MNISTClassifier()
model.load_state_dict(torch.load(f'{PATH}/models/{args.model}.pt'))
model.to(args.device)


# Get data
transform = transforms.Compose([
   transforms.ToTensor(),
])
train = datasets.MNIST(f'{PATH}/data/', train=True, transform=transform, download=True)
val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)
train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
val_loader = DataLoader(val, batch_size=128, num_workers=cpu_count())


def fgsm_attack(model, loss_fn, eps):
    fgsm = FGSM(eps)

    def fgsm_attack_(batch):
        x, y = batch
        x = x.to(args.device)
        y = y.to(args.device)
        x_adv = fgsm.create_adversarial_sample(model, x, y, loss_fn)
        return x_adv, y

    attacker = fgsm_attack_

    return attacker


# Evaluate accuracy on FGSM
for eps in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:

    # acc = evaluate(model,
    #                val_loader,
    #                ['accuracy'],
    #                prepare_batch=fgsm_attack(model, cross_entropy, eps),
    #                loss_fn=cross_entropy)
    #
    # print(acc)

    fgsm = FGSM(eps)
    total = 0
    acc = 0
    for x, y in val_loader:
        total += x.size(0)

        x_adv = fgsm.create_adversarial_sample(model, x.to(args.device), y.to(args.device), cross_entropy)

        y_pred = model(x_adv)

        acc += accuracy(y.to(args.device), y_pred) * x.size(0)

    print(eps, acc/total)

