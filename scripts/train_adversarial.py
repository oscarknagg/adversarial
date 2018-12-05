from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from multiprocessing import cpu_count
from olympic.callbacks import *
import argparse
import olympic

from adversarial.models import MNISTClassifier
from adversarial.attacks import *
from config import PATH


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--attack')
parser.add_argument('--eps', type=float)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

if args.dataset != 'mnist':
    raise NotImplementedError


########
# Data #
########
transform = transforms.Compose([
   transforms.ToTensor(),
])

train = datasets.MNIST(f'{PATH}/data/', train=True, transform=transform, download=True)
val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)

train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
val_loader = DataLoader(val, batch_size=128, num_workers=cpu_count())

#########
# Model #
#########
model = MNISTClassifier().to(args.device)
optimiser = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

#################
# Training loop #
#################
callbacks = [
    Evaluate(val_loader),
    ReduceLROnPlateau(monitor='val_accuracy', patience=5),
    ModelCheckpoint(
        f'{PATH}/models/{args.dataset}_attack={args.attack}_eps={args.eps}.pt',
        save_best_only=True,
        monitor='val_loss',
        verbose=True
    ),
    CSVLogger(f'{PATH}/logs/{args.dataset}_attack={args.attack}_eps={args.eps}.csv')
]


def adversarial_update_fgsm(model, optimiser, loss_fn, x, y, epoch, eps, **kwargs):
    """Performs a single update against an adversary"""
    model.train()

    # Adversial perturbation
    x_adv = FGSM(eps).create_adversarial_sample(model, x, y, loss_fn)

    optimiser.zero_grad()
    y_pred = model(x_adv)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


olympic.fit(
    model,
    optimiser,
    loss_fn,
    dataloader=train_loader,
    epochs=10,
    metrics=['accuracy'],
    callbacks=callbacks,
    update_fn=adversarial_update_fgsm,
    update_fn_kwargs={'eps': args.eps},
    prepare_batch=lambda batch: (batch[0].to(args.device), batch[1].to(args.device))
)
