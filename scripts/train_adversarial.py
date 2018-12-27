from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from multiprocessing import cpu_count
from olympic.callbacks import *
import argparse
import olympic

from adversarial.models import MNISTClassifier
from adversarial.attacks import *
from adversarial.functional import *
from adversarial.datasets import RestrictedImageNet
from config import PATH


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--attack')
parser.add_argument('--eps', type=float)
parser.add_argument('--step', type=float)
parser.add_argument('--k', type=int)
parser.add_argument('--norm', default='inf')
parser.add_argument('--device', default='cuda')
parser.add_argument('--epochs', type=int)
args = parser.parse_args()


if args.norm != 'inf':
    norm = int(args.norm)
else:
    norm = args.norm


########
# Data #
########
if args.dataset == 'mnist':
    transform = transforms.Compose([
       transforms.ToTensor(),
    ])

    train = datasets.MNIST(f'{PATH}/data/', train=True, transform=transform, download=True)
    val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)

    train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
    val_loader = DataLoader(val, batch_size=128, num_workers=cpu_count())
elif args.dataset == 'restricted_imagenet':
    rng = np.random.RandomState(0)
    data = RestrictedImageNet()

    indices = np.array(range(len(data)))
    rng.shuffle(indices)
    train_indices = indices[:int(len(indices)*0.9)]
    val_indices = indices[int(len(indices)*0.9):]

    train = Subset(data, train_indices)
    val = Subset(data, val_indices)

    train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
    val_loader = DataLoader(val, batch_size=128, num_workers=cpu_count())
else:
    raise ValueError('Unsupported dataset')


#########
# Model #
#########
if args.dataset == 'mnist':
    model = MNISTClassifier().to(args.device)
elif args.dataset == 'restricted_imagenet':
    model = models.resnet50(num_classes=RestrictedImageNet().num_classes()).to(args.device)
else:
    raise ValueError('Unsupported norm')

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


def adversarial_update(model, optimiser, loss_fn, x, y, epoch, eps, step, k, norm, **kwargs):
    """Performs a single update against an adversary"""
    model.train()

    # Adversial perturbation
    if norm == 'inf':
        x_adv = iterated_fgsm(model, x, y, loss_fn, k=k, step=step, eps=eps, norm='inf')
    elif norm == 2:
        x_adv = pgd(model, x, y, loss_fn, k=k, step=step, eps=eps, norm=2)
    else:
        raise ValueError('Unsupported norm')

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
    epochs=args.epochs,
    metrics=['accuracy'],
    callbacks=callbacks,
    update_fn=adversarial_update,
    update_fn_kwargs={'eps': args.eps, 'step': args.step, 'norm': norm, 'k': args.k},
    prepare_batch=lambda batch: (batch[0].to(args.device), batch[1].to(args.device))
)
