from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from multiprocessing import cpu_count
from olympic.callbacks import *
import argparse
import olympic

from adversarial.models import MNISTClassifier
from config import PATH

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

if args.dataset != 'mnist':
    raise NotImplementedError


transform = transforms.Compose([
   transforms.ToTensor(),
])

train = datasets.MNIST(f'{PATH}/data/', train=True, transform=transform, download=True)
val = datasets.MNIST(f'{PATH}/data/', train=False, transform=transform, download=True)

train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
val_loader = DataLoader(val, batch_size=128, num_workers=cpu_count())

model = MNISTClassifier().to(args.device)
optimiser = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

callbacks = [
    Evaluate(val_loader),
    ReduceLROnPlateau(monitor='val_accuracy', patience=5),
    ModelCheckpoint(f'{PATH}/models/{args.dataset}_natural.pt', save_best_only=True, monitor='val_loss', verbose=True),
    CSVLogger(f'{PATH}/logs/{args.dataset}_natural.csv')
]

olympic.fit(
    model,
    optimiser,
    loss_fn,
    dataloader=train_loader,
    epochs=10,
    metrics=['accuracy'],
    callbacks=callbacks,
    prepare_batch=lambda batch: (batch[0].to(args.device), batch[1].to(args.device))
)
