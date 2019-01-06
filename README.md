# adversarial

This repository contains PyTorch code to create and defend against
adversarial attacks.

See [this Medium article](https://towardsdatascience.com/know-your-enemy-7f7c5038bdf3)
for a discussion on how to use and defend against
the projected gradient attack.

![PGD Attack](https://github.com/oscarknagg/few-shot/blob/master/assets/pgd_attack_imagenet_example.png)

# Contents

- A Jupyter notebook demonstrating how to use and defend against
the projected gradient attack (see `notebooks/`)

- `adversarial.functional` contains functional style implementations of
a view different types of adversarial attacks
    - Fast Gradient Sign Method - white box - batch implementation
    - Projected Gradient Descent - white box - batch implementation
    - Local-search attack - black box, score-based - single image
    - Boundary attack - black box, decision-based - single imagae


# Setup
## Requirements

Listed in `requirements.txt`. Install with
`pip install -r requirements.txt` preferably in a virtualenv.

## Tests (optional)

Run `pytest` in the root directory to run all tests.