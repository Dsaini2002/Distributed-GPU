import torch.nn as nn


def get_model():
    return nn.Sequential(
        nn.Linear(8, 128),
        nn.ReLU(),

        nn.Linear(128, 256),
        nn.ReLU(),

        nn.Linear(256, 128),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Linear(64, 1)
    )
