#!/usr/bin/env python3.8

import torch.nn as nn
import random
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms


def load_cifar10(train, batch_size):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((299, 299), antialias=True),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.norm1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        return x


def set_seed():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main():
    set_seed()

    model = Model()
    model.train()

    print(" === Before === ")
    print(model.norm1.running_mean)
    print(model.norm1.running_var)

    for data, _ in load_cifar10(True, 64):
        model(data)

    print(" === After === ")
    print(model.norm1.running_mean)
    print(model.norm1.running_var)



if __name__ == "__main__":
    main()
