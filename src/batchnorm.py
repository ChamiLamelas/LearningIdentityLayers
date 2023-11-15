#!/usr/bin/env python3.8

import torch.nn as nn
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import random


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


def mse(a, b):
    err = torch.sum(torch.square(a - b))
    return (
        0
        if torch.isclose(err, torch.zeros(1).to(torch.device("cuda:0")))
        else err.item()
    )


def update_statistics(model):
    loader = load_cifar10(True, 64)

    evaluation_idx = random.randint(0, len(loader) - 1)

    model.train()
    for i, (data, _) in enumerate(loader):
        data = data.to(torch.device("cuda:0"))
        model(data)

        if i == evaluation_idx:
            evaluation_image = data

    return evaluation_image


def set_weights(model):
    model.weight = nn.Parameter(torch.sqrt(model.running_var + model.eps))
    model.bias = nn.Parameter(model.running_mean)


def evaluate(model, evaluation_image):
    model.eval()
    print(f"{mse(evaluation_image, model(evaluation_image)):.6f}")


def main():
    model = nn.BatchNorm2d(3).to(torch.device("cuda:0"))

    evaluation_image = update_statistics(model)
    set_weights(model)
    evaluate(model, evaluation_image)


if __name__ == "__main__":
    main()
