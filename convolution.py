#!/usr/bin/env python3.8

import torch.nn as nn
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn.functional as F
import time
from datetime import timedelta
from math import ceil


def learn_identity(channels, kernel):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)

    print(
        f"\n=== Learning convolutional identity for channels = {channels}, kernel = {kernel} ==="
    )
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(f"cuda:0")

    x = torch.randn((8, channels, 28, 28)).to(device)

    model = nn.Conv2d(
        channels,
        channels,
        kernel_size=kernel,
        stride=1,
        padding=(kernel[0] // 2, kernel[1] // 2),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    learning_rate = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    start_time = time.time()

    max_abs_err = torch.ones(1).to(device)
    zero = torch.zeros(1).to(device)
    epoch = 0
    while not torch.isclose(max_abs_err, zero, atol=2e-5):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, x)
        if (epoch + 1) % 500 == 0:
            learning_rate.step()
            print(
                f"\n=== Epoch {epoch + 1} ===\nloss = {loss.item():.8f}\nlearning rate = {optimizer.param_groups[0]['lr']:.8f}"
            )
            abs_err = torch.abs(output - x)
            max_abs_err = torch.max(abs_err).to(device)
            print(
                f"Absolute performance:\nmax abs err = {max_abs_err:.8f}\nmean abs err = {torch.mean(abs_err):.8f}"
            )
        loss.backward()
        optimizer.step()
        epoch += 1

    end_time = time.time()
    print(
        f"\nAfter {epoch} epochs ({timedelta(seconds=ceil(end_time-start_time))} hh:mm:ss)\nLearned parameters:",
        list(model.parameters()),
    )


learn_identity(4, 3)
learn_identity(32, 3)
learn_identity(64, 3)
learn_identity(256, 3)
learn_identity(32, 7)
learn_identity(64, 7)
learn_identity(256, 7)
