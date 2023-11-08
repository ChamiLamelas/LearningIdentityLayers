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


def learn_identity(dimension):
    print(f"\n=== Learning linear identity for dimension = {dimension} ===")
    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(f"cuda:0")

    x = torch.randn((8, dimension)).to(device)

    model = nn.Linear(dimension, dimension).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    learning_rate = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    start_time = time.time()

    max_abs_err = torch.ones(1).to(device)
    zero = torch.zeros(1).to(device)
    epoch = 0
    while not torch.isclose(max_abs_err, zero, atol=1e-6):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, x)
        if (epoch + 1) % 500 == 0:
            learning_rate.step()
            print(
                f"\n=== Epoch {epoch + 1} ===\nloss = {loss.item():.8f}\nlearning rate = {optimizer.param_groups[0]['lr']:.8f}"
            )
            abs_err = torch.abs(model(x) - x)
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


learn_identity(8)
learn_identity(32)
learn_identity(256)
learn_identity(1024)
learn_identity(4096)
