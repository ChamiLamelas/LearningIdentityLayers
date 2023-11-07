#!/usr/bin/env python3.8

import torch.nn as nn
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn.functional as F

EPOCHS = 4000
SEED = 42
DIMENSION = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

x = torch.randn((8, DIMENSION))

model = nn.Linear(DIMENSION, DIMENSION)

optimizer = optim.Adam(model.parameters(), lr=0.1)
learning_rate = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, x)
    if (epoch + 1) % 500 == 0:
        learning_rate.step()
        print(
            f"\n=== Epoch {epoch + 1}/{EPOCHS} ===\nloss = {loss.item():.6f}\nlearning rate = {optimizer.param_groups[0]['lr']}"
        )
        abs_err = torch.abs(model(x) - x)
        print(
            f"Absolute performance:\nmax abs err = {torch.max(abs_err):.6f}\nmean abs err = {torch.mean(abs_err):.6f}"
        )
    loss.backward()
    optimizer.step()

print("\nLearned parameters:", list(model.parameters()))
