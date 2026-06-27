import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_class import AlzheimerDataset
from models.alzheimer_model import Alzheimer3DCNN

train_dataset = AlzheimerDataset(mode="train")
test_dataset = AlzheimerDataset(mode="test")

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)

model = Alzheimer3DCNN()

weights = torch.tensor(
    [0.3264, 1.4913, 4.1190, 43.25],
    dtype=torch.float32
)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for volumes, labels in train_loader:

        volumes = volumes.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(volumes)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}  Loss = {running_loss:.4f}"
    )

print("Training Finished")

torch.save(
    model.state_dict(),
    "checkpoints/alzheimer_model.pth"
)

print("Model Saved")
