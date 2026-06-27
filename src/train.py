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

dataset = AlzheimerDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = Alzheimer3DCNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Starting Training...")

for epoch in range(1):

    running_loss = 0

    for volumes, labels in loader:

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
            "Loss:",
            round(loss.item(), 4)
        )

        break

    print(
        "Epoch Loss:",
        round(running_loss, 4)
    )

print("Training Test Complete")
