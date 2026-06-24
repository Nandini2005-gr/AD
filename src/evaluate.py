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
from torch.utils.data import DataLoader

from dataset_class import AlzheimerDataset
from models.alzheimer_model import Alzheimer3DCNN

test_dataset = AlzheimerDataset(
    mode="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

model = Alzheimer3DCNN()

model.load_state_dict(
    torch.load(
        "checkpoints/alzheimer_model.pth"
    )
)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for volumes, labels in test_loader:

        volumes = volumes.unsqueeze(1)

        outputs = model(volumes)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

accuracy = 100 * correct / total

print()
print("Correct:", correct)
print("Total:", total)
print("Accuracy:", round(accuracy, 2), "%")