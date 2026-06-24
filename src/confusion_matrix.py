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
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from dataset_class import AlzheimerDataset
from models.vision_mamba import VisionMamba3D


test_dataset = AlzheimerDataset(
    mode="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

model = VisionMamba3D()

model.load_state_dict(
    torch.load(
        "checkpoints/vision_mamba.pth"
    )
)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for volumes, labels in test_loader:

        volumes = volumes.unsqueeze(1)

        outputs = model(volumes)

        _, predicted = torch.max(
            outputs,
            1
        )

        y_true.append(
            labels.item()
        )

        y_pred.append(
            predicted.item()
        )

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        digits=4
    )
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Vision Mamba Confusion Matrix"
)

plt.savefig(
    "confusion_matrix.png"
)

plt.show()

print("\nConfusion Matrix Saved")