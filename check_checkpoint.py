import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset_class import AlzheimerDataset
from models.vision_mamba import VisionMamba3D

model = VisionMamba3D()
model.load_state_dict(torch.load("../checkpoints/vision_mamba_best.pth", map_location="cpu"))
model.eval()

test_dataset = AlzheimerDataset(mode="test")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for volumes, labels in test_loader:
        volumes = volumes.unsqueeze(1)
        outputs = model(volumes)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print("Accuracy:", acc)
print(classification_report(
    all_labels, all_preds,
    target_names=["Non Demented", "Very Mild", "Mild", "Moderate"],
    zero_division=0
))