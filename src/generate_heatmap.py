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

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset_class import AlzheimerDataset
from models.vision_mamba import VisionMamba3D


# Load trained model
model = VisionMamba3D()

model.load_state_dict(
    torch.load(
        "checkpoints/vision_mamba.pth"
    )
)

model.eval()

feature_maps = []


def hook_fn(module, input, output):
    feature_maps.append(output)


# Register hook
hook = model.mamba3.register_forward_hook(
    hook_fn
)

# Load test dataset
dataset = AlzheimerDataset(
    mode="train"
)

class_names = {
    0: "Non Demented",
    1: "Very Mild Dementia",
    2: "Mild Dementia",
    3: "Moderate Dementia"
}

positions = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 0),
    3: (1, 1)
}

used_classes = set()

# Create single page with 4 heatmaps
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

print("Generating heatmaps...")

for idx in range(len(dataset)):

    volume, label = dataset[idx]

    if label in used_classes:
        continue

    used_classes.add(label)

    feature_maps.clear()

    input_volume = volume.unsqueeze(0)
    input_volume = input_volume.unsqueeze(0)

    with torch.no_grad():
        output = model(input_volume)

    feature = feature_maps[0][0]

    heatmap = torch.mean(
        feature,
        dim=0
    )

    heatmap = heatmap.cpu().numpy()

    middle_slice = heatmap[
        heatmap.shape[0] // 2
    ]

    middle_slice = cv2.resize(
        middle_slice,
        (64, 64)
    )

    middle_slice = (
        middle_slice - middle_slice.min()
    ) / (
        middle_slice.max() - middle_slice.min() + 1e-8
    )

    row, col = positions[label]

    ax = axes[row][col]

    img = ax.imshow(
        middle_slice,
        cmap="jet"
    )

    ax.set_title(
        class_names[label],
        fontsize=12,
        fontweight="bold"
    )

    ax.axis("off")

    fig.colorbar(
        img,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    print(f"Added: {class_names[label]}")

    if len(used_classes) == 4:
        break

plt.suptitle(
    "Vision Mamba Heatmaps for Alzheimer's MRI Classes",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "all_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: all_heatmaps.png")

hook.remove()
