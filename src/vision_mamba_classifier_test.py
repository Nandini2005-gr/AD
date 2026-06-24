import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

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


hook = model.mamba3.register_forward_hook(
    hook_fn
)

dataset = AlzheimerDataset(
    mode="test"
)

volume, label = dataset[0]

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
    middle_slice.max() - middle_slice.min()
)

plt.imshow(
    middle_slice,
    cmap="jet"
)

plt.colorbar()

plt.title(
    "Vision Mamba Heatmap"
)

plt.savefig(
    "heatmap.png"
)

plt.show()

print("Heatmap Saved")

hook.remove()