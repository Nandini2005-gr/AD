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

from dataset_class import AlzheimerDataset
from models.vision_mamba import VisionMamba3D


# Load Model
model = VisionMamba3D()

model.load_state_dict(
    torch.load(
        "checkpoints/vision_mamba.pth"
    )
)

model.eval()


# Hook Storage
feature_maps = []


def hook_fn(module, input, output):

    feature_maps.append(output)


# Hook Last Mamba Block
hook = model.mamba3.register_forward_hook(
    hook_fn
)


# Load One Patient
dataset = AlzheimerDataset(
    mode="test"
)

volume, label = dataset[0]

volume = volume.unsqueeze(0)
volume = volume.unsqueeze(0)


# Forward Pass
with torch.no_grad():

    output = model(volume)


print("Prediction Output Shape:",
      output.shape)

print("Feature Maps Shape:",
      feature_maps[0].shape)

print("True Label:",
      label)


hook.remove()