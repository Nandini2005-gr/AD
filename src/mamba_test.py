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

from models.vision_mamba import VisionMamba3D

model = VisionMamba3D()

x = torch.randn(
    2,
    1,
    61,
    64,
    64
)

y = model(x)

print("Output Shape:", y.shape)