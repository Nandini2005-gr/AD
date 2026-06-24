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

from models.alzheimer_model import Alzheimer3DCNN

model = Alzheimer3DCNN()

x = torch.randn(
    2,
    1,
    61,
    64,
    64
)

y = model(x)

print("Output Shape:", y.shape)