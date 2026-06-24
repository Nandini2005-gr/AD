import torch
import torch.nn as nn

class Alzheimer3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2),

            nn.Conv3d(
                8,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                61440,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                4
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x