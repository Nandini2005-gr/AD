import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        embed_dim=64
    ):

        super().__init__()

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=4,
            stride=4
        )

    def forward(self, x):

        return self.proj(x)


class MambaBlock(nn.Module):

    def __init__(self, dim=64):

        super().__init__()

        self.norm = nn.BatchNorm3d(dim)

        self.conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        residual = x

        x = self.norm(x)

        x = self.conv(x)

        x = self.relu(x)

        x = x + residual

        return x


class VisionMamba3D(nn.Module):

    def __init__(self):

        super().__init__()

        self.patch_embed = PatchEmbedding3D()

        self.mamba1 = MambaBlock(64)

        self.mamba2 = MambaBlock(64)

        self.mamba3 = MambaBlock(64)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Linear(
            64,
            4
        )

    def forward(self, x):

        x = self.patch_embed(x)

        x = self.mamba1(x)

        x = self.mamba2(x)

        x = self.mamba3(x)

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.classifier(x)

        return x