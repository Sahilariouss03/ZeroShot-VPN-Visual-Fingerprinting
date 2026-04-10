from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class FlowPicEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, in_channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.input_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.projection = nn.Sequential(
            nn.Linear(256 + (in_channels * 8 * 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        pooled = self.pool(features)
        pooled_features = torch.flatten(pooled, start_dim=1)
        pooled_inputs = torch.flatten(self.input_pool(inputs), start_dim=1)
        embeddings = self.projection(torch.cat([pooled_features, pooled_inputs], dim=1))
        return F.normalize(embeddings, p=2, dim=1)
