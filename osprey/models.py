from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, channel_size: int) -> None:
        """Very small CNN for testing code"""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel_size, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
