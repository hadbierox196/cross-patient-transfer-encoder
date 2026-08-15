"""Stimulus encoder architecture: image -> per-electrode stimulation amplitudes."""

import torch
import torch.nn as nn

from . import config


class StimulusEncoder(nn.Module):
    """Compact CNN mapping a grayscale image to a stimulation amplitude vector.

    Architecture: two conv blocks (16, 32 channels) -> adaptive avg pool (4x4)
    -> two FC layers (128 hidden units) -> sigmoid output bounded to [0, 1].
    """

    def __init__(self, n_electrodes: int, img_size: int = config.IMG_SIZE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, n_electrodes), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
