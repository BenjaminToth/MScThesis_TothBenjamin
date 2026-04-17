from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet for EEG artifact detection.

    This implementation follows the compact EEGNet design using:
    1) Temporal convolution
    2) Depthwise spatial convolution
    3) Depthwise-separable temporal convolution

    Accepted input shapes:
        - (B, C, T)
        - (B, T, C)
        - (B, 1, C, T)
        - (B, 1, T, C)

    Where:
        B = batch size
        C = number of EEG channels
        T = number of time samples

    Output:
        - logits with shape (B, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_samples: int | None = None,
        n_classes: int = 6,
        dropout: float = 0.5,
        kernel_length: int = 63,
        depthwise_kernel_length: int = 15,
        f1: int = 8,
        d: int = 2,
        f2: int | None = None,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0")
        if kernel_length <= 0 or depthwise_kernel_length <= 0:
            raise ValueError("Convolution kernel lengths must be > 0")
        if kernel_length % 2 == 0:
            raise ValueError("kernel_length should be odd for symmetric padding")
        if depthwise_kernel_length % 2 == 0:
            raise ValueError(
                "depthwise_kernel_length should be odd for symmetric padding"
            )
        if f1 <= 0 or d <= 0:
            raise ValueError("f1 and d must be > 0")

        if f2 is None:
            f2 = f1 * d
        if f2 <= 0:
            raise ValueError("f2 must be > 0")

        self.n_channels = n_channels
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                in_channels=f1,
                out_channels=f1 * d,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f1 * d,
                kernel_size=(1, depthwise_kernel_length),
                stride=(1, 1),
                padding=(0, depthwise_kernel_length // 2),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=f1 * d,
                out_channels=f2,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(f2, n_classes)

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to shape (B, 1, C, T)."""
        if x.ndim == 3:
            if x.shape[1] == self.n_channels:
                x = x.unsqueeze(1) 
            elif x.shape[2] == self.n_channels:
                x = x.transpose(1, 2).unsqueeze(1)  
            else:
                raise ValueError(
                    "3D input must have shape (B, C, T) or (B, T, C). "
                    f"Got shape {tuple(x.shape)} with expected n_channels={self.n_channels}."
                )

        elif x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    "4D input must have shape (B, 1, C, T) or (B, 1, T, C). "
                    f"Got shape {tuple(x.shape)}."
                )

            if x.shape[2] == self.n_channels:
                pass  
            elif x.shape[3] == self.n_channels:
                x = x.transpose(2, 3)  
            else:
                raise ValueError(
                    "4D input must have channels in dim=2 or dim=3. "
                    f"Got shape {tuple(x.shape)} with expected n_channels={self.n_channels}."
                )
        else:
            raise ValueError(
                "Input must have shape (B, C, T), (B, T, C), (B, 1, C, T), or (B, 1, T, C). "
                f"Got {x.ndim} dimensions."
            )

        if x.shape[2] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels after normalization, but got {x.shape[2]}."
            )

        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self._normalize_input_shape(x)
        x = self._forward_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits