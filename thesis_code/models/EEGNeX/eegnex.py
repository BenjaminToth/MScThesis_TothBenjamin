from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithMaxNorm(nn.Conv2d):
    """Conv2d with max-norm constraint applied on every forward pass.

    This mirrors the common EEG CNN practice used for the spatial conv in EEGNeX.
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if max_norm <= 0:
            raise ValueError("max_norm must be > 0")
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w = self.weight.view(self.out_channels, -1)
            norms = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            desired = torch.clamp(norms, max=self.max_norm)
            self.weight.mul_((desired / norms).view(self.out_channels, 1, 1, 1))
        return super().forward(x)


class LinearWithMaxNorm(nn.Linear):
    """Linear with max-norm constraint applied on every forward pass."""

    def __init__(self, *args, max_norm: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if max_norm <= 0:
            raise ValueError("max_norm must be > 0")
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w = self.weight
            norms = w.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            desired = torch.clamp(norms, max=self.max_norm)
            self.weight.mul_(desired / norms)
        return super().forward(x)


class EEGNeX(nn.Module):
    """EEGNeX-style model for EEG classification.

    Canonical paper-style defaults correspond to EEGNeX-8,32:
      - filter_1 = 8
      - filter_2 = 32
      - depth_multiplier = 2
      - kernel_block_1_2 = 32
      - kernel_block_4 = 16
      - kernel_block_5 = 16
      - dilation_block_4 = 2
      - dilation_block_5 = 4
      - avg_pool_block4 = 4
      - avg_pool_block5 = 8
      - dropout = 0.5

    Accepted input shapes:
      - (B, C, T)
      - (B, T, C)
      - (B, 1, C, T)
      - (B, 1, T, C)

    Output:
      - logits of shape (B, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_classes: int = 6,
        n_samples: Optional[int] = None,
        activation: type[nn.Module] = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        filter_2: int = 32,
        dropout: float = 0.5,
        kernel_block_1_2: int = 32,
        kernel_block_4: int = 16,
        dilation_block_4: int = 2,
        avg_pool_block4: int = 4,
        kernel_block_5: int = 16,
        dilation_block_5: int = 4,
        avg_pool_block5: int = 8,
        max_norm_conv: float = 1.0,
        max_norm_linear: float = 0.25,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be > 0")
        if filter_1 <= 0 or filter_2 <= 0:
            raise ValueError("filter_1 and filter_2 must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0")
        for name, value in {
            "kernel_block_1_2": kernel_block_1_2,
            "kernel_block_4": kernel_block_4,
            "kernel_block_5": kernel_block_5,
            "dilation_block_4": dilation_block_4,
            "dilation_block_5": dilation_block_5,
            "avg_pool_block4": avg_pool_block4,
            "avg_pool_block5": avg_pool_block5,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples) if n_samples is not None else None

        self.filter_1 = int(filter_1)
        self.filter_2 = int(filter_2)
        self.filter_3 = self.filter_2 * int(depth_multiplier)

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=(1, kernel_block_1_2),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(self.filter_1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=(1, kernel_block_1_2),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(self.filter_2),
        )

        self.block_3 = nn.Sequential(
            Conv2dWithMaxNorm(
                in_channels=self.filter_2,
                out_channels=self.filter_3,
                kernel_size=(self.n_channels, 1),
                groups=self.filter_2,
                bias=False,
                max_norm=max_norm_conv,
            ),
            nn.BatchNorm2d(self.filter_3),
            activation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block4), padding=(0, 1)),
            nn.Dropout(p=dropout),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_3,
                out_channels=self.filter_2,
                kernel_size=(1, kernel_block_4),
                dilation=(1, dilation_block_4),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(self.filter_2),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=(1, kernel_block_5),
                dilation=(1, dilation_block_5),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(self.filter_1),
            activation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_block5), padding=(0, 1)),
            nn.Dropout(p=dropout),
        )

        self.classifier: Optional[LinearWithMaxNorm]
        if self.n_samples is not None:
            in_features = self._calc_in_features(self.n_samples, avg_pool_block4, avg_pool_block5)
            self.classifier = LinearWithMaxNorm(
                in_features=in_features,
                out_features=self.n_classes,
                max_norm=max_norm_linear,
            )
        else:
            self.classifier = None
            self._max_norm_linear = float(max_norm_linear)

    @staticmethod
    def _pool_out_length(length: int, kernel: int, pad: int = 1, stride: Optional[int] = None) -> int:
        if stride is None:
            stride = kernel
        return math.floor((length + 2 * pad - kernel) / stride) + 1

    def _calc_in_features(self, n_times: int, pool4: int, pool5: int) -> int:
        t_after_pool4 = self._pool_out_length(n_times, kernel=pool4, pad=1, stride=pool4)
        t_after_pool5 = self._pool_out_length(t_after_pool4, kernel=pool5, pad=1, stride=pool5)
        return self.filter_1 * t_after_pool5

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to (B, 1, C, T)."""
        if x.ndim == 3:
            if x.shape[1] == self.n_channels:
                x = x.unsqueeze(1)  
            elif x.shape[2] == self.n_channels:
                x = x.transpose(1, 2).unsqueeze(1)  
            else:
                raise ValueError(
                    "3D input must have shape (B, C, T) or (B, T, C). "
                    f"Got {tuple(x.shape)} with n_channels={self.n_channels}."
                )
        elif x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    "4D input must have shape (B, 1, C, T) or (B, 1, T, C). "
                    f"Got {tuple(x.shape)}."
                )
            if x.shape[2] == self.n_channels:
                pass
            elif x.shape[3] == self.n_channels:
                x = x.transpose(2, 3)
            else:
                raise ValueError(
                    "4D input must place channels in dim 2 or 3. "
                    f"Got {tuple(x.shape)} with n_channels={self.n_channels}."
                )
        else:
            raise ValueError(
                "Input must have shape (B, C, T), (B, T, C), (B, 1, C, T), or (B, 1, T, C). "
                f"Got {x.ndim} dimensions."
            )

        if x.shape[2] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels after normalization, got {x.shape[2]}."
            )
        return x

    def _ensure_classifier(self, x: torch.Tensor) -> None:
        if self.classifier is None:
            in_features = x.shape[1]
            self.classifier = LinearWithMaxNorm(
                in_features=in_features,
                out_features=self.n_classes,
                max_norm=self._max_norm_linear,
            ).to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input_shape(x)

        x = self.block_1(x)  
        x = self.block_2(x) 
        x = self.block_3(x)  
        x = self.block_4(x)  
        x = self.block_5(x)  

        x = torch.flatten(x, start_dim=1)  
        self._ensure_classifier(x)
        assert self.classifier is not None
        logits = self.classifier(x)
        return logits