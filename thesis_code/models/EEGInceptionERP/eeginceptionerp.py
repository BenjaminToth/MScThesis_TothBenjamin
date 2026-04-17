from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class ConvBNActDrop(nn.Module):
    """Conv2d -> BatchNorm2d -> Activation -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
        bias: bool = False,
        dropout: float = 0.5,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("kernel dimensions must be > 0")
        if groups <= 0:
            raise ValueError("groups must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0")

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InceptionStage1Branch(nn.Module):
    """Temporal conv + depthwise spatial conv for one branch of stage 1."""

    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        temporal_kernel: int,
        depth_multiplier: int = 2,
        dropout: float = 0.5,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_filters <= 0:
            raise ValueError("n_filters must be > 0")
        if temporal_kernel <= 0:
            raise ValueError("temporal_kernel must be > 0")
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be > 0")

        self.temporal = ConvBNActDrop(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            dropout=dropout,
            activation=activation,
        )

        self.spatial = ConvBNActDrop(
            in_channels=n_filters,
            out_channels=n_filters * depth_multiplier,
            kernel_size=(1, n_channels),
            padding=(0, 0),
            groups=n_filters,
            dropout=dropout,
            activation=activation,
        )

        self.out_channels = n_filters * depth_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)  
        x = self.spatial(x)   
        return x


class InceptionStage2Branch(nn.Module):
    """Temporal-only branch for stage 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int,
        dropout: float = 0.5,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be > 0")
        if temporal_kernel <= 0:
            raise ValueError("temporal_kernel must be > 0")

        self.block = ConvBNActDrop(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_kernel, 1),
            padding=(temporal_kernel // 2, 0),
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EEGInceptionERP(nn.Module):
    """EEGInceptionERP-style model for EEG classification.

    This implementation follows the documented architecture:
      1) Inception stage 1:
         - 3 parallel temporal branches
         - per-branch depthwise spatial conv across all channels
         - concat + average pooling
      2) Inception stage 2:
         - 3 parallel temporal branches
         - concat + average pooling
      3) Two short temporal conv blocks with progressive pooling
      4) Flatten + linear classifier

    Accepted input shapes:
        - (B, C, T)
        - (B, T, C)
        - (B, 1, C, T)
        - (B, 1, T, C)

    Internal normalized shape:
        - (B, 1, T, C)

    Notes:
        - The original documented defaults are tied to ~1 s windows at 128 Hz.
        - This implementation keeps the same general structure but allows other
          window lengths and channel counts.
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_samples: Optional[int] = 1000,
        n_classes: int = 6,
        n_filters: int = 8,
        depth_multiplier: int = 2,
        stage1_kernel_sizes: Sequence[int] = (64, 32, 16),
        stage2_kernel_sizes: Sequence[int] = (16, 8, 4),
        stage1_pool_size: int = 4,
        stage2_pool_size: int = 2,
        stage3_kernel_size: int = 8,
        stage4_kernel_size: int = 4,
        stage3_pool_size: int = 2,
        stage4_pool_size: int = 2,
        dropout: float = 0.5,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if n_filters <= 0:
            raise ValueError("n_filters must be > 0")
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be > 0")
        if len(stage1_kernel_sizes) != 3 or len(stage2_kernel_sizes) != 3:
            raise ValueError("stage1_kernel_sizes and stage2_kernel_sizes must each have length 3")
        if any(k <= 0 for k in stage1_kernel_sizes) or any(k <= 0 for k in stage2_kernel_sizes):
            raise ValueError("all inception kernel sizes must be > 0")
        if stage1_pool_size <= 0 or stage2_pool_size <= 0 or stage3_pool_size <= 0 or stage4_pool_size <= 0:
            raise ValueError("all pool sizes must be > 0")
        if stage3_kernel_size <= 0 or stage4_kernel_size <= 0:
            raise ValueError("stage3_kernel_size and stage4_kernel_size must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0")

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.n_classes = int(n_classes)

        self.stage1_branches = nn.ModuleList(
            [
                InceptionStage1Branch(
                    n_channels=n_channels,
                    n_filters=n_filters,
                    temporal_kernel=k,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout,
                    activation=activation,
                )
                for k in stage1_kernel_sizes
            ]
        )
        stage1_out_channels = 3 * n_filters * depth_multiplier
        self.stage1_pool = nn.AvgPool2d(
            kernel_size=(stage1_pool_size, 1),
            stride=(stage1_pool_size, 1),
        )

        self.stage2_branches = nn.ModuleList(
            [
                InceptionStage2Branch(
                    in_channels=stage1_out_channels,
                    out_channels=n_filters,
                    temporal_kernel=k,
                    dropout=dropout,
                    activation=activation,
                )
                for k in stage2_kernel_sizes
            ]
        )
        stage2_out_channels = 3 * n_filters
        self.stage2_pool = nn.AvgPool2d(
            kernel_size=(stage2_pool_size, 1),
            stride=(stage2_pool_size, 1),
        )

        self.stage3 = ConvBNActDrop(
            in_channels=stage2_out_channels,
            out_channels=n_filters,
            kernel_size=(stage3_kernel_size, 1),
            padding=(stage3_kernel_size // 2, 0),
            dropout=dropout,
            activation=activation,
        )
        self.stage3_pool = nn.AvgPool2d(
            kernel_size=(stage3_pool_size, 1),
            stride=(stage3_pool_size, 1),
        )

        self.stage4 = ConvBNActDrop(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=(stage4_kernel_size, 1),
            padding=(stage4_kernel_size // 2, 0),
            dropout=dropout,
            activation=activation,
        )
        self.stage4_pool = nn.AvgPool2d(
            kernel_size=(stage4_pool_size, 1),
            stride=(stage4_pool_size, 1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_filters, n_classes)

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to shape (B, 1, T, C)."""
        if x.ndim == 3:
            if x.shape[1] == self.n_channels:
                x = x.transpose(1, 2).unsqueeze(1)  
            elif x.shape[2] == self.n_channels:
                x = x.unsqueeze(1)  
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
                x = x.transpose(2, 3)  
            elif x.shape[3] == self.n_channels:
                pass  
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

        if x.shape[3] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels after normalization, but got {x.shape[3]}."
            )

        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1
        s1 = [branch(x) for branch in self.stage1_branches]  
        x = torch.cat(s1, dim=1)                             
        x = self.stage1_pool(x)                              

        # Stage 2
        s2 = [branch(x) for branch in self.stage2_branches]  
        x = torch.cat(s2, dim=1)                            
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3(x)
        x = self.stage3_pool(x)

        # Stage 4
        x = self.stage4(x)
        x = self.stage4_pool(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input_shape(x)  
        x = self._forward_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits