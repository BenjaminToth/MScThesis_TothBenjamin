from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convolutional patch embedding used in EEGConformer.

    Input:
        x: (B, 1, C, T)

    Output:
        tokens: (B, N, D)
            N = number of temporal patches
            D = embedding dimension
    """

    def __init__(
        self,
        n_channels: int,
        emb_size: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if emb_size <= 0:
            raise ValueError("emb_size must be > 0")
        if filter_time_length <= 0:
            raise ValueError("filter_time_length must be > 0")
        if pool_time_length <= 0:
            raise ValueError("pool_time_length must be > 0")
        if pool_time_stride <= 0:
            raise ValueError("pool_time_stride must be > 0")
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError("drop_prob must satisfy 0.0 <= drop_prob < 1.0")

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=emb_size,
                kernel_size=(1, filter_time_length),
                stride=(1, 1),
                padding=(0, filter_time_length // 2),
                bias=False,
            ),
            nn.Conv2d(
                in_channels=emb_size,
                out_channels=emb_size,
                kernel_size=(n_channels, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(emb_size),
            activation(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length),
                stride=(1, pool_time_stride),
            ),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          
        x = x.squeeze(2)          
        x = x.transpose(1, 2)     
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        att_drop_prob: float = 0.5,
        proj_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()

        if emb_size <= 0:
            raise ValueError("emb_size must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if emb_size % num_heads != 0:
            raise ValueError(
                f"emb_size ({emb_size}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= att_drop_prob < 1.0):
            raise ValueError("att_drop_prob must satisfy 0.0 <= att_drop_prob < 1.0")
        if not (0.0 <= proj_drop_prob < 1.0):
            raise ValueError("proj_drop_prob must satisfy 0.0 <= proj_drop_prob < 1.0")

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(emb_size, emb_size * 3, bias=True)
        self.attn_drop = nn.Dropout(att_drop_prob)
        self.proj = nn.Linear(emb_size, emb_size)
        self.proj_drop = nn.Dropout(proj_drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, emb = x.shape

        qkv = self.qkv(x)  
        qkv = qkv.reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                 
        out = out.transpose(1, 2).reshape(bsz, n_tokens, emb)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        expansion: int = 4,
        drop_prob: float = 0.5,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        if emb_size <= 0:
            raise ValueError("emb_size must be > 0")
        if expansion <= 0:
            raise ValueError("expansion must be > 0")
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError("drop_prob must satisfy 0.0 <= drop_prob < 1.0")

        hidden = emb_size * expansion
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden),
            activation(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden, emb_size),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        att_drop_prob: float = 0.5,
        ff_drop_prob: float = 0.5,
        ff_expansion: int = 4,
        activation_transfor: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadSelfAttention(
            emb_size=emb_size,
            num_heads=num_heads,
            att_drop_prob=att_drop_prob,
            proj_drop_prob=ff_drop_prob,
        )

        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForwardBlock(
            emb_size=emb_size,
            expansion=ff_expansion,
            drop_prob=ff_drop_prob,
            activation=activation_transfor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class ClassificationHead(nn.Module):
    """Flatten-all-tokens classification head.

    This matches the common EEGConformer implementation style:
    flatten transformer features and classify with an MLP.
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden_features: int = 256,
        drop_prob: float = 0.5,
        return_features: bool = False,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be > 0")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if hidden_features <= 0:
            raise ValueError("hidden_features must be > 0")
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError("drop_prob must satisfy 0.0 <= drop_prob < 1.0")

        self.return_features = return_features

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_features, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.flatten(x, start_dim=1)  
        if self.return_features:
            return feats
        return self.fc(feats)


class EEGConformer(nn.Module):
    """EEGConformer-style model for EEG classification.

    Accepted input shapes:
        - (B, C, T)
        - (B, T, C)
        - (B, 1, C, T)
        - (B, 1, T, C)

    Parameters roughly aligned with the published / Braindecode defaults:
        n_filters_time = 40
        filter_time_length = 25
        pool_time_length = 75
        pool_time_stride = 15
        att_depth = 6
        att_heads = 10
        drop_prob = 0.5
        att_drop_prob = 0.5
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_samples: Optional[int] = 1000,
        n_classes: int = 6,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        drop_prob: float = 0.5,
        att_depth: int = 6,
        att_heads: int = 10,
        att_drop_prob: float = 0.5,
        final_fc_length: int | str = "auto",
        return_features: bool = False,
        activation: type[nn.Module] = nn.ELU,
        activation_transfor: type[nn.Module] = nn.GELU,
        ff_expansion: int = 4,
        cls_hidden_features: int = 256,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if n_filters_time <= 0:
            raise ValueError("n_filters_time must be > 0")
        if att_depth <= 0:
            raise ValueError("att_depth must be > 0")
        if att_heads <= 0:
            raise ValueError("att_heads must be > 0")

        self.n_channels = n_channels
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.n_classes = n_classes
        self.emb_size = n_filters_time
        self.return_features = return_features

        self.patch_embedding = PatchEmbedding(
            n_channels=n_channels,
            emb_size=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob,
            activation=activation,
        )

        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    emb_size=n_filters_time,
                    num_heads=att_heads,
                    att_drop_prob=att_drop_prob,
                    ff_drop_prob=drop_prob,
                    ff_expansion=ff_expansion,
                    activation_transfor=activation_transfor,
                )
                for _ in range(att_depth)
            ]
        )

        self.classification_head: Optional[ClassificationHead] = None
        self._cls_hidden_features = cls_hidden_features
        self._drop_prob = drop_prob

        if final_fc_length != "auto":
            if not isinstance(final_fc_length, int) or final_fc_length <= 0:
                raise ValueError(
                    "final_fc_length must be 'auto' or a positive integer"
                )
            self.classification_head = ClassificationHead(
                in_features=final_fc_length,
                n_classes=n_classes,
                hidden_features=cls_hidden_features,
                drop_prob=drop_prob,
                return_features=return_features,
            )

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
                f"Expected {self.n_channels} channels after normalization, got {x.shape[2]}."
            )

        return x

    def _ensure_classification_head(self, x: torch.Tensor) -> None:
        if self.classification_head is None:
            in_features = x.shape[1] * x.shape[2]  
            self.classification_head = ClassificationHead(
                in_features=in_features,
                n_classes=self.n_classes,
                hidden_features=self._cls_hidden_features,
                drop_prob=self._drop_prob,
                return_features=self.return_features,
            ).to(device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input_shape(x)   
        x = self.patch_embedding(x)          
        x = self.transformer(x)              
        self._ensure_classification_head(x)
        assert self.classification_head is not None
        x = self.classification_head(x)     
        return x