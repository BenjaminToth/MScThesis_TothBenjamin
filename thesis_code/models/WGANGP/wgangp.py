from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGGenerator(nn.Module):
    """Generator for EEG windows.

    Input:
      - latent vector z with shape (B, latent_dim)

    Output:
      - synthetic EEG with shape (B, n_channels, n_samples)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_channels: int = 17,
        n_samples: int = 2000,
        base_channels: int = 256,
        base_length: int | None = None,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if base_length is not None and base_length <= 0:
            raise ValueError("base_length must be > 0 when provided")

        output_activation = output_activation.lower().strip()
        if output_activation not in {"tanh", "linear"}:
            raise ValueError("output_activation must be one of {'tanh', 'linear'}")

        self.latent_dim = int(latent_dim)
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.base_channels = int(base_channels)
        self.base_length = int(base_length) if base_length is not None else max(8, math.ceil(n_samples / 16))
        self.output_activation = output_activation

        self.proj = nn.Linear(self.latent_dim, self.base_channels * self.base_length)

        self.block1 = self._up_block(self.base_channels, self.base_channels)
        self.block2 = self._up_block(self.base_channels, self.base_channels // 2)
        self.block3 = self._up_block(self.base_channels // 2, self.base_channels // 4)
        self.block4 = self._up_block(self.base_channels // 4, self.base_channels // 8)

        self.to_signal = nn.Conv1d(self.base_channels // 8, self.n_channels, kernel_size=7, padding=3)

    @staticmethod
    def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
        if out_channels <= 0:
            raise ValueError("out_channels became <= 0; increase base_channels")
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z shape (B, {self.latent_dim}), got {tuple(z.shape)}."
            )

        x = self.proj(z)
        x = x.view(z.shape[0], self.base_channels, self.base_length)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.to_signal(x)
        x = F.interpolate(x, size=self.n_samples, mode="linear", align_corners=False)

        if self.output_activation == "tanh":
            x = torch.tanh(x)
        return x


class Critic(nn.Module):
    """Wasserstein critic for EEG windows.

    Input:
      - EEG tensor with shape (B, n_channels, n_samples)

    Output:
      - critic score with shape (B, 1)
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_samples: int = 2000,
        channels: Sequence[int] = (64, 128, 256, 512),
        pool_length: int = 16,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if len(channels) == 0:
            raise ValueError("channels must contain at least one value")
        if any(c <= 0 for c in channels):
            raise ValueError("all channels values must be > 0")
        if pool_length <= 0:
            raise ValueError("pool_length must be > 0")

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.channels = tuple(int(c) for c in channels)
        self.pool_length = int(pool_length)

        blocks = []
        in_ch = self.n_channels
        for out_ch in self.channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2, bias=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(self.pool_length)
        self.head = nn.Linear(self.channels[-1] * self.pool_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}.")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {x.shape[1]}."
            )

        h = self.features(x)
        h = self.pool(h)
        h = torch.flatten(h, start_dim=1)
        score = self.head(h)
        return score


class WGANGP(nn.Module):
    """Reusable WGAN-GP model bundle for per-class training.

    The same architecture is intended to be instantiated, trained, and saved
    separately for each class.

    Accepted critic input shapes:
      - (B, C, T)
      - (B, T, C)
      - (B, 1, C, T)
      - (B, 1, T, C)
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_samples: int = 2000,
        n_classes: int = 6,
        latent_dim: int = 128,
        gen_base_channels: int = 256,
        gen_base_length: int | None = None,
        critic_channels: Sequence[int] = (64, 128, 256, 512),
        critic_pool_length: int = 16,
        lambda_gp: float = 10.0,
        class_name: str | None = None,
        class_index: int | None = None,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if lambda_gp <= 0.0:
            raise ValueError("lambda_gp must be > 0")
        if class_index is not None and class_index < 0:
            raise ValueError("class_index must be >= 0 when provided")

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.n_classes = int(n_classes)
        self.latent_dim = int(latent_dim)
        self.lambda_gp = float(lambda_gp)

        self.class_name = class_name
        self.class_index = int(class_index) if class_index is not None else None

        self.generator = EEGGenerator(
            latent_dim=self.latent_dim,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            base_channels=gen_base_channels,
            base_length=gen_base_length,
            output_activation=output_activation,
        )
        self.critic = Critic(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            channels=critic_channels,
            pool_length=critic_pool_length,
        )

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input into (B, C, T) for critic scoring."""
        if x.ndim == 3:
            if x.shape[1] == self.n_channels:
                pass
            elif x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)
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
            x = x[:, 0, ...]
            if x.shape[1] == self.n_channels:
                pass
            elif x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)
            else:
                raise ValueError(
                    "4D input must place channels in dim=2 or dim=3. "
                    f"Got {tuple(x.shape)} with n_channels={self.n_channels}."
                )
        else:
            raise ValueError(
                "Input must have shape (B, C, T), (B, T, C), (B, 1, C, T), or (B, 1, T, C)."
            )

        if x.shape[2] != self.n_samples:
            x = F.interpolate(x, size=self.n_samples, mode="linear", align_corners=False)
        return x

    def sample_noise(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim, device=device)

    def generate(
        self,
        z: torch.Tensor | None = None,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if z is None:
            if batch_size is None:
                raise ValueError("Provide either z or batch_size")
            z = self.sample_noise(batch_size=batch_size, device=device)
        return self.generator(z)

    def critic_score(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input_shape(x)
        return self.critic(x)

    def gradient_penalty(
        self,
        real_x: torch.Tensor,
        fake_x: torch.Tensor,
        lambda_gp: float | None = None,
    ) -> torch.Tensor:
        real_x = self._normalize_input_shape(real_x)
        fake_x = self._normalize_input_shape(fake_x)

        if real_x.shape != fake_x.shape:
            raise ValueError(
                f"real_x and fake_x must have identical shapes, got {tuple(real_x.shape)} and {tuple(fake_x.shape)}"
            )

        batch_size = real_x.shape[0]
        eps = torch.rand(batch_size, 1, 1, device=real_x.device, dtype=real_x.dtype)
        x_hat = eps * real_x + (1.0 - eps) * fake_x
        x_hat.requires_grad_(True)

        score_hat = self.critic(x_hat)
        grad_outputs = torch.ones_like(score_hat)
        grads = torch.autograd.grad(
            outputs=score_hat,
            inputs=x_hat,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grads = grads.view(batch_size, -1)
        gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
        gp_lambda = self.lambda_gp if lambda_gp is None else float(lambda_gp)
        return gp * gp_lambda

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate synthetic EEG from latent vectors."""
        return self.generator(z)
