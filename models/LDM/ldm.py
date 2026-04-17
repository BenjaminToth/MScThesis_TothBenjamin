from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t shape (B,), got {tuple(t.shape)}")

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, time_emb_dim: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channels must be > 0")
        if time_emb_dim <= 0:
            raise ValueError("time_emb_dim must be > 0")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        g1 = min(8, self.in_channels)
        while self.in_channels % g1 != 0:
            g1 -= 1
        g2 = min(8, self.out_channels)
        while self.out_channels % g2 != 0:
            g2 -= 1

        self.norm1 = nn.GroupNorm(g1, self.in_channels)
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g2, self.out_channels)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, 2 * self.out_channels)

        self.skip = (
            nn.Identity() if self.in_channels == self.out_channels else nn.Conv1d(self.in_channels, self.out_channels, 1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}")
        if t_emb.ndim != 2:
            raise ValueError(f"Expected t_emb shape (B, D), got {tuple(t_emb.shape)}")

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.time_proj(F.silu(t_emb))
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale[:, :, None]
        shift = shift[:, :, None]

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class EEGVAE1D(nn.Module):
    """Simple 1D Conv VAE for EEG windows.

    Encodes x:(B,C,T) -> latent z:(B,latent_channels,latent_length) and decodes back to x_hat:(B,C,T).
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_samples: int,
        latent_channels: int = 32,
        base_channels: int = 128,
        downsample_factor: int = 16,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if latent_channels <= 0:
            raise ValueError("latent_channels must be > 0")
        if base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if downsample_factor <= 1 or (downsample_factor & (downsample_factor - 1)) != 0:
            raise ValueError("downsample_factor must be a power of two > 1")

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.downsample_factor = int(downsample_factor)

        n_down = int(round(math.log2(self.downsample_factor)))

        enc_layers: list[nn.Module] = []
        in_ch = self.n_channels
        ch = self.base_channels
        for _ in range(n_down):
            g = min(8, ch)
            while ch % g != 0:
                g -= 1
            enc_layers.extend(
                [
                    nn.Conv1d(in_ch, ch, kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(g, ch),
                    nn.SiLU(),
                ]
            )
            in_ch = ch
            ch = min(ch * 2, 512)
        self.encoder = nn.Sequential(*enc_layers)

        enc_out_ch = in_ch
        self.to_stats = nn.Conv1d(enc_out_ch, 2 * self.latent_channels, kernel_size=1)

        dec_layers: list[nn.Module] = []
        ch = enc_out_ch
        self.from_latent = nn.Conv1d(self.latent_channels, ch, kernel_size=1)

        for _ in range(n_down):
            out_ch = max(ch // 2, self.base_channels)
            g = min(8, out_ch)
            while out_ch % g != 0:
                g -= 1
            dec_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(ch, out_ch, kernel_size=5, padding=2),
                    nn.GroupNorm(g, out_ch),
                    nn.SiLU(),
                ]
            )
            ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)
        self.to_signal = nn.Conv1d(ch, self.n_channels, kernel_size=7, padding=3)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(f"Expected C={self.n_channels}, got {x.shape[1]}")

        h = self.encoder(x)
        stats = self.to_stats(h)
        mu, logvar = stats.chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Expected z shape (B, Cz, Tz), got {tuple(z.shape)}")
        if z.shape[1] != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got {z.shape[1]}")

        h = self.from_latent(z)
        h = self.decoder(h)
        x_hat = self.to_signal(h)
        x_hat = F.interpolate(x_hat, size=self.n_samples, mode="linear", align_corners=False)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


class UNet1D(nn.Module):
    """Small 1D U-Net that predicts noise in latent space."""

    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 4),
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if len(channel_mults) == 0:
            raise ValueError("channel_mults must not be empty")

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(m) for m in channel_mults)
        self.time_emb_dim = int(time_emb_dim)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
        )

        self.in_conv = nn.Conv1d(self.in_channels, self.base_channels, kernel_size=3, padding=1)

        downs: list[nn.Module] = []
        self.downsample_layers: list[nn.Module] = []
        ch = self.base_channels
        self._skip_channels: list[int] = []

        for mult in self.channel_mults:
            out_ch = self.base_channels * mult
            downs.append(ResBlock1D(ch, out_ch, time_emb_dim=self.time_emb_dim))
            self._skip_channels.append(out_ch)
            downs.append(nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1))
            ch = out_ch
        self.down = nn.ModuleList(downs)

        self.mid1 = ResBlock1D(ch, ch, time_emb_dim=self.time_emb_dim)
        self.mid2 = ResBlock1D(ch, ch, time_emb_dim=self.time_emb_dim)

        ups: list[nn.Module] = []
        for mult, skip_ch in reversed(list(zip(self.channel_mults, self._skip_channels))):
            out_ch = self.base_channels * mult
            ups.append(nn.Upsample(scale_factor=2, mode="nearest"))
            ups.append(ResBlock1D(ch + skip_ch, out_ch, time_emb_dim=self.time_emb_dim))
            ch = out_ch
        self.up = nn.ModuleList(ups)

        g = min(8, ch)
        while ch % g != 0:
            g -= 1
        self.out_norm = nn.GroupNorm(g, ch)
        self.out_conv = nn.Conv1d(ch, self.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {x.shape[1]}")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t shape (B,), got {tuple(t.shape)}")

        t_emb = self.time_embed(t)

        h = self.in_conv(x)
        skips: list[torch.Tensor] = []

        i = 0
        for _mult in self.channel_mults:
            h = self.down[i](h, t_emb)
            i += 1
            skips.append(h)
            h = self.down[i](h)
            i += 1

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        j = 0
        for _mult in reversed(self.channel_mults):
            h = self.up[j](h)
            j += 1
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = self.up[j](h, t_emb)
            j += 1

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor


def _cosine_beta_schedule(timesteps: int, *, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion1D(nn.Module):
    """DDPM utilities for latent tensors shaped (B, C, T)."""

    def __init__(self, *, timesteps: int = 1000, beta_schedule: str = "cosine") -> None:
        super().__init__()
        if timesteps <= 1:
            raise ValueError("timesteps must be > 1")
        beta_schedule = str(beta_schedule).strip().lower()
        if beta_schedule not in {"cosine", "linear"}:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")

        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule

        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, self.timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        schedule = DiffusionSchedule(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        )

        for name, tensor in schedule.__dict__.items():
            self.register_buffer(name, tensor)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(-1, t).float()
        while out.ndim < len(x_shape):
            out = out[..., None]
        return out

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_omab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_omab * noise

    def p_mean_variance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps_pred = model(x, t)

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_ab_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        model_mean = sqrt_recip_alpha_t * (x - betas_t * eps_pred / (sqrt_one_minus_ab_t + 1e-8))
        model_var = betas_t
        return model_mean, model_var

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        model_mean, model_var = self.p_mean_variance(model, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float()
        while nonzero_mask.ndim < x.ndim:
            nonzero_mask = nonzero_mask[..., None]
        return model_mean + nonzero_mask * torch.sqrt(model_var) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        *,
        device: torch.device | str | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        device = _default_device(device)
        b, c, tlen = shape
        img = torch.randn((b, c, tlen), device=device)

        total = int(self.timesteps)
        n_steps = int(num_inference_steps) if num_inference_steps is not None else total
        n_steps = max(1, min(n_steps, total))

        if n_steps == total:
            timesteps = list(range(total - 1, -1, -1))
        else:
            idx = torch.linspace(total - 1, 0, n_steps, dtype=torch.long)
            timesteps = [int(v.item()) for v in idx]

        for i in timesteps:
            tt = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, tt)
        return img


class EEGLatentDiffusion(nn.Module):
    """VAE + latent-space diffusion bundle for EEG windows."""

    def __init__(
        self,
        *,
        n_channels: int,
        n_samples: int,
        latent_channels: int = 32,
        vae_base_channels: int = 128,
        vae_downsample_factor: int = 16,
        unet_base_channels: int = 128,
        unet_channel_mults: Sequence[int] = (1, 2, 4),
        diffusion_timesteps: int = 1000,
        diffusion_beta_schedule: str = "cosine",
        latent_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)

        self.vae = EEGVAE1D(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            latent_channels=int(latent_channels),
            base_channels=int(vae_base_channels),
            downsample_factor=int(vae_downsample_factor),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, self.n_channels, self.n_samples), dtype=torch.float32)
            mu, _logvar = self.vae.encode(dummy)
            latent_len = int(mu.shape[-1])

        self.latent_channels = int(latent_channels)
        self.latent_length = int(latent_len)

        self.unet = UNet1D(
            in_channels=self.latent_channels,
            base_channels=int(unet_base_channels),
            channel_mults=tuple(int(x) for x in unet_channel_mults),
        )

        self.diffusion = GaussianDiffusion1D(timesteps=int(diffusion_timesteps), beta_schedule=diffusion_beta_schedule)

        self.register_buffer("latent_scale", torch.tensor(float(latent_scale), dtype=torch.float32))

    def set_latent_scale(self, scale: float) -> None:
        s = float(scale)
        if not math.isfinite(s) or s <= 0:
            raise ValueError(f"latent_scale must be finite and > 0, got {scale}")
        self.latent_scale.fill_(s)

    def scale_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_scale

    def unscale_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z / (self.latent_scale + 1e-12)

    def encode(self, x: torch.Tensor, *, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar) if sample else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int,
        device: torch.device | str | None = None,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        device = _default_device(device)
        self.eval()
        z = self.diffusion.p_sample_loop(
            self.unet,
            (int(batch_size), int(self.latent_channels), int(self.latent_length)),
            device=device,
            num_inference_steps=int(num_inference_steps),
        )
        z = self.unscale_latent(z)
        x = self.decode(z)
        return x


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float,
    recon_loss: str = "l1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if recon_loss == "l1":
        recon = F.l1_loss(x_hat, x)
    elif recon_loss == "mse":
        recon = F.mse_loss(x_hat, x)
    else:
        raise ValueError("recon_loss must be 'l1' or 'mse'")

    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = recon + float(kl_weight) * kl
    return total, recon, kl
