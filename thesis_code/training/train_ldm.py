from __future__ import annotations

import argparse
import json
import logging
import math
import random
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from dataloader.eeg_dataset import ARTIFACT_CLASSES, EEGWindowDataset, build_file_list
from dataloader.wgangp_dataset import SingleClassEEGWindowDataset
from models.LDM import EEGLatentDiffusion
from models.LDM.ldm import vae_loss


LOGGER = logging.getLogger("train_ldm")


@dataclass(frozen=True)
class RunSettings:
    # Data
    split_root: str
    data_subdir: str
    batch_size: int
    num_workers: int
    sfreq: float
    window_sec: float
    stride_sec: float
    min_overlap_sec: float
    min_overlap_frac_artifact: float
    normalize: bool

    # VAE
    latent_channels: int
    vae_base_channels: int
    vae_downsample_factor: int
    vae_epochs: int
    vae_kl_weight: float
    vae_recon_loss: str
    vae_lr: float

    # Diffusion
    diffusion_timesteps: int
    diffusion_beta_schedule: str
    unet_base_channels: int
    diffusion_epochs: int
    diffusion_lr: float
    num_inference_steps: int

    # Latent scaling
    latent_scale: float
    latent_scale_estimate_batches: int

    # Training
    train_samples_per_epoch: int
    grad_clip: float
    device: str
    gpu: int
    seed: int

    # Class
    class_name: str
    class_index: int
    exclusive: bool

    # Output
    run_name: str


def _setup_logging(*, log_file: Path | None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_seed(seed_arg: int) -> int:
    seed_int = int(seed_arg)
    if seed_int >= 0:
        return seed_int
    return int(secrets.randbits(32))


def _resolve_device(args_device: str, *, gpu_index: int) -> torch.device:
    d = str(args_device).strip().lower()

    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"

    if d == "cpu":
        return torch.device("cpu")

    if d.startswith("cuda"):
        if d == "cuda":
            d = f"cuda:{int(gpu_index)}"

        dev = torch.device(d)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

        n = torch.cuda.device_count()
        if dev.index is None:
            raise RuntimeError(f"Invalid CUDA device: {d}")
        if dev.index < 0 or dev.index >= n:
            raise RuntimeError(f"Requested GPU index {dev.index} but only {n} CUDA device(s) are available")
        return dev

    return torch.device(d)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_output_structure(saved_root: Path) -> None:
    for cn in ARTIFACT_CLASSES:
        _ensure_dir(saved_root / cn)
    _ensure_dir(saved_root / "logs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-class Latent Diffusion (VAE + DDPM) generators for artifact windows.")

    # Data
    p.add_argument("--split-root", type=str, default="data/01_tcp_ar_split")
    p.add_argument("--data-subdir", type=str, default="01_tcp_ar")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--sfreq", type=float, default=250.0)
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--stride-sec", type=float, default=0.5)
    p.add_argument("--min-overlap-sec", type=float, default=0.25)
    p.add_argument("--min-overlap-frac-artifact", type=float, default=0.25)
    p.add_argument("--no-normalize", action="store_true")

    p.add_argument(
        "--class-name",
        type=str,
        default="",
        help=f"Artifact class to train ({', '.join(ARTIFACT_CLASSES)}).",
    )
    p.add_argument(
        "--all-classes",
        action="store_true",
        help="Train all classes sequentially (writes one model per class).",
    )
    p.add_argument(
        "--exclusive",
        action="store_true",
        help="Only use windows where this class is the only positive label.",
    )

    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--vae-base-channels", type=int, default=128)
    p.add_argument("--vae-downsample-factor", type=int, default=16)
    p.add_argument("--vae-epochs", type=int, default=10)
    p.add_argument("--vae-kl-weight", type=float, default=1e-4)
    p.add_argument("--vae-recon-loss", choices=["l1", "mse"], default="l1")
    p.add_argument("--vae-lr", type=float, default=1e-3)

    p.add_argument("--diffusion-timesteps", type=int, default=1000)
    p.add_argument("--diffusion-beta-schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--unet-base-channels", type=int, default=128)
    p.add_argument("--diffusion-epochs", type=int, default=20)
    p.add_argument("--diffusion-lr", type=float, default=2e-4)
    p.add_argument("--num-inference-steps", type=int, default=50)

    p.add_argument(
        "--latent-scale",
        type=float,
        default=0.0,
        help=(
            "Latent scaling multiplier applied before diffusion. "
            "0 = auto-estimate to make latent std ~ 1, <0 = disable (use 1.0), >0 = fixed scale."
        ),
    )
    p.add_argument(
        "--latent-scale-estimate-batches",
        type=int,
        default=50,
        help="Batches used to estimate latent std when --latent-scale=0.",
    )

    p.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=50000,
        help="Approximate number of windows to draw per epoch (0 = no cap).",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (>=0 uses the provided seed; -1 uses a fresh random seed).",
    )

    p.add_argument("--run-name", type=str, default="")

    return p.parse_args()


def _write_outputs(out_dir: Path, *, model: EEGLatentDiffusion, settings: RunSettings, history: Dict[str, List[float]]) -> None:
    _ensure_dir(out_dir)
    torch.save(model.vae.state_dict(), out_dir / "vae_last.pt")
    torch.save(model.unet.state_dict(), out_dir / "unet_last.pt")
    (out_dir / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    (out_dir / "history.json").write_text(json.dumps({"settings": asdict(settings), "history": history}, indent=2))


def _train_one_class(
    *,
    base_ds: EEGWindowDataset,
    class_name: str,
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    run_name: str,
) -> None:
    if class_name not in ARTIFACT_CLASSES:
        raise ValueError(f"Unknown class_name '{class_name}'. Expected one of: {ARTIFACT_CLASSES}")

    class_index = int(ARTIFACT_CLASSES.index(class_name))

    ds = SingleClassEEGWindowDataset(
        base_ds,
        class_index=class_index,
        class_name=class_name,
        include_other_labels=(not bool(args.exclusive)),
    )

    sample_x = ds[0]
    n_channels = int(sample_x.shape[0])
    n_samples = int(sample_x.shape[1])

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        drop_last=True,
    )

    model = EEGLatentDiffusion(
        n_channels=n_channels,
        n_samples=n_samples,
        latent_channels=int(args.latent_channels),
        vae_base_channels=int(args.vae_base_channels),
        vae_downsample_factor=int(args.vae_downsample_factor),
        unet_base_channels=int(args.unet_base_channels),
        diffusion_timesteps=int(args.diffusion_timesteps),
        diffusion_beta_schedule=str(args.diffusion_beta_schedule),
    ).to(device)

    vae_opt = torch.optim.Adam(model.vae.parameters(), lr=float(args.vae_lr))
    unet_opt = torch.optim.Adam(model.unet.parameters(), lr=float(args.diffusion_lr))

    settings = RunSettings(
        split_root=str(args.split_root),
        data_subdir=str(args.data_subdir),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        sfreq=float(args.sfreq),
        window_sec=float(args.window_sec),
        stride_sec=float(args.stride_sec),
        min_overlap_sec=float(args.min_overlap_sec),
        min_overlap_frac_artifact=float(args.min_overlap_frac_artifact),
        normalize=(not bool(args.no_normalize)),
        latent_channels=int(args.latent_channels),
        vae_base_channels=int(args.vae_base_channels),
        vae_downsample_factor=int(args.vae_downsample_factor),
        vae_epochs=int(args.vae_epochs),
        vae_kl_weight=float(args.vae_kl_weight),
        vae_recon_loss=str(args.vae_recon_loss),
        vae_lr=float(args.vae_lr),
        diffusion_timesteps=int(args.diffusion_timesteps),
        diffusion_beta_schedule=str(args.diffusion_beta_schedule),
        unet_base_channels=int(args.unet_base_channels),
        diffusion_epochs=int(args.diffusion_epochs),
        diffusion_lr=float(args.diffusion_lr),
        num_inference_steps=int(args.num_inference_steps),
        latent_scale=1.0,
        latent_scale_estimate_batches=int(args.latent_scale_estimate_batches),
        train_samples_per_epoch=int(args.train_samples_per_epoch),
        grad_clip=float(args.grad_clip),
        device=str(device),
        gpu=int(args.gpu),
        seed=int(args.seed),
        class_name=str(class_name),
        class_index=int(class_index),
        exclusive=bool(args.exclusive),
        run_name=str(run_name),
    )

    history: Dict[str, List[float]] = {
        "vae_loss": [],
        "vae_recon": [],
        "vae_kl": [],
        "diffusion_loss": [],
    }

    max_batches = None
    if int(args.train_samples_per_epoch) > 0:
        max_batches = int(math.ceil(int(args.train_samples_per_epoch) / max(int(args.batch_size), 1)))

    _ensure_dir(out_dir)

    LOGGER.info(
        "Class '%s': total_windows=%d selected_windows=%d | n_channels=%d n_samples=%d",
        class_name,
        int(ds.stats.total_windows),
        int(ds.stats.selected_windows),
        n_channels,
        n_samples,
    )

    for epoch in range(1, int(args.vae_epochs) + 1):
        model.train()

        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n_batches = 0

        for batch_idx, real_x in enumerate(loader, start=1):
            real_x = real_x.to(device)

            x_hat, mu, logvar = model.vae(real_x)
            loss, recon, kl = vae_loss(
                real_x,
                x_hat,
                mu,
                logvar,
                kl_weight=float(args.vae_kl_weight),
                recon_loss=str(args.vae_recon_loss),
            )

            vae_opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.vae.parameters(), max_norm=float(args.grad_clip))
            vae_opt.step()

            loss_sum += float(loss.item())
            recon_sum += float(recon.item())
            kl_sum += float(kl.item())
            n_batches += 1

            if max_batches is not None and batch_idx >= int(max_batches):
                break

        mean_loss = loss_sum / max(n_batches, 1)
        mean_recon = recon_sum / max(n_batches, 1)
        mean_kl = kl_sum / max(n_batches, 1)

        history["vae_loss"].append(mean_loss)
        history["vae_recon"].append(mean_recon)
        history["vae_kl"].append(mean_kl)

        LOGGER.info("[%s][VAE %d/%d] loss=%.6f recon=%.6f kl=%.6f", class_name, epoch, int(args.vae_epochs), mean_loss, mean_recon, mean_kl)
        (out_dir / "history.json").write_text(json.dumps({"settings": asdict(settings), "history": history}, indent=2))
        torch.save(model.vae.state_dict(), out_dir / "vae_last.pt")

    for p in model.vae.parameters():
        p.requires_grad_(False)
    model.vae.eval()

    latent_scale = float(args.latent_scale)
    if latent_scale < 0:
        latent_scale = 1.0
    elif latent_scale == 0.0:
        n_est = max(1, int(args.latent_scale_estimate_batches))
        n_seen = 0
        mean = 0.0
        mean2 = 0.0

        with torch.no_grad():
            for real_x in loader:
                real_x = real_x.to(device)
                _z, mu, _logvar = model.encode(real_x, sample=False)
                mu_f = mu.detach().float()
                mean += float(mu_f.mean().item())
                mean2 += float((mu_f * mu_f).mean().item())
                n_seen += 1
                if n_seen >= n_est:
                    break

            mean /= max(n_seen, 1)
            mean2 /= max(n_seen, 1)
            var = max(mean2 - (mean * mean), 1e-12)
            est_std = float(math.sqrt(var))
        latent_scale = 1.0 / (est_std + 1e-8)

    latent_scale = float(max(1e-3, min(1e3, latent_scale)))
    model.set_latent_scale(latent_scale)
    settings = RunSettings(**{**asdict(settings), "latent_scale": latent_scale})
    LOGGER.info("[%s] Latent scale: %.6f", class_name, latent_scale)

    for epoch in range(1, int(args.diffusion_epochs) + 1):
        model.train()

        loss_sum = 0.0
        n_batches = 0

        for batch_idx, real_x in enumerate(loader, start=1):
            real_x = real_x.to(device)

            with torch.no_grad():
                z, _mu, _logvar = model.encode(real_x, sample=False)
                z = model.scale_latent(z)

            b = int(z.shape[0])
            t = torch.randint(0, int(model.diffusion.timesteps), (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(z)
            z_noisy = model.diffusion.q_sample(z, t, noise=noise)
            noise_pred = model.unet(z_noisy, t)

            loss = torch.mean((noise_pred - noise) ** 2)

            unet_opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=float(args.grad_clip))
            unet_opt.step()

            loss_sum += float(loss.item())
            n_batches += 1

            if max_batches is not None and batch_idx >= int(max_batches):
                break

        mean_loss = loss_sum / max(n_batches, 1)
        history["diffusion_loss"].append(mean_loss)

        LOGGER.info("[%s][DIFF %d/%d] loss=%.6f", class_name, epoch, int(args.diffusion_epochs), mean_loss)
        (out_dir / "history.json").write_text(json.dumps({"settings": asdict(settings), "history": history}, indent=2))
        torch.save(model.unet.state_dict(), out_dir / "unet_last.pt")

    with torch.no_grad():
        x_gen = model.generate(
            batch_size=int(args.batch_size),
            device=device,
            num_inference_steps=int(args.num_inference_steps),
        )
    if tuple(x_gen.shape[1:]) != (n_channels, n_samples):
        raise RuntimeError(f"Generation produced shape {tuple(x_gen.shape)} expected (B, {n_channels}, {n_samples})")

    _write_outputs(out_dir, model=model, settings=settings, history=history)


def main() -> None:
    args = _parse_args()
    args.seed = _resolve_seed(int(args.seed))
    _seed_everything(int(args.seed))

    device = _resolve_device(args.device, gpu_index=int(args.gpu))

    split_root = Path(args.split_root)
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    run_name = str(args.run_name).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_root = Path("results") / "saved_ldm"
    _ensure_output_structure(saved_root)

    log_path = saved_root / "logs" / f"{run_name}.log"
    _setup_logging(log_file=log_path)

    class_name = str(args.class_name).strip().lower()
    if (not bool(args.all_classes)) and (not class_name):
        raise ValueError("Provide --class-name or use --all-classes")

    normalize = not bool(args.no_normalize)

    train_dir = split_root / "train" / str(args.data_subdir)
    files = build_file_list(train_dir)
    if not files:
        raise FileNotFoundError(f"No EDF files found under: {train_dir}")

    base_ds = EEGWindowDataset(
        files,
        sfreq=float(args.sfreq),
        window_sec=float(args.window_sec),
        stride_sec=float(args.stride_sec),
        label_names=ARTIFACT_CLASSES,
        min_overlap_sec=float(args.min_overlap_sec),
        min_overlap_frac_of_artifact=float(args.min_overlap_frac_artifact),
        normalize=normalize,
        cache=True,
        augment_pink_noise=False,
        augment_pink_noise_prob=0.0,
        augment_time_domain=False,
        augment_time_domain_crop_frac=1.0,
        augment_time_domain_shift_frac=0.0,
        augment_segment_recombination=False,
        augment_segment_recombination_prob=0.0,
        augment_segment_recombination_n_segments=8,
        augment_channel_dropout=False,
        augment_channel_dropout_prob=0.0,
        augment_channel_dropout_frac=0.0,
        augment_mixup=False,
        augment_mixup_prob=0.0,
        augment_gaussian_noise=False,
        augment_gaussian_noise_prob=0.0,
        augment_gaussian_noise_snr_db=20.0,
        seed=int(args.seed),
    )

    LOGGER.info("Run name: %s", run_name)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Train windows (all labels): %d", len(base_ds))
    LOGGER.info("Logging to: %s", log_path)

    if bool(args.all_classes):
        class_list = list(ARTIFACT_CLASSES)
    else:
        if class_name not in ARTIFACT_CLASSES:
            raise ValueError(f"Unknown class_name '{class_name}'. Expected one of: {ARTIFACT_CLASSES}")
        class_list = [class_name]

    for cn in class_list:
        out_dir = saved_root / cn
        _train_one_class(
            base_ds=base_ds,
            class_name=cn,
            out_dir=out_dir,
            device=device,
            args=args,
            run_name=run_name,
        )


if __name__ == "__main__":
    main()
