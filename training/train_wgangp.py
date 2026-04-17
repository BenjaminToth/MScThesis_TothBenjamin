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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from dataloader.eeg_dataset import ARTIFACT_CLASSES, EEGWindowDataset, build_file_list
from dataloader.wgangp_dataset import SingleClassEEGWindowDataset
from models.WGANGP import WGANGP


LOGGER = logging.getLogger("train_wgangp")


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

    # Model
    latent_dim: int
    lambda_gp: float
    output_activation: str

    # Optim
    lr_g: float
    lr_d: float
    beta1: float
    beta2: float

    # Training
    epochs: int
    n_critic: int
    train_samples_per_epoch: int
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train WGAN-GP generators for artifact classes.")

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

    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument(
        "--output-activation",
        type=str,
        default="linear",
        choices=["linear", "tanh"],
        help="Generator output activation. Use 'linear' for z-scored windows.",
    )

    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--n-critic",
        type=int,
        default=5,
        help="Number of critic updates per generator update.",
    )
    p.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=50000,
        help="Approximate number of windows to draw per epoch (0 = no cap).",
    )

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


def _write_outputs(
    out_dir: Path,
    *,
    model: WGANGP,
    settings: RunSettings,
    history: Dict[str, List[float]],
) -> None:
    """Write the minimal artifacts needed for later generation."""
    _ensure_dir(out_dir)

    torch.save(model.generator.state_dict(), out_dir / "generator_last.pt")
    (out_dir / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    (out_dir / "history.json").write_text(json.dumps({"settings": asdict(settings), "history": history}, indent=2))


def _ensure_output_structure(saved_root: Path) -> None:
        """Ensure the fixed output folder layout exists.

        Layout:
            results/saved_wgangp/
                chew/
                elec/
                elpp/
                eyem/
                musc/
                shiv/
                logs/
        """
        for cn in ARTIFACT_CLASSES:
                _ensure_dir(saved_root / cn)
        _ensure_dir(saved_root / "logs")


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

    model = WGANGP(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=len(ARTIFACT_CLASSES),
        latent_dim=int(args.latent_dim),
        lambda_gp=float(args.lambda_gp),
        class_name=class_name,
        class_index=class_index,
        output_activation=str(args.output_activation),
    ).to(device)

    g_opt = torch.optim.Adam(
        model.generator.parameters(),
        lr=float(args.lr_g),
        betas=(float(args.beta1), float(args.beta2)),
    )
    d_opt = torch.optim.Adam(
        model.critic.parameters(),
        lr=float(args.lr_d),
        betas=(float(args.beta1), float(args.beta2)),
    )

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
        latent_dim=int(args.latent_dim),
        lambda_gp=float(args.lambda_gp),
        output_activation=str(args.output_activation),
        lr_g=float(args.lr_g),
        lr_d=float(args.lr_d),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        epochs=int(args.epochs),
        n_critic=int(args.n_critic),
        train_samples_per_epoch=int(args.train_samples_per_epoch),
        device=str(device),
        gpu=int(args.gpu),
        seed=int(args.seed),
        class_name=str(class_name),
        class_index=int(class_index),
        exclusive=bool(args.exclusive),
        run_name=str(run_name),
    )

    history: Dict[str, List[float]] = {
        "critic_loss": [],
        "gen_loss": [],
        "gp": [],
        "wasserstein": [],
    }

    start_epoch = 1

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
    if max_batches is not None:
        LOGGER.info("Capping training to ~%d windows/epoch (~%d batches)", int(args.train_samples_per_epoch), max_batches)

    n_critic = max(1, int(args.n_critic))

    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()

        c_loss_sum = 0.0
        g_loss_sum = 0.0
        gp_sum = 0.0
        wass_sum = 0.0
        n_batches = 0

        for batch_idx, real_x in enumerate(loader, start=1):
            real_x = real_x.to(device)

            for _ in range(n_critic):
                z = model.sample_noise(batch_size=real_x.shape[0], device=device)
                fake_x = model.generate(z=z)

                d_real = model.critic_score(real_x)
                d_fake = model.critic_score(fake_x.detach())

                wasserstein = d_real.mean() - d_fake.mean()
                gp = model.gradient_penalty(real_x, fake_x.detach())
                d_loss = -wasserstein + gp

                d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                d_opt.step()

            z = model.sample_noise(batch_size=real_x.shape[0], device=device)
            fake_x = model.generate(z=z)
            d_fake_for_g = model.critic_score(fake_x)
            g_loss = -d_fake_for_g.mean()

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            c_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            gp_sum += float(gp.item())
            wass_sum += float(wasserstein.item())
            n_batches += 1

            if max_batches is not None and batch_idx >= int(max_batches):
                break

        mean_c = c_loss_sum / max(n_batches, 1)
        mean_g = g_loss_sum / max(n_batches, 1)
        mean_gp = gp_sum / max(n_batches, 1)
        mean_w = wass_sum / max(n_batches, 1)

        history["critic_loss"].append(mean_c)
        history["gen_loss"].append(mean_g)
        history["gp"].append(mean_gp)
        history["wasserstein"].append(mean_w)

        LOGGER.info(
            "[%s][%d/%d] critic=%.4f gen=%.4f wasserstein=%.4f gp=%.4f",
            class_name,
            epoch,
            int(args.epochs),
            mean_c,
            mean_g,
            mean_w,
            mean_gp,
        )

        (out_dir / "history.json").write_text(json.dumps({"settings": asdict(settings), "history": history}, indent=2))

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

    saved_root = Path("results") / "saved_wgangp"
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
