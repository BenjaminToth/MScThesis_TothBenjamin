from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch

from dataloader.eeg_dataset import ARTIFACT_CLASSES, EEGWindowDataset, build_file_list
from models.CNNLSTM.cnnlstm import CNNLSTM
from models.EEGConformer.eegconformer import EEGConformer
from models.EEGInceptionERP.eeginceptionerp import EEGInceptionERP
from models.EEGNet import EEGNet
from models.EEGNeX.eegnex import EEGNeX

from dataloader.gan_synthetic_dataset import build_major_class_synthetic_augmentation
from dataloader.ldm_synthetic_dataset import build_major_class_ldm_augmentation


LOGGER = logging.getLogger("training")
MAJOR_CLASSES: Tuple[str, ...] = ("chew", "elec", "eyem", "musc")


@dataclass(frozen=True)
class RunSettings:
    # Data
    split_root: str
    batch_size: int
    num_workers: int
    sfreq: float
    window_sec: float
    stride_sec: float
    min_overlap_sec: float
    min_overlap_frac_artifact: float
    normalize: bool

    # Augmentation (train only)
    aug_pink_noise: bool
    aug_pink_noise_prob: float
    aug_pink_noise_snr_db: float
    aug_time_domain: bool
    aug_time_domain_crop_frac: float
    aug_time_domain_shift_frac: float
    aug_segment_recombination: bool
    aug_segment_recombination_prob: float
    aug_segment_recombination_n_segments: int
    aug_channel_dropout: bool
    aug_channel_dropout_prob: float
    aug_channel_dropout_frac: float
    aug_mixup: bool
    aug_mixup_prob: float
    aug_gaussian_noise: bool
    aug_gaussian_noise_prob: float
    aug_gaussian_noise_snr_db: float

    # GAN augmentation (train only)
    aug_wgangp_major: bool
    aug_wgangp_n_per_class: int
    aug_wgangp_root: str

    # LDM augmentation (train only)
    aug_ldm_major: bool
    aug_ldm_n_per_class: int
    aug_ldm_root: str
    aug_ldm_num_inference_steps: int

    # Model
    model: str
    n_channels: int
    n_classes: int
    model_kwargs: Dict[str, Any]

    # Training
    epochs: int
    lr: float
    weight_decay: float
    device: str
    gpu: int
    seed: int

    # Epoch sizing / schedule
    train_samples_per_epoch: int
    sampler_warmup_epochs: int

    # Imbalance
    sampler: str
    repeat_threshold: float
    major_sampler_classes: Tuple[str, ...]

    # Loss (ASL)
    gamma_pos: float
    gamma_neg: float
    clip: float

    # Metrics
    pred_threshold: float

    # Threshold tuning (dev)
    tune_thresholds: bool
    threshold_grid_size: int

    # Output
    run_name: str


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True  
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_seed(seed_arg: int) -> int:
    """Resolve the seed value used for the run.

    - If seed_arg >= 0, use it as-is.
    - If seed_arg < 0, draw a fresh seed from OS entropy.
    """

    seed_int = int(seed_arg)
    if seed_int >= 0:
        return seed_int

    resolved = int(secrets.randbits(32))
    LOGGER.info("Resolved random seed (from --seed %d): %d", seed_int, resolved)
    return resolved


def _device_from_arg(device: str) -> torch.device:
    raise RuntimeError("Use _resolve_device(args_device, gpu_index) instead")


def _resolve_device(args_device: str, *, gpu_index: int) -> torch.device:
    d = str(args_device).strip().lower()

    if d == "auto":
        if torch.cuda.is_available():
            d = "cuda"
        else:
            d = "cpu"

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


def build_model(
    model_name: str,
    *,
    n_channels: int,
    n_samples: int | None,
    n_classes: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    name = str(model_name).strip().lower()

    if name == "eegnet":
        model_kwargs: Dict[str, Any] = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "dropout": 0.5,
            "kernel_length": 63,
            "depthwise_kernel_length": 15,
            "f1": 8,
            "d": 2,
            "f2": None,
        }
        return EEGNet(**model_kwargs), model_kwargs

    if name == "eegnex":
        model_kwargs = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "depth_multiplier": 2,
            "filter_1": 8,
            "filter_2": 32,
            "dropout": 0.5,
            "kernel_block_1_2": 32,
            "kernel_block_4": 16,
            "dilation_block_4": 2,
            "avg_pool_block4": 4,
            "kernel_block_5": 16,
            "dilation_block_5": 4,
            "avg_pool_block5": 8,
            "max_norm_conv": 1.0,
            "max_norm_linear": 0.25,
        }
        return EEGNeX(**model_kwargs), model_kwargs

    if name == "eegconformer":
        model_kwargs = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_filters_time": 40,
            "filter_time_length": 25,
            "pool_time_length": 75,
            "pool_time_stride": 15,
            "drop_prob": 0.25,
            "att_depth": 4,
            "att_heads": 10,
            "att_drop_prob": 0.1,
            "final_fc_length": "auto",
            "return_features": False,
            "ff_expansion": 2,
            "cls_hidden_features": 128,
        }
        return EEGConformer(**model_kwargs), model_kwargs

    if name == "cnnlstm":
        model_kwargs = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "cnn_hidden_1": 64,
            "cnn_hidden_2": 128,
            "kernel_size_1": 15,
            "kernel_size_2": 7,
            "pool_size_1": 2,
            "pool_size_2": 2,
            "lstm_hidden_size": 192,
            "lstm_num_layers": 2,
            "bidirectional": True,
            "dropout": 0.35,
        }
        return CNNLSTM(**model_kwargs), model_kwargs

    if name == "eeginceptionerp":
        model_kwargs = {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_filters": 8,
            "depth_multiplier": 2,
            "stage1_kernel_sizes": (64, 32, 16),
            "stage2_kernel_sizes": (16, 8, 4),
            "stage1_pool_size": 4,
            "stage2_pool_size": 2,
            "stage3_kernel_size": 8,
            "stage4_kernel_size": 4,
            "stage3_pool_size": 2,
            "stage4_pool_size": 2,
            "dropout": 0.5,
        }
        return EEGInceptionERP(**model_kwargs), model_kwargs

    raise ValueError(
        f"Unknown model: {model_name}. Supported models: eegnet, eegnex, eegconformer, cnnlstm, eeginceptionerp"
    )


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_pos: float,
    gamma_neg: float,
    clip: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sigmoid ASL (multi-label) on logits.

    - logits: (B, C)
    - targets: (B, C) in {0,1}

    Returns mean loss over batch and classes.
    """

    targets = targets.to(dtype=logits.dtype)

    prob = torch.sigmoid(logits)
    prob_pos = prob
    prob_neg = 1.0 - prob

    if clip > 0:
        prob_neg = torch.clamp(prob_neg + clip, max=1.0)

    loss_pos = targets * torch.log(torch.clamp(prob_pos, min=eps))
    loss_neg = (1.0 - targets) * torch.log(torch.clamp(prob_neg, min=eps))

    loss = loss_pos + loss_neg

    if gamma_pos > 0 or gamma_neg > 0:
        pt = targets * prob_pos + (1.0 - targets) * prob_neg
        gamma = targets * gamma_pos + (1.0 - targets) * gamma_neg
        focal_weight = torch.pow(torch.clamp(1.0 - pt, min=0.0), gamma)
        loss = loss * focal_weight

    return -loss.mean()


def _dataset_for_split(split_root: Path, split_name: Literal["train", "dev", "test"], data_subdir: str) -> EEGWindowDataset:
    edf_dir = split_root / split_name / data_subdir
    files = build_file_list(edf_dir)
    if not files:
        raise FileNotFoundError(f"No EDF files found under: {edf_dir}")
    return EEGWindowDataset(files)


def _build_datasets(
    split_root: Path,
    *,
    data_subdir: str,
    sfreq: float,
    window_sec: float,
    stride_sec: float,
    min_overlap_sec: float,
    min_overlap_frac_artifact: float,
    normalize: bool,
    seed: int,
    aug_pink_noise: bool,
    aug_pink_noise_prob: float,
    aug_pink_noise_snr_db: float,
    aug_time_domain: bool,
    aug_time_domain_crop_frac: float,
    aug_time_domain_shift_frac: float,
    aug_segment_recombination: bool,
    aug_segment_recombination_prob: float,
    aug_segment_recombination_n_segments: int,
    aug_channel_dropout: bool,
    aug_channel_dropout_prob: float,
    aug_channel_dropout_frac: float,
    aug_mixup: bool,
    aug_mixup_prob: float,
    aug_gaussian_noise: bool,
    aug_gaussian_noise_prob: float,
    aug_gaussian_noise_snr_db: float,
) -> Tuple[EEGWindowDataset, EEGWindowDataset]:
    train_dir = split_root / "train" / data_subdir
    dev_dir = split_root / "dev" / data_subdir

    train_files = build_file_list(train_dir)
    dev_files = build_file_list(dev_dir)
    if not train_files:
        raise FileNotFoundError(f"No EDF files found under: {train_dir}")
    if not dev_files:
        raise FileNotFoundError(f"No EDF files found under: {dev_dir}")

    dataset_kwargs = {
        "sfreq": float(sfreq),
        "window_sec": float(window_sec),
        "stride_sec": float(stride_sec),
        "label_names": ARTIFACT_CLASSES,
        "min_overlap_sec": float(min_overlap_sec),
        "min_overlap_frac_of_artifact": float(min_overlap_frac_artifact),
        "normalize": bool(normalize),
    }

    train_ds = EEGWindowDataset(
        train_files,
        **dataset_kwargs,
        augment_pink_noise=bool(aug_pink_noise),
        augment_pink_noise_prob=float(aug_pink_noise_prob) if bool(aug_pink_noise) else 0.0,
        augment_pink_noise_snr_db=float(aug_pink_noise_snr_db),
        augment_time_domain=bool(aug_time_domain),
        augment_time_domain_crop_frac=float(aug_time_domain_crop_frac),
        augment_time_domain_shift_frac=float(aug_time_domain_shift_frac),
        augment_segment_recombination=bool(aug_segment_recombination),
        augment_segment_recombination_prob=float(aug_segment_recombination_prob) if bool(aug_segment_recombination) else 0.0,
        augment_segment_recombination_n_segments=int(aug_segment_recombination_n_segments),
        augment_channel_dropout=bool(aug_channel_dropout),
        augment_channel_dropout_prob=float(aug_channel_dropout_prob) if bool(aug_channel_dropout) else 0.0,
        augment_channel_dropout_frac=float(aug_channel_dropout_frac),
        augment_mixup=bool(aug_mixup),
        augment_mixup_prob=float(aug_mixup_prob) if bool(aug_mixup) else 0.0,
        augment_gaussian_noise=bool(aug_gaussian_noise),
        augment_gaussian_noise_prob=float(aug_gaussian_noise_prob) if bool(aug_gaussian_noise) else 0.0,
        augment_gaussian_noise_snr_db=float(aug_gaussian_noise_snr_db),
        seed=int(seed),
    )
    dev_ds = EEGWindowDataset(
        dev_files,
        **dataset_kwargs,
        augment_pink_noise=False,
        augment_pink_noise_prob=0.0,
        augment_pink_noise_snr_db=float(aug_pink_noise_snr_db),
        augment_time_domain=False,
        augment_time_domain_crop_frac=float(aug_time_domain_crop_frac),
        augment_time_domain_shift_frac=float(aug_time_domain_shift_frac),
        augment_segment_recombination=False,
        augment_segment_recombination_prob=0.0,
        augment_segment_recombination_n_segments=int(aug_segment_recombination_n_segments),
        augment_channel_dropout=False,
        augment_channel_dropout_prob=0.0,
        augment_channel_dropout_frac=float(aug_channel_dropout_frac),
        augment_mixup=False,
        augment_mixup_prob=0.0,
        augment_gaussian_noise=False,
        augment_gaussian_noise_prob=0.0,
        augment_gaussian_noise_snr_db=float(aug_gaussian_noise_snr_db),
        seed=int(seed),
    )

    return train_ds, dev_ds


def _compute_repeat_factor_weights(
    train_ds: EEGWindowDataset,
    *,
    repeat_threshold: float,
    class_indices: Tuple[int, ...] | None = None,
    class_group_name: str = "all",
    eps: float = 1e-12,
) -> torch.DoubleTensor:
    """Compute per-window repeat-factor weights from the dataset index.

    Uses the COCO-style repeat factor idea adapted to multi-label windows:
      r_j = max(1, max_{c in positives(j)} sqrt(t / f_c))
    where f_c is the fraction of windows that contain class c.

    Windows with no positive labels get weight 1.
    """

    if repeat_threshold <= 0:
        raise ValueError("repeat_threshold must be > 0")

    n_classes = len(train_ds.label_names)
    n = len(train_ds)

    active_indices: Tuple[int, ...]
    if class_indices is None:
        active_indices = tuple(range(n_classes))
    else:
        active_indices = tuple(int(i) for i in class_indices)
        if not active_indices:
            raise ValueError("class_indices must contain at least one class index")
        for i in active_indices:
            if i < 0 or i >= n_classes:
                raise ValueError(f"Invalid class index {i}; expected in [0, {n_classes - 1}]")

    class_counts = np.zeros(n_classes, dtype=np.int64)
    masks: List[int] = []
    for win in train_ds._index: 
        masks.append(int(win.label_mask))
        for c in active_indices:
            if win.label_mask & (1 << c):
                class_counts[c] += 1

    class_freq = class_counts.astype(np.float64) / max(n, 1)

    class_r = np.ones(n_classes, dtype=np.float64)
    for c in active_indices:
        f = max(class_freq[c], eps)
        class_r[c] = math.sqrt(repeat_threshold / f) if f < repeat_threshold else 1.0

    weights = np.ones(n, dtype=np.float64)
    for j, mask in enumerate(masks):
        if mask == 0:
            continue
        rj = 1.0
        for c in active_indices:
            if mask & (1 << c):
                rj = max(rj, class_r[c])
        weights[j] = rj

    w_t = torch.as_tensor(weights, dtype=torch.double)

    freq_str = ", ".join(
        f"{train_ds.label_names[i]}:{class_freq[i]:.4f}" for i in active_indices
    )
    LOGGER.info(
        "Train label frequency (window-level, %s classes): %s",
        class_group_name,
        freq_str,
    )
    LOGGER.info(
        "Repeat-factor weights: min=%.3f mean=%.3f max=%.3f",
        float(w_t.min().item()),
        float(w_t.mean().item()),
        float(w_t.max().item()),
    )

    return w_t


def _make_loaders(
    train_ds: EEGWindowDataset,
    dev_ds: EEGWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    sampler: str,
    repeat_threshold: float,
    seed: int,
    device: torch.device,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    sampler_name = str(sampler).strip().lower()

    if sampler_name == "repeat_factor":
        weights = _compute_repeat_factor_weights(
            train_ds,
            repeat_threshold=repeat_threshold,
            class_group_name="all",
        )
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=gen,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    elif sampler_name == "major_only":
        missing = [name for name in MAJOR_CLASSES if name not in train_ds.label_to_idx]
        if missing:
            raise ValueError(
                "sampler='major_only' requires major classes to exist in dataset labels; "
                f"missing: {missing}"
            )
        major_idx = tuple(train_ds.label_to_idx[name] for name in MAJOR_CLASSES)
        weights = _compute_repeat_factor_weights(
            train_ds,
            repeat_threshold=repeat_threshold,
            class_indices=major_idx,
            class_group_name="major_only",
        )
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=gen,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    elif sampler_name == "none":
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    dev_loader = torch.utils.data.DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, dev_loader


def _train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gamma_pos: float,
    gamma_neg: float,
    clip: float,
    show_progress: bool,
    epoch: int,
    epochs: int,
    max_batches: int | None = None,
) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    iterator = loader
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=False)

    for batch_idx, (x, y) in enumerate(iterator, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = asymmetric_loss_with_logits(
            logits,
            y,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        if max_batches is not None and batch_idx >= int(max_batches):
            break

        if show_progress and n_batches > 0:
            iterator.set_postfix(loss=f"{(total_loss / n_batches):.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    gamma_pos: float,
    gamma_neg: float,
    clip: float,
    show_progress: bool,
    epoch: int,
    epochs: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    n_batches = 0

    probs: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    iterator = loader
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(loader, desc=f"val   {epoch}/{epochs}", leave=False)

    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = asymmetric_loss_with_logits(
            logits,
            y,
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )

        total_loss += float(loss.item())
        n_batches += 1

        p = torch.sigmoid(logits).detach().cpu().numpy()
        t = y.detach().cpu().numpy()
        probs.append(p)
        targets.append(t)

        if show_progress and n_batches > 0:
            iterator.set_postfix(loss=f"{(total_loss / n_batches):.4f}")

    val_loss = total_loss / max(n_batches, 1)
    p_all = np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)
    t_all = np.concatenate(targets, axis=0) if targets else np.zeros((0, 0), dtype=np.float32)

    return val_loss, p_all, t_all


def _compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    pred_threshold: float,
) -> Tuple[float, float, List[float], List[float], List[float], List[float]]:
    from sklearn.metrics import average_precision_score, f1_score

    if probs.size == 0:
        return 0.0, 0.0, [], [], [], []

    y_true = (targets >= 0.5).astype(np.int32)
    y_pred = (probs >= float(pred_threshold)).astype(np.int32)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = [float(x) for x in np.asarray(per_class_f1_arr).ravel().tolist()]

    tp = (y_pred * y_true).sum(axis=0).astype(np.float64)
    fp = (y_pred * (1 - y_true)).sum(axis=0).astype(np.float64)
    fn = ((1 - y_pred) * y_true).sum(axis=0).astype(np.float64)
    per_class_precision = [float(x) for x in (tp / (tp + fp + 1e-12)).ravel().tolist()]
    per_class_recall = [float(x) for x in (tp / (tp + fn + 1e-12)).ravel().tolist()]

    per_class_ap: List[float] = []
    for c in range(y_true.shape[1]):
        try:
            ap = float(average_precision_score(y_true[:, c], probs[:, c]))
        except Exception:
            ap = 0.0
        per_class_ap.append(ap)

    macro_pr_auc = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    return macro_f1, macro_pr_auc, per_class_f1, per_class_ap, per_class_precision, per_class_recall


def _f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    denom = (2.0 * tp + fp + fn)
    return (2.0 * tp) / (denom + eps)


def _subset_macro(values: List[float], names: Tuple[str, ...], subset: Tuple[str, ...]) -> float:
    idx = [i for i, n in enumerate(names) if n in set(subset)]
    if not idx:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr[idx].mean())


def _tune_per_class_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    grid_size: int,
) -> List[float]:
    """Pick per-class probability thresholds that maximize F1 on the given set.

    This is purely for evaluation/inference calibration; it does not change training.
    """

    if probs.size == 0:
        return []

    if grid_size < 2:
        raise ValueError("threshold_grid_size must be >= 2")

    y_true = (targets >= 0.5).astype(np.int32)
    n_classes = int(y_true.shape[1])

    grid = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)

    thresholds: List[float] = []
    for c in range(n_classes):
        yt = y_true[:, c].astype(np.int32)
        if int(yt.sum()) == 0:
            thresholds.append(1.0)
            continue

        pc = probs[:, c].astype(np.float64)

        preds = (pc[None, :] >= grid[:, None]).astype(np.int32)
        tp = (preds * yt[None, :]).sum(axis=1).astype(np.float64)
        fp = (preds * (1 - yt)[None, :]).sum(axis=1).astype(np.float64)
        fn = (((1 - preds) * yt[None, :])).sum(axis=1).astype(np.float64)
        f1 = _f1_from_counts(tp, fp, fn)

        best = int(np.argmax(f1))

        best_f1 = float(f1[best])
        ties = np.where(np.isclose(f1, best_f1, rtol=0.0, atol=1e-12))[0]
        if ties.size > 0:
            best = int(ties[-1])

        thresholds.append(float(grid[best]))

    return thresholds


def _compute_metrics_per_class_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: List[float],
) -> Tuple[float, float, List[float], List[float], List[float], List[float]]:
    from sklearn.metrics import average_precision_score, f1_score

    if probs.size == 0:
        return 0.0, 0.0, [], [], [], []

    y_true = (targets >= 0.5).astype(np.int32)
    thr = np.asarray(thresholds, dtype=np.float64)
    if thr.ndim != 1 or thr.shape[0] != y_true.shape[1]:
        raise ValueError("thresholds must have shape (n_classes,)")

    y_pred = (probs >= thr[None, :]).astype(np.int32)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = [float(x) for x in np.asarray(per_class_f1_arr).ravel().tolist()]

    tp = (y_pred * y_true).sum(axis=0).astype(np.float64)
    fp = (y_pred * (1 - y_true)).sum(axis=0).astype(np.float64)
    fn = ((1 - y_pred) * y_true).sum(axis=0).astype(np.float64)
    per_class_precision = [float(x) for x in (tp / (tp + fp + 1e-12)).ravel().tolist()]
    per_class_recall = [float(x) for x in (tp / (tp + fn + 1e-12)).ravel().tolist()]

    per_class_ap: List[float] = []
    for c in range(y_true.shape[1]):
        try:
            ap = float(average_precision_score(y_true[:, c], probs[:, c]))
        except Exception:
            ap = 0.0
        per_class_ap.append(ap)
    macro_pr_auc = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    return macro_f1, macro_pr_auc, per_class_f1, per_class_ap, per_class_precision, per_class_recall


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _round_metrics_recursively(obj: Any, decimals: int = 4) -> Any:
    """Recursively round all float values in a data structure to the specified decimal places."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: _round_metrics_recursively(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_round_metrics_recursively(item, decimals) for item in obj)
    else:
        return obj


def _save_checkpoint(
    out_path: Path,
    *,
    model: torch.nn.Module,
    model_name: str,
    model_kwargs: Dict[str, Any],
    label_names: Tuple[str, ...],
    settings: RunSettings,
    epoch: int,
    best_val_macro_f1: float,
    best_thresholds_per_class: List[float] | None = None,
) -> None:
    payload = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "label_names": list(label_names),
        "settings": asdict(settings),
        "epoch": int(epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_thresholds_per_class": list(best_thresholds_per_class or []),
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, out_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an EEG artifact detector.")

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
        "--aug-pink-noise",
        action="store_true",
        help="Apply pink-noise (1/f) augmentation to training windows only.",
    )
    p.add_argument(
        "--aug-pink-noise-prob",
        type=float,
        default=0.5,
        help="Probability of applying pink noise to a training window.",
    )
    p.add_argument(
        "--aug-pink-noise-snr-db",
        type=float,
        default=20.0,
        help="SNR (dB) of added pink noise relative to the window RMS (lower = stronger noise).",
    )
    p.add_argument(
        "--aug-time-domain",
        action="store_true",
        help="Apply time-domain cropping + jitter augmentation to training windows only.",
    )
    p.add_argument(
        "--aug-time-domain-crop-frac",
        type=float,
        default=0.9,
        help="Fraction of window to keep during cropping (e.g., 0.9 = 90%%).",
    )
    p.add_argument(
        "--aug-time-domain-shift-frac",
        type=float,
        default=0.05,
        help="Max positional jitter as fraction of window length (e.g., 0.05 = ±5%%).",
    )
    p.add_argument(
        "--aug-segment-recombination",
        action="store_true",
        help="Apply segment recombination with phase-aware alignment to training windows only.",
    )
    p.add_argument(
        "--aug-segment-recombination-prob",
        type=float,
        default=0.5,
        help="Probability of applying segment recombination to a training window.",
    )
    p.add_argument(
        "--aug-segment-recombination-n-segments",
        type=int,
        default=8,
        help="Number of segments to divide the window into for recombination (default: 8).",
    )
    p.add_argument(
        "--aug-channel-dropout",
        action="store_true",
        help="Apply channel dropout augmentation to training windows only.",
    )
    p.add_argument(
        "--aug-channel-dropout-prob",
        type=float,
        default=0.5,
        help="Probability of applying channel dropout to a training window (default: 0.5).",
    )
    p.add_argument(
        "--aug-channel-dropout-frac",
        type=float,
        default=0.1,
        help="Fraction of channels to randomly zero out when applying dropout (default: 0.1).",
    )
    p.add_argument(
        "--aug-mixup",
        action="store_true",
        help="Apply Mixup augmentation: blend target with two noise windows at 90 percent and 10 percent blend ratio.",
    )
    p.add_argument(
        "--aug-mixup-prob",
        type=float,
        default=0.5,
        help="Probability of applying Mixup to a training window.",
    )
    p.add_argument(
        "--aug-gaussian-noise",
        action="store_true",
        help="Apply Gaussian (white) noise augmentation to training windows only.",
    )
    p.add_argument(
        "--aug-gaussian-noise-prob",
        type=float,
        default=0.5,
        help="Probability of applying Gaussian noise to a training window.",
    )
    p.add_argument(
        "--aug-gaussian-noise-snr-db",
        type=float,
        default=20.0,
        help="SNR (dB) of added Gaussian noise relative to the window RMS (lower = stronger noise).",
    )

    p.add_argument(
        "--aug-wgangp-major",
        action="store_true",
        help="Augment training data with WGAN-GP synthetic windows for major classes (chew/elec/eyem/musc).",
    )
    p.add_argument(
        "--aug-wgangp-n-per-class",
        type=int,
        default=10000,
        help="Number of synthetic windows to add per major class when --aug-wgangp-major is enabled.",
    )
    p.add_argument(
        "--aug-wgangp-root",
        type=str,
        default="results/saved_wgangp",
        help="Root folder containing per-class WGAN-GP generators (expects <root>/<class>/generator_last.pt).",
    )

    p.add_argument(
        "--aug-ldm-major",
        action="store_true",
        help="Augment training data with LDM synthetic windows for major classes (chew/elec/eyem/musc).",
    )
    p.add_argument(
        "--aug-ldm-n-per-class",
        type=int,
        default=10000,
        help="Number of synthetic windows to add per major class when --aug-ldm-major is enabled.",
    )
    p.add_argument(
        "--aug-ldm-root",
        type=str,
        default="results/saved_ldm",
        help="Root folder containing per-class LDM checkpoints (expects <root>/<class>/{vae_last.pt,unet_last.pt,settings.json}).",
    )
    p.add_argument(
        "--aug-ldm-num-inference-steps",
        type=int,
        default=-1,
        help="Override LDM diffusion inference steps for augmentation (>0). Use -1 to use the checkpoint setting.",
    )

    p.add_argument(
        "--model",
        type=str,
        default="eegnet",
        help="Model architecture to train (e.g., eegnet, eegnex, eegconformer, cnnlstm, eeginceptionerp). Model hyperparameters are internal defaults.",
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use when --device is 'auto' or 'cuda' (ignored for --device cpu or explicit cuda:N).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (>=0 uses the provided seed; -1 uses a fresh random seed).",
    )

    p.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=50000,
        help="Approximate number of training windows to draw per epoch (0 = no cap). Useful when stride is small and the dataset becomes huge.",
    )
    p.add_argument(
        "--sampler-warmup-epochs",
        type=int,
        default=1,
        help="Use uniform shuffling (no repeat-factor oversampling) for the first N epochs, then enable the configured --sampler.",
    )

    p.add_argument("--sampler", type=str, choices=["repeat_factor", "major_only", "none"], default="repeat_factor")
    p.add_argument("--repeat-threshold", type=float, default=0.1)

    p.add_argument("--gamma-pos", type=float, default=1.0)
    p.add_argument("--gamma-neg", type=float, default=2.0)
    p.add_argument("--clip", type=float, default=0.05)

    p.add_argument("--pred-threshold", type=float, default=0.5)
    p.add_argument(
        "--no-tune-thresholds",
        action="store_true",
        help="Disable per-class threshold tuning on dev (uses only --pred-threshold).",
    )
    p.add_argument(
        "--threshold-grid-size",
        type=int,
        default=101,
        help="Number of thresholds in [0,1] to sweep per class when tuning on dev.",
    )

    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (epoch-level logging still printed).",
    )

    p.add_argument("--run-name", type=str, default="")

    return p.parse_args()


def main() -> None:
    _setup_logging()

    args = _parse_args()
    args.seed = _resolve_seed(int(args.seed))
    _seed_everything(int(args.seed))
    LOGGER.info("Using seed: %d", int(args.seed))

    show_progress = (not bool(args.no_progress))

    device = _resolve_device(args.device, gpu_index=int(args.gpu))

    split_root = Path(args.split_root)
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    run_name = str(args.run_name).strip()
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    normalize = not bool(args.no_normalize)

    train_ds, dev_ds = _build_datasets(
        split_root,
        data_subdir=str(args.data_subdir),
        sfreq=float(args.sfreq),
        window_sec=float(args.window_sec),
        stride_sec=float(args.stride_sec),
        min_overlap_sec=float(args.min_overlap_sec),
        min_overlap_frac_artifact=float(args.min_overlap_frac_artifact),
        normalize=normalize,
        seed=int(args.seed),
        aug_pink_noise=bool(args.aug_pink_noise),
        aug_pink_noise_prob=float(args.aug_pink_noise_prob),
        aug_pink_noise_snr_db=float(args.aug_pink_noise_snr_db),
        aug_time_domain=bool(args.aug_time_domain),
        aug_time_domain_crop_frac=float(args.aug_time_domain_crop_frac),
        aug_time_domain_shift_frac=float(args.aug_time_domain_shift_frac),
        aug_segment_recombination=bool(args.aug_segment_recombination),
        aug_segment_recombination_prob=float(args.aug_segment_recombination_prob),
        aug_segment_recombination_n_segments=int(args.aug_segment_recombination_n_segments),
        aug_channel_dropout=bool(args.aug_channel_dropout),
        aug_channel_dropout_prob=float(args.aug_channel_dropout_prob),
        aug_channel_dropout_frac=float(args.aug_channel_dropout_frac),
        aug_mixup=bool(args.aug_mixup),
        aug_mixup_prob=float(args.aug_mixup_prob),
        aug_gaussian_noise=bool(args.aug_gaussian_noise),
        aug_gaussian_noise_prob=float(args.aug_gaussian_noise_prob),
        aug_gaussian_noise_snr_db=float(args.aug_gaussian_noise_snr_db),
    )

    if bool(args.aug_wgangp_major):
        if int(args.aug_wgangp_n_per_class) <= 0:
            raise ValueError("--aug-wgangp-n-per-class must be > 0")

        train_ds = build_major_class_synthetic_augmentation(
            real_train_ds=train_ds,
            wgangp_root=Path(str(args.aug_wgangp_root)),
            major_classes=tuple(MAJOR_CLASSES),
            n_per_class=int(args.aug_wgangp_n_per_class),
            sfreq=float(args.sfreq),
            window_sec=float(args.window_sec),
            normalize=normalize,
            seed=int(args.seed),
        )

    if bool(args.aug_ldm_major):
        if int(args.aug_ldm_n_per_class) <= 0:
            raise ValueError("--aug-ldm-n-per-class must be > 0")

        n_steps_override: int | None
        if int(args.aug_ldm_num_inference_steps) <= 0:
            n_steps_override = None
        else:
            n_steps_override = int(args.aug_ldm_num_inference_steps)

        gen_device = device
        if gen_device.type == "cuda" and int(args.num_workers) > 0:
            raise ValueError(
                "LDM augmentation uses CUDA generation; set --num-workers 0 to avoid CUDA inside DataLoader workers."
            )

        train_ds = build_major_class_ldm_augmentation(
            real_train_ds=train_ds,
            ldm_root=Path(str(args.aug_ldm_root)),
            major_classes=tuple(MAJOR_CLASSES),
            n_per_class=int(args.aug_ldm_n_per_class),
            sfreq=float(args.sfreq),
            window_sec=float(args.window_sec),
            normalize=normalize,
            seed=int(args.seed),
            device=gen_device,
            num_inference_steps_override=n_steps_override,
        )

    n_channels = int(train_ds[0][0].shape[0])
    n_samples = int(train_ds[0][0].shape[1])
    n_classes = int(train_ds[0][1].shape[0])

    model, model_kwargs = build_model(
        args.model,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
    )
    model = model.to(device)

    train_loader, dev_loader = _make_loaders(
        train_ds,
        dev_ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        sampler=str(args.sampler),
        repeat_threshold=float(args.repeat_threshold),
        seed=int(args.seed),
        device=device,
    )

    warmup_epochs = max(0, int(args.sampler_warmup_epochs))
    if warmup_epochs > 0:
        warmup_loader, _ = _make_loaders(
            train_ds,
            dev_ds,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            sampler="none",
            repeat_threshold=float(args.repeat_threshold),
            seed=int(args.seed),
            device=device,
        )
    else:
        warmup_loader = train_loader

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    settings = RunSettings(
        split_root=str(split_root),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        sfreq=float(args.sfreq),
        window_sec=float(args.window_sec),
        stride_sec=float(args.stride_sec),
        min_overlap_sec=float(args.min_overlap_sec),
        min_overlap_frac_artifact=float(args.min_overlap_frac_artifact),
        normalize=normalize,

        aug_pink_noise=bool(args.aug_pink_noise),
        aug_pink_noise_prob=float(args.aug_pink_noise_prob) if bool(args.aug_pink_noise) else 0.0,
        aug_pink_noise_snr_db=float(args.aug_pink_noise_snr_db),
        aug_time_domain=bool(args.aug_time_domain),
        aug_time_domain_crop_frac=float(args.aug_time_domain_crop_frac),
        aug_time_domain_shift_frac=float(args.aug_time_domain_shift_frac),
        aug_segment_recombination=bool(args.aug_segment_recombination),
        aug_segment_recombination_prob=float(args.aug_segment_recombination_prob) if bool(args.aug_segment_recombination) else 0.0,
        aug_segment_recombination_n_segments=int(args.aug_segment_recombination_n_segments),
        aug_channel_dropout=bool(args.aug_channel_dropout),
        aug_channel_dropout_prob=float(args.aug_channel_dropout_prob) if bool(args.aug_channel_dropout) else 0.0,
        aug_channel_dropout_frac=float(args.aug_channel_dropout_frac),
        aug_mixup=bool(args.aug_mixup),
        aug_mixup_prob=float(args.aug_mixup_prob) if bool(args.aug_mixup) else 0.0,
        aug_gaussian_noise=bool(args.aug_gaussian_noise),
        aug_gaussian_noise_prob=float(args.aug_gaussian_noise_prob) if bool(args.aug_gaussian_noise) else 0.0,
        aug_gaussian_noise_snr_db=float(args.aug_gaussian_noise_snr_db),

        aug_wgangp_major=bool(args.aug_wgangp_major),
        aug_wgangp_n_per_class=int(args.aug_wgangp_n_per_class),
        aug_wgangp_root=str(args.aug_wgangp_root),

        aug_ldm_major=bool(args.aug_ldm_major),
        aug_ldm_n_per_class=int(args.aug_ldm_n_per_class),
        aug_ldm_root=str(args.aug_ldm_root),
        aug_ldm_num_inference_steps=int(args.aug_ldm_num_inference_steps),
        model=str(args.model),
        n_channels=n_channels,
        n_classes=n_classes,
        model_kwargs=dict(model_kwargs),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=str(device),
        gpu=int(args.gpu),
        seed=int(args.seed),

        train_samples_per_epoch=int(args.train_samples_per_epoch),
        sampler_warmup_epochs=int(args.sampler_warmup_epochs),
        sampler=str(args.sampler),
        repeat_threshold=float(args.repeat_threshold),
        major_sampler_classes=tuple(MAJOR_CLASSES),
        gamma_pos=float(args.gamma_pos),
        gamma_neg=float(args.gamma_neg),
        clip=float(args.clip),
        pred_threshold=float(args.pred_threshold),
        tune_thresholds=(not bool(args.no_tune_thresholds)),
        threshold_grid_size=int(args.threshold_grid_size),
        run_name=run_name,
    )

    exp_dir = Path("results") / "experiments"
    models_dir = Path("results") / "saved_models"
    _ensure_dir(exp_dir)
    _ensure_dir(models_dir)

    ckpt_path = models_dir / f"{run_name}_best.pt"
    random_id = secrets.randbelow(100000000)
    json_filename = f"{args.model}_{random_id:08d}.json"
    json_path = exp_dir / json_filename

    best_metrics = {
        "val_loss": {"value": float("inf"), "epoch": -1},
        "val_macro_PR_AUC": {"value": -float("inf"), "epoch": -1},
        "val_macro_PR_AUC_major": {"value": -float("inf"), "epoch": -1},
        "val_macro_F1_tuned": {"value": -float("inf"), "epoch": -1},
        "val_macro_F1_major_tuned": {"value": -float("inf"), "epoch": -1},
        "val_macro_F1_minor_tuned": {"value": -float("inf"), "epoch": -1},
        "val_thresholds_per_class": {"value": None, "epoch": -1},
        "val_F1_per_class_tuned": {"value": None, "epoch": -1},
        "val_precision_per_class_tuned": {"value": None, "epoch": -1},
        "val_recall_per_class_tuned": {"value": None, "epoch": -1},
    }
    best_epoch = -1
    best_val_macro_f1 = -float("inf")
    best_thresholds_per_class: List[float] = []

    LOGGER.info("Run name: %s", run_name)
    LOGGER.info("Device: %s", device)
    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Train windows: %d | Dev windows: %d", len(train_ds), len(dev_ds))

    if bool(args.aug_wgangp_major):
        LOGGER.info(
            "Augmentation: WGAN-GP major-class synthetic enabled (n_per_class=%d, classes=%s, root=%s) [train only]",
            int(args.aug_wgangp_n_per_class),
            ",".join(MAJOR_CLASSES),
            str(args.aug_wgangp_root),
        )
    if bool(args.aug_ldm_major):
        steps_str = (
            "checkpoint" if int(args.aug_ldm_num_inference_steps) <= 0 else str(int(args.aug_ldm_num_inference_steps))
        )
        LOGGER.info(
            "Augmentation: LDM major-class synthetic enabled (n_per_class=%d, classes=%s, root=%s, steps=%s) [train only]",
            int(args.aug_ldm_n_per_class),
            ",".join(MAJOR_CLASSES),
            str(args.aug_ldm_root),
            steps_str,
        )
    if bool(args.aug_pink_noise):
        LOGGER.info(
            "Augmentation: pink noise enabled (prob=%.3f, snr_db=%.2f) [train only]",
            float(args.aug_pink_noise_prob),
            float(args.aug_pink_noise_snr_db),
        )
    if bool(args.aug_gaussian_noise):
        LOGGER.info(
            "Augmentation: gaussian noise enabled (prob=%.3f, snr_db=%.2f) [train only]",
            float(args.aug_gaussian_noise_prob),
            float(args.aug_gaussian_noise_snr_db),
        )
    if bool(args.aug_time_domain):
        LOGGER.info(
            "Augmentation: time-domain transform enabled (crop_frac=%.2f, shift_frac=%.2f) [train only]",
            float(args.aug_time_domain_crop_frac),
            float(args.aug_time_domain_shift_frac),
        )
    if bool(args.aug_segment_recombination):
        LOGGER.info(
            "Augmentation: segment recombination enabled (prob=%.3f, n_segments=%d) [train only]",
            float(args.aug_segment_recombination_prob),
            int(args.aug_segment_recombination_n_segments),
        )
    if int(args.train_samples_per_epoch) > 0:
        approx_batches = int(math.ceil(int(args.train_samples_per_epoch) / max(int(args.batch_size), 1)))
        LOGGER.info("Capping training to ~%d windows/epoch (~%d batches)", int(args.train_samples_per_epoch), approx_batches)

    for epoch in range(1, int(args.epochs) + 1):
        active_train_loader = warmup_loader if (epoch <= warmup_epochs) else train_loader

        max_batches = None
        if int(args.train_samples_per_epoch) > 0:
            max_batches = int(math.ceil(int(args.train_samples_per_epoch) / max(int(args.batch_size), 1)))

        train_loss = _train_one_epoch(
            model,
            active_train_loader,
            optimizer,
            device=device,
            gamma_pos=float(args.gamma_pos),
            gamma_neg=float(args.gamma_neg),
            clip=float(args.clip),
            show_progress=show_progress,
            epoch=epoch,
            epochs=int(args.epochs),
            max_batches=max_batches,
        )

        val_loss, probs, targets = _validate_one_epoch(
            model,
            dev_loader,
            device=device,
            gamma_pos=float(args.gamma_pos),
            gamma_neg=float(args.gamma_neg),
            clip=float(args.clip),
            show_progress=show_progress,
            epoch=epoch,
            epochs=int(args.epochs),
        )
        (
            val_macro_f1,
            val_macro_pr_auc,
            val_per_class_f1,
            val_per_class_ap,
            val_per_class_precision,
            val_per_class_recall,
        ) = _compute_metrics(
            probs,
            targets,
            pred_threshold=float(args.pred_threshold),
        )

        tuned_thresholds: List[float] = []
        val_macro_f1_tuned = 0.0
        val_macro_pr_auc_tuned = 0.0
        val_per_class_f1_tuned: List[float] = []
        val_per_class_ap_tuned: List[float] = []
        val_per_class_precision_tuned: List[float] = []
        val_per_class_recall_tuned: List[float] = []
        if not bool(args.no_tune_thresholds):
            tuned_thresholds = _tune_per_class_thresholds(
                probs,
                targets,
                grid_size=int(args.threshold_grid_size),
            )
            (
                val_macro_f1_tuned,
                val_macro_pr_auc_tuned,
                val_per_class_f1_tuned,
                val_per_class_ap_tuned,
                val_per_class_precision_tuned,
                val_per_class_recall_tuned,
            ) = _compute_metrics_per_class_thresholds(
                probs,
                targets,
                thresholds=tuned_thresholds,
            )
        else:
            val_macro_f1_tuned = float(val_macro_f1)
            val_macro_pr_auc_tuned = float(val_macro_pr_auc)
            val_per_class_f1_tuned = [float(x) for x in val_per_class_f1]
            val_per_class_ap_tuned = [float(x) for x in val_per_class_ap]
            val_per_class_precision_tuned = [float(x) for x in val_per_class_precision]
            val_per_class_recall_tuned = [float(x) for x in val_per_class_recall]
            tuned_thresholds = [float(args.pred_threshold)] * len(train_ds.label_names)

        val_macro_pr_auc_major = _subset_macro(val_per_class_ap, train_ds.label_names, MAJOR_CLASSES)
        val_macro_f1_major_tuned = _subset_macro(val_per_class_f1_tuned, train_ds.label_names, ("chew", "elec", "eyem", "musc"))
        val_macro_f1_minor_tuned = _subset_macro(val_per_class_f1_tuned, train_ds.label_names, ("elpp", "shiv"))
        thresholds_dict = {
            name: float(tuned_thresholds[i]) if i < len(tuned_thresholds) else float(args.pred_threshold)
            for i, name in enumerate(train_ds.label_names)
        }

        val_f1_per_class_tuned_dict = {
            name: float(val_per_class_f1_tuned[i]) if i < len(val_per_class_f1_tuned) else 0.0
            for i, name in enumerate(train_ds.label_names)
        }
        val_precision_per_class_tuned_dict = {
            name: float(val_per_class_precision_tuned[i]) if i < len(val_per_class_precision_tuned) else 0.0
            for i, name in enumerate(train_ds.label_names)
        }
        val_recall_per_class_tuned_dict = {
            name: float(val_per_class_recall_tuned[i]) if i < len(val_per_class_recall_tuned) else 0.0
            for i, name in enumerate(train_ds.label_names)
        }

        if float(val_loss) < best_metrics["val_loss"]["value"]:
            best_metrics["val_loss"] = {"value": float(val_loss), "epoch": epoch}
        if float(val_macro_pr_auc) > best_metrics["val_macro_PR_AUC"]["value"]:
            best_metrics["val_macro_PR_AUC"] = {"value": float(val_macro_pr_auc), "epoch": epoch}
        if float(val_macro_pr_auc_major) > best_metrics["val_macro_PR_AUC_major"]["value"]:
            best_metrics["val_macro_PR_AUC_major"] = {"value": float(val_macro_pr_auc_major), "epoch": epoch}

        is_best_tuned_f1 = float(val_macro_f1_tuned) > best_metrics["val_macro_F1_tuned"]["value"]
        if is_best_tuned_f1:
            best_metrics["val_macro_F1_tuned"] = {"value": float(val_macro_f1_tuned), "epoch": epoch}
            best_metrics["val_thresholds_per_class"] = {"value": thresholds_dict, "epoch": epoch}
            best_metrics["val_F1_per_class_tuned"] = {"value": val_f1_per_class_tuned_dict, "epoch": epoch}
            best_metrics["val_precision_per_class_tuned"] = {"value": val_precision_per_class_tuned_dict, "epoch": epoch}
            best_metrics["val_recall_per_class_tuned"] = {"value": val_recall_per_class_tuned_dict, "epoch": epoch}

        if float(val_macro_f1_major_tuned) > best_metrics["val_macro_F1_major_tuned"]["value"]:
            best_metrics["val_macro_F1_major_tuned"] = {"value": float(val_macro_f1_major_tuned), "epoch": epoch}
        if float(val_macro_f1_minor_tuned) > best_metrics["val_macro_F1_minor_tuned"]["value"]:
            best_metrics["val_macro_F1_minor_tuned"] = {"value": float(val_macro_f1_minor_tuned), "epoch": epoch}

        score_for_selection = float(val_macro_f1_tuned)

        improved = score_for_selection > best_val_macro_f1
        if improved:
            best_val_macro_f1 = float(score_for_selection)
            best_epoch = int(epoch)
            best_thresholds_per_class = list(tuned_thresholds)
            _save_checkpoint(
                ckpt_path,
                model=model,
                model_name=str(args.model),
                model_kwargs=model_kwargs,
                label_names=tuple(train_ds.label_names),
                settings=settings,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                best_thresholds_per_class=best_thresholds_per_class,
            )

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_F1=%.4f | val_macro_F1_tuned=%.4f | val_macro_PR_AUC=%.4f | saved=%s",
            epoch,
            int(args.epochs),
            train_loss,
            val_loss,
            val_macro_f1,
            val_macro_f1_tuned,
            val_macro_pr_auc,
            "yes" if improved else "no",
        )


    best_summary = {k: {"value": v["value"]} for k, v in best_metrics.items() if v["value"] is not None}

    settings_dict = asdict(settings)
    settings_dict.pop("run_name", None)
    
    # Round all metrics to 4 decimal places
    best_summary_rounded = _round_metrics_recursively(best_summary, decimals=4)
    results = {
        "best_metrics": best_summary_rounded,
        "settings": settings_dict,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    LOGGER.info("Wrote results JSON: %s", json_path)
    LOGGER.info("Best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
