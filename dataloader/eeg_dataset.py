from __future__ import annotations

import warnings
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ARTIFACT_CLASSES: Tuple[str, ...] = ("chew", "elec", "elpp", "eyem", "musc", "shiv")
_NON_SPLIT_LABELS = {"bad_acq_skip"}

AVG17_ELECS: Tuple[str, ...] = (
    "FP1", "F7", "T3", "T5", "O1",
    "FP2", "F8", "T4", "T6", "O2",
    "C3", "CZ", "C4",
    "F3", "P3", "F4", "P4",
)

_REQUIRED_TCP_ELECTRODES = {
    "FP1", "F7", "T3", "T5", "O1",
    "FP2", "F8", "T4", "T6", "O2",
    "C3", "C4", "F3", "P3", "F4", "P4",
}

TCP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)

TCP_CHANNELS: Tuple[str, ...] = tuple(f"{a}-{c}" for a, c in TCP_PAIRS)
TCP_CHANNELS_SUFFIXED: Tuple[str, ...] = tuple(f"{ch}_TCP" for ch in TCP_CHANNELS)


@dataclass(frozen=True)
class _WindowIndex:
    file_idx: int
    start_sample: int
    end_sample: int
    label_mask: int


@dataclass(frozen=True)
class _AnnInterval:
    onset: float
    end: float
    label_indices: Tuple[int, ...]


@dataclass(frozen=True)
class _ParsedChannelName:
    base: str
    montage: str | None


def _parse_channel_name(name: str) -> _ParsedChannelName:
    """Parse TUH channel names into a base token and optional montage suffix.

    Examples:
      - 'EEG FP1-REF' -> base='FP1', montage=None
      - 'C3_AVG'      -> base='C3',  montage='AVG'
      - 'F7-T3_TCP'   -> base='F7-T3', montage='TCP'
    """
    ch = str(name).upper()
    ch = re.sub(r"\bEEG\b\s*", "", ch)
    ch = ch.strip()

    for suffix in ("-REF", "_REF", " REF", "-LE", "_LE", " LE"):
        if ch.endswith(suffix):
            ch = ch[: -len(suffix)].strip()
            break

    ch = re.sub(r"\s+", "", ch)

    montage: str | None = None
    m = re.search(r"(?:[-_]?)(AVG|TCP|CZ|LAP)$", ch)
    if m:
        montage = m.group(1)
        ch = ch[: -len(m.group(0))]

    base = ch.strip(" -_\t")
    return _ParsedChannelName(base=base, montage=montage)


def _clean_channel_name(name: str) -> str:
    """Canonicalize TUH channel names (e.g., EEG FP1-REF -> FP1)."""
    return _parse_channel_name(name).base


def _normalize_and_dedup_channel_names(raw: mne.io.Raw) -> None:
    """Rename channels to a normalized form, preserving montage suffixes where present.

    Output examples:
      - 'EEG FP1-REF' -> 'FP1'
      - 'FP1_AVG'     -> 'FP1_AVG'
      - 'F7-T3_TCP'   -> 'F7-T3_TCP'
    """
    mapping: Dict[str, str] = {}
    seen: set[str] = set()
    for ch in raw.ch_names:
        parsed = _parse_channel_name(ch)
        if not parsed.base:
            continue
        new = parsed.base if parsed.montage is None else f"{parsed.base}_{parsed.montage}"
        if new not in seen:
            mapping[ch] = new
            seen.add(new)
    if mapping:
        raw.rename_channels(mapping)


def _has_avg_channels(raw_ch_names: Sequence[str]) -> bool:
    for ch in raw_ch_names:
        parsed = _parse_channel_name(ch)
        if parsed.base and parsed.montage == "AVG":
            return True
    return False


def _load_and_clean_raw(edf_path: str) -> mne.io.Raw | None:
    """Load EDF and normalize EEG channel names to canonical electrode labels."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    _normalize_and_dedup_channel_names(raw)

    raw.pick(mne.pick_types(raw.info, eeg=True))
    if not raw.ch_names:
        return None

    finite_mask = np.isfinite(raw._data)
    if not finite_mask.all():
        raw._data[~finite_mask] = 0.0

    return raw


def _build_avg17_montage(raw: mne.io.Raw, *, already_avg: bool) -> mne.io.Raw | None:
    """Build fixed-order 17ch average-reference montage matching the repo's preprocessing.

    - If already_avg=True, picks existing *_AVG channels and does not re-reference.
    - Else, picks available electrodes, applies average reference, then suffixes *_AVG.
    - Missing channels are zero-filled so output always contains all AVG17_ELECS.
    """
    desired = [f"{ch}_AVG" for ch in AVG17_ELECS]

    if already_avg:
        names = set(raw.ch_names)
        src_picks = [ch for ch in desired if ch in names]
        if len(src_picks) < 1:
            return None

        r = raw.copy().pick(src_picks)

        present_set = set(r.ch_names)
        missing = [ch for ch in desired if ch not in present_set]
        if missing:
            info = mne.create_info(ch_names=missing, sfreq=r.info["sfreq"], ch_types=["eeg"] * len(missing))
            zeros = np.zeros((len(missing), r.n_times), dtype=r._data.dtype)
            r_missing = mne.io.RawArray(zeros, info, verbose=False)
            r.add_channels([r_missing], force_update_info=True)

        r.pick(desired)
        r.reorder_channels(desired)
        return r

    base_names = set(raw.ch_names)
    src_picks = [ch for ch in AVG17_ELECS if ch in base_names]
    if len(src_picks) < 2:
        return None

    r = raw.copy().pick(src_picks)
    mne.set_eeg_reference(r, ref_channels="average", projection=False, verbose=False)
    r.rename_channels({ch: f"{ch}_AVG" for ch in r.ch_names})

    present_set = set(r.ch_names)
    missing = [ch for ch in desired if ch not in present_set]
    if missing:
        info = mne.create_info(ch_names=missing, sfreq=r.info["sfreq"], ch_types=["eeg"] * len(missing))
        zeros = np.zeros((len(missing), r.n_times), dtype=r._data.dtype)
        r_missing = mne.io.RawArray(zeros, info, verbose=False)
        r.add_channels([r_missing], force_update_info=True)

    r.pick(desired)
    r.reorder_channels(desired)
    return r


def _create_tcp_montage(raw: mne.io.Raw) -> mne.io.Raw | None:
    """Create the standard 20-channel TCP bipolar montage."""
    if "CZ" not in raw.ch_names:
        if ("C3" in raw.ch_names) and ("C4" in raw.ch_names):
            c3 = raw.copy().pick(["C3"]).get_data()
            c4 = raw.copy().pick(["C4"]).get_data()
            cz = 0.5 * (c3 + c4)
            info = mne.create_info(["CZ"], sfreq=raw.info["sfreq"], ch_types=["eeg"])
            raw_cz = mne.io.RawArray(cz, info, verbose=False)
            raw = raw.copy().add_channels([raw_cz], force_update_info=True)
        else:
            return None

    missing = _REQUIRED_TCP_ELECTRODES.difference(raw.ch_names)
    if missing:
        return None

    anodes, cathodes = zip(*TCP_PAIRS)
    ch_names = [f"{a}-{c}_TCP" for a, c in TCP_PAIRS]
    raw_tcp = mne.set_bipolar_reference(
        raw,
        anode=list(anodes),
        cathode=list(cathodes),
        ch_name=ch_names,
        drop_refs=True,
        copy=True,
        verbose=False,
    ).pick(ch_names)

    return raw_tcp


def _pick_existing_tcp_channels(raw: mne.io.Raw) -> mne.io.Raw | None:
    """Pick TCP channels directly when the recording already stores bipolar derivations."""
    names = set(raw.ch_names)

    if all(ch in names for ch in TCP_CHANNELS):
        return raw.copy().pick(list(TCP_CHANNELS), ordered=True)

    if all(ch in names for ch in TCP_CHANNELS_SUFFIXED):
        return raw.copy().pick(list(TCP_CHANNELS_SUFFIXED), ordered=True)

    return None


def _decompose_label(label: str) -> List[str]:
    """Split composite labels (eyem_musc -> [eyem, musc]) while preserving known non-composite tokens."""
    lab = str(label).strip().lower()
    if ("_" not in lab) or (lab in _NON_SPLIT_LABELS):
        return [lab]

    parts = [p for p in lab.split("_") if p]
    return parts if len(parts) > 1 else [lab]


def _parse_description_to_labels(description: str) -> List[str]:
    """Extract labels from EDF annotation text in the form '{channel}:{label}'."""
    desc = str(description).strip()
    if ":" in desc:
        _, raw_label = desc.split(":", 1)
    else:
        raw_label = desc
    return _decompose_label(raw_label)


def _mask_to_tensor(mask: int, n_labels: int) -> torch.Tensor:
    y = torch.zeros(n_labels, dtype=torch.float32)
    for i in range(n_labels):
        if mask & (1 << i):
            y[i] = 1.0
    return y


def _pink_noise_1f(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Generate 1/f (pink) noise with ~unit RMS.

    Uses simple frequency-domain shaping without adding new dependencies.
    Returns float32 array with shape (n_samples,).
    """

    n = int(n_samples)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)

    n_freq = n // 2 + 1

    re = rng.normal(0.0, 1.0, size=(n_freq,)).astype(np.float64, copy=False)
    im = rng.normal(0.0, 1.0, size=(n_freq,)).astype(np.float64, copy=False)
    spec = re + 1j * im

    k = np.arange(n_freq, dtype=np.float64)
    scale = np.ones((n_freq,), dtype=np.float64)
    scale[0] = 0.0
    scale[1:] = 1.0 / np.sqrt(k[1:])
    spec *= scale

    noise = np.fft.irfft(spec, n=n)
    noise = noise.astype(np.float32, copy=False)
    noise -= float(noise.mean())

    rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    if rms > 0:
        noise = noise / rms
    return noise


def _add_pink_noise_snr_db(x: np.ndarray, rng: np.random.Generator, *, snr_db: float) -> np.ndarray:
    """Add per-channel pink noise at a target SNR (dB) relative to signal RMS.

    - x: (C, T) float32
    Returns a new array (or modified copy) with same shape/dtype.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (C,T); got {x.shape}")

    snr_linear = float(10.0 ** (float(snr_db) / 20.0))
    snr_linear = max(snr_linear, 1e-12)

    out = np.asarray(x, dtype=np.float32)
    n_channels, n_samples = int(out.shape[0]), int(out.shape[1])

    for ch in range(n_channels):
        sig = out[ch]
        sig_rms = float(np.sqrt(np.mean(sig * sig) + 1e-12))
        noise_rms_target = sig_rms / snr_linear
        if noise_rms_target <= 0:
            continue
        noise = _pink_noise_1f(rng, n_samples)
        out[ch] = sig + noise * noise_rms_target

    return out


def _add_gaussian_noise_snr_db(x: np.ndarray, rng: np.random.Generator, *, snr_db: float) -> np.ndarray:
    """Add per-channel Gaussian (white) noise at a target SNR (dB) relative to signal RMS.

    - x: (C, T) float32
    Returns a new array (or modified copy) with same shape/dtype.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (C,T); got {x.shape}")

    snr_linear = float(10.0 ** (float(snr_db) / 20.0))
    snr_linear = max(snr_linear, 1e-12)

    out = np.asarray(x, dtype=np.float32)
    n_channels, n_samples = int(out.shape[0]), int(out.shape[1])

    for ch in range(n_channels):
        sig = out[ch]
        sig_rms = float(np.sqrt(np.mean(sig * sig) + 1e-12))
        noise_rms_target = sig_rms / snr_linear
        if noise_rms_target <= 0:
            continue
        noise = rng.normal(0.0, 1.0, size=(n_samples,)).astype(np.float32)
        out[ch] = sig + noise * noise_rms_target

    return out


def _channel_dropout(x: np.ndarray, rng: np.random.Generator, *, frac: float) -> np.ndarray:
    """Randomly zero out entire channels.

    - x: (C, T) float32
    - frac: fraction of channels to drop (e.g., 0.1 = 10%)

    Returns a new array with same shape as input.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (C,T); got {x.shape}")

    if not (0.0 <= float(frac) < 1.0):
        raise ValueError(f"frac must be in [0, 1); got {frac}")

    out = np.asarray(x, dtype=np.float32, copy=True)
    n_channels = int(out.shape[0])

    n_drop = int(np.ceil(n_channels * float(frac)))
    if n_drop < 1:
        return out

    channels_to_drop = rng.choice(n_channels, size=min(n_drop, n_channels), replace=False)
    out[channels_to_drop, :] = 0.0

    return out.astype(np.float32)


def _augment_with_noise_windows(
    x_target: np.ndarray,
    y_target: np.ndarray,
    x_noise1: np.ndarray,
    x_noise2: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend target window with averaged noise from two other windows (randomized weights).

    - x_target: (C, T) float32 target signal (dominant, 90%)
    - y_target: (n_classes,) float32 target labels (preserved)
    - x_noise1, x_noise2: (C, T) float32 noise windows to average
    - Returns: (x_mixed, y_target) where x_mixed = 0.9*x_target + 0.1*avg(noise1, noise2)
    """

    if x_target.ndim != 2 or x_noise1.ndim != 2 or x_noise2.ndim != 2:
        raise ValueError("All x arrays must have shape (C, T)")
    if x_target.shape != x_noise1.shape or x_target.shape != x_noise2.shape:
        raise ValueError(f"Shape mismatch: target {x_target.shape} vs noise {x_noise1.shape}, {x_noise2.shape}")
    if y_target.ndim != 1:
        raise ValueError(f"Expected y with shape (n_classes,); got {y_target.shape}")

    w1, w2 = rng.random(2)
    w1 = w1 / (w1 + w2)
    w2 = 1.0 - w1

    x_noise_avg = w1 * x_noise1 + w2 * x_noise2

    x_mixed = 0.9 * x_target + 0.1 * x_noise_avg

    return x_mixed.astype(np.float32), y_target.astype(np.float32)


def _time_domain_transform(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    crop_frac: float = 0.9,
    shift_frac: float = 0.1,
) -> np.ndarray:
    """Apply random cropping + positional jitter to EEG signal (no wrap-around).

    - x: (C, T) float32
    - crop_frac: fraction of samples to keep in crop (e.g., 0.9 = 90%)
    - shift_frac: max shift as fraction of window length (e.g., 0.1 = ±10%)

    Returns a new array with same shape as input.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (C,T); got {x.shape}")

    n_samples = int(x.shape[1])

    crop_size = int(n_samples * float(crop_frac))
    if crop_size < 1:
        crop_size = 1
    if crop_size >= n_samples:
        crop_size = n_samples - 1 if n_samples > 1 else 1

    crop_start = int(rng.integers(0, max(1, n_samples - crop_size + 1)))
    cropped = x[:, crop_start : crop_start + crop_size]

    pad_total = n_samples - crop_size
    if pad_total > 0:
        pad_left = int(rng.integers(0, pad_total + 1))
        pad_right = pad_total - pad_left
        padded = np.pad(cropped, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)
    else:
        padded = cropped

    max_shift = int(n_samples * float(shift_frac))
    if max_shift < 1:
        return padded.astype(np.float32)

    shift_amount = int(rng.integers(-max_shift, max_shift + 1))

    if shift_amount > 0:
        shifted = np.pad(padded[:, :-shift_amount], ((0, 0), (shift_amount, 0)), mode="constant", constant_values=0.0)
    elif shift_amount < 0:
        shifted = np.pad(padded[:, -shift_amount:], ((0, 0), (0, -shift_amount)), mode="constant", constant_values=0.0)
    else:
        shifted = padded

    return shifted.astype(np.float32)


def _segment_recombination_phase_aware(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    n_segments: int = 8,
) -> np.ndarray:
    """Shuffle time segments with FFT-based phase alignment at boundaries.

    - x: (C, T) float32
    - n_segments: number of segments to divide the window into (must be >= 2)

    Divides the window into n_segments, randomly shuffles them, and applies
    phase alignment at segment boundaries using FFT to minimize discontinuities.

    Returns a new array with same shape as input.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (C,T); got {x.shape}")

    if int(n_segments) < 2:
        raise ValueError(f"n_segments must be >= 2; got {n_segments}")

    n_channels, n_samples = int(x.shape[0]), int(x.shape[1])
    n_segments = int(n_segments)

    segment_size = n_samples // n_segments
    if segment_size < 1:
        return np.asarray(x, dtype=np.float32)

    segments: List[np.ndarray] = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else n_samples
        segments.append(x[:, start:end])

    segment_indices = list(range(n_segments))
    rng.shuffle(segment_indices)

    shuffled_parts = [segments[idx] for idx in segment_indices]
    shuffled = np.concatenate(shuffled_parts, axis=1)

    out = np.asarray(shuffled, dtype=np.float32)

    for ch in range(n_channels):
        sig = out[ch, :]
        fft_result = np.fft.rfft(sig)
        phase = np.angle(fft_result)

        phase_diff = np.diff(phase)

        cumsum_phase_correction = np.zeros_like(phase)
        for i in range(len(phase_diff)):
            if abs(phase_diff[i]) > np.pi / 2:  
                cumsum_phase_correction[i + 1] = (
                    cumsum_phase_correction[i] + np.sign(phase_diff[i]) * 2 * np.pi
                )
            else:
                cumsum_phase_correction[i + 1] = cumsum_phase_correction[i]

        corrected_phase = phase + cumsum_phase_correction
        magnitude = np.abs(fft_result)

        corrected_fft = magnitude * np.exp(1j * corrected_phase)
        sig_corrected = np.fft.irfft(corrected_fft, n=len(sig))

        out[ch, :] = sig_corrected.astype(np.float32)

    return out.astype(np.float32)


class EEGWindowDataset(Dataset):
    """Windowed average-reference (17ch) EEG dataset with multi-label targets.

    Window labels are assigned per artifact annotation interval based on the overlap
    (in seconds) between the window [t0, t1] and the artifact interval.

    A label is set for a window if there is any positive overlap and either:
    - overlap >= min_overlap_sec, OR
    - overlap >= min_overlap_frac_of_artifact * (artifact_duration)

    Each item returns:
    - x: torch.FloatTensor with shape (17, window_samples)
    - y: torch.FloatTensor with shape (num_labels,) as a multi-hot vector
    """

    def __init__(
        self,
        edf_paths: Sequence[Path | str],
        *,
        sfreq: float = 250.0,
        window_sec: float = 4.0,
        stride_sec: float = 4.0,
        label_names: Sequence[str] = ARTIFACT_CLASSES,
        min_overlap_sec: float = 0.5,
        min_overlap_frac_of_artifact: float = 0.5,
        normalize: bool = True,
        cache: bool = True,

        augment_pink_noise: bool = False,
        augment_pink_noise_prob: float = 0.5,
        augment_pink_noise_snr_db: float = 20.0,
        augment_time_domain: bool = False,
        augment_time_domain_crop_frac: float = 0.9,
        augment_time_domain_shift_frac: float = 0.05,
        augment_segment_recombination: bool = False,
        augment_segment_recombination_prob: float = 0.5,
        augment_segment_recombination_n_segments: int = 8,
        augment_channel_dropout: bool = False,
        augment_channel_dropout_prob: float = 0.5,
        augment_channel_dropout_frac: float = 0.1,
        augment_mixup: bool = False,
        augment_mixup_prob: float = 0.5,
        augment_gaussian_noise: bool = False,
        augment_gaussian_noise_prob: float = 0.5,
        augment_gaussian_noise_snr_db: float = 20.0,
        seed: int = 0,
    ) -> None:
        if window_sec <= 0 or stride_sec <= 0:
            raise ValueError("window_sec and stride_sec must be > 0")
        if sfreq <= 0:
            raise ValueError("sfreq must be > 0")

        if min_overlap_sec < 0:
            raise ValueError("min_overlap_sec must be >= 0")
        if not (0.0 <= float(min_overlap_frac_of_artifact) <= 1.0):
            raise ValueError("min_overlap_frac_of_artifact must be in [0, 1]")

        self.edf_paths: List[Path] = [Path(p) for p in edf_paths]
        self.sfreq = float(sfreq)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.window_samples = int(round(self.window_sec * self.sfreq))
        self.stride_samples = int(round(self.stride_sec * self.sfreq))
        self.min_overlap_sec = float(min_overlap_sec)
        self.min_overlap_frac_of_artifact = float(min_overlap_frac_of_artifact)
        self.normalize = bool(normalize)
        self.use_cache = bool(cache)

        self.augment_pink_noise = bool(augment_pink_noise)
        self.augment_pink_noise_prob = float(augment_pink_noise_prob)
        self.augment_pink_noise_snr_db = float(augment_pink_noise_snr_db)
        self.augment_time_domain = bool(augment_time_domain)
        self.augment_time_domain_crop_frac = float(augment_time_domain_crop_frac)
        self.augment_time_domain_shift_frac = float(augment_time_domain_shift_frac)
        self.augment_segment_recombination = bool(augment_segment_recombination)
        self.augment_segment_recombination_prob = float(augment_segment_recombination_prob)
        self.augment_segment_recombination_n_segments = int(augment_segment_recombination_n_segments)
        self.augment_channel_dropout = bool(augment_channel_dropout)
        self.augment_channel_dropout_prob = float(augment_channel_dropout_prob)
        self.augment_channel_dropout_frac = float(augment_channel_dropout_frac)
        self.augment_mixup = bool(augment_mixup)
        self.augment_mixup_prob = float(augment_mixup_prob)
        self.augment_gaussian_noise = bool(augment_gaussian_noise)
        self.augment_gaussian_noise_prob = float(augment_gaussian_noise_prob)
        self.augment_gaussian_noise_snr_db = float(augment_gaussian_noise_snr_db)
        self.seed = int(seed)

        if not (0.0 <= self.augment_pink_noise_prob <= 1.0):
            raise ValueError("augment_pink_noise_prob must be in [0,1]")
        if not (0.0 < self.augment_time_domain_crop_frac <= 1.0):
            raise ValueError("augment_time_domain_crop_frac must be in (0, 1]")
        if not (0.0 <= self.augment_segment_recombination_prob <= 1.0):
            raise ValueError("augment_segment_recombination_prob must be in [0,1]")
        if self.augment_segment_recombination_n_segments < 2:
            raise ValueError("augment_segment_recombination_n_segments must be >= 2")
        if not (0.0 <= self.augment_channel_dropout_prob <= 1.0):
            raise ValueError("augment_channel_dropout_prob must be in [0,1]")
        if not (0.0 <= self.augment_channel_dropout_frac < 1.0):
            raise ValueError("augment_channel_dropout_frac must be in [0, 1)")
        if not (0.0 <= self.augment_mixup_prob <= 1.0):
            raise ValueError("augment_mixup_prob must be in [0,1]")
        if not (0.0 <= self.augment_gaussian_noise_prob <= 1.0):
            raise ValueError("augment_gaussian_noise_prob must be in [0,1]")

        self.label_names: Tuple[str, ...] = tuple(str(x).strip().lower() for x in label_names)
        self.label_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.label_names)}

        if not self.edf_paths:
            raise ValueError("No EDF files provided")
        if self.window_samples <= 0 or self.stride_samples <= 0:
            raise ValueError("Computed window/stride samples must be > 0")

        self._index: List[_WindowIndex] = []
        self._cache: Dict[int, np.ndarray] = {}

        self._aug_rng: np.random.Generator | None = None
        self._aug_worker_id: int | None = None

        self._build_index()
        if not self._index:
            raise RuntimeError("No valid windows were indexed. Check channel coverage and annotations.")

    def _get_aug_rng(self) -> np.random.Generator | None:

        any_aug_enabled = (
            (bool(self.augment_pink_noise) and float(self.augment_pink_noise_prob) > 0.0)
            or (bool(self.augment_gaussian_noise) and float(self.augment_gaussian_noise_prob) > 0.0)
            or (bool(self.augment_mixup) and float(self.augment_mixup_prob) > 0.0)
            or (bool(self.augment_segment_recombination) and float(self.augment_segment_recombination_prob) > 0.0)
            or bool(self.augment_time_domain)
            or (bool(self.augment_channel_dropout) and float(self.augment_channel_dropout_prob) > 0.0)
        )
        if not any_aug_enabled:
            return None

        info = torch.utils.data.get_worker_info()
        worker_id = int(info.id) if info is not None else 0
        if (self._aug_rng is None) or (self._aug_worker_id != worker_id):
            self._aug_worker_id = worker_id
            self._aug_rng = np.random.default_rng(int(self.seed) + 10007 * worker_id)
        return self._aug_rng

    def _build_index(self) -> None:
        for file_idx, path in enumerate(self.edf_paths):
            try:
                raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
            except Exception as exc:
                warnings.warn(f"Skipping {path}: failed to open EDF ({exc})")
                continue

            file_sfreq = float(raw.info["sfreq"])
            if not np.isclose(file_sfreq, self.sfreq, atol=1e-3):
                warnings.warn(
                    f"Skipping {path}: sfreq={file_sfreq} does not match requested sfreq={self.sfreq}"
                )
                continue

            parsed = [_parse_channel_name(ch) for ch in raw.ch_names]
            base_names = {p.base for p in parsed if p.base and p.montage is None}
            avg_base_names = {p.base for p in parsed if p.base and p.montage == "AVG"}
            already_avg = bool(avg_base_names)

            if already_avg:
                n_available = sum(1 for ch in AVG17_ELECS if ch in avg_base_names)
                if n_available < 1:
                    warnings.warn(f"Skipping {path}: no usable *_AVG channels for 17ch average-reference montage")
                    continue
            else:
                n_available = sum(1 for ch in AVG17_ELECS if ch in base_names)
                if n_available < 2:
                    warnings.warn(
                        f"Skipping {path}: not enough source electrodes for 17ch average reference (have {n_available}, need >=2)"
                    )
                    continue

            n_samples = int(raw.n_times)
            if n_samples < self.window_samples:
                continue

            intervals = self._extract_annotation_intervals(raw.annotations)

            max_start = n_samples - self.window_samples
            for start in range(0, max_start + 1, self.stride_samples):
                end = start + self.window_samples
                t0 = start / self.sfreq
                t1 = end / self.sfreq
                label_mask = self._label_mask_for_window(intervals, t0=t0, t1=t1)
                self._index.append(
                    _WindowIndex(
                        file_idx=file_idx,
                        start_sample=start,
                        end_sample=end,
                        label_mask=label_mask,
                    )
                )

    def _extract_annotation_intervals(self, annotations: mne.Annotations) -> List[_AnnInterval]:
        intervals: List[_AnnInterval] = []
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            start = float(onset)
            dur = float(duration)
            if dur <= 0:
                continue

            labels = _parse_description_to_labels(str(description))
            idxs = tuple(
                sorted({self.label_to_idx[lab] for lab in labels if lab in self.label_to_idx})
            )
            if not idxs:
                continue

            intervals.append(
                _AnnInterval(
                    onset=start,
                    end=start + dur,
                    label_indices=idxs,
                )
            )
        return intervals

    def _label_mask_for_window(self, intervals: Sequence[_AnnInterval], *, t0: float, t1: float) -> int:
        mask = 0
        for ann in intervals:
            overlap = min(t1, ann.end) - max(t0, ann.onset)

            if overlap <= 0:
                continue

            if self.min_overlap_sec <= 0 and self.min_overlap_frac_of_artifact <= 0:
                pass
            else:
                sec_ok = self.min_overlap_sec > 0 and overlap >= self.min_overlap_sec
                art_dur = ann.end - ann.onset
                frac_ok = (
                    self.min_overlap_frac_of_artifact > 0
                    and art_dur > 0
                    and overlap >= self.min_overlap_frac_of_artifact * art_dur
                )
                if not (sec_ok or frac_ok):
                    continue
            for li in ann.label_indices:
                mask |= (1 << li)
        return mask

    def __len__(self) -> int:
        return len(self._index)

    def _load_avg17_array(self, file_idx: int) -> np.ndarray:
        if self.use_cache and file_idx in self._cache:
            return self._cache[file_idx]

        path = self.edf_paths[file_idx]
        raw = _load_and_clean_raw(str(path))
        if raw is None:
            raise RuntimeError(f"Failed to load EEG channels from {path}")

        already_avg = _has_avg_channels(raw.ch_names)
        raw_avg = _build_avg17_montage(raw, already_avg=already_avg)
        if raw_avg is None:
            raise RuntimeError(f"Failed to build 17ch average-reference montage from {path}")

        x = raw_avg.get_data().astype(np.float32, copy=False)
        if self.use_cache:
            self._cache[file_idx] = x
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        win = self._index[idx]
        arr = self._load_avg17_array(win.file_idx)

        x = arr[:, win.start_sample : win.end_sample]
        if x.shape[1] != self.window_samples:
            raise RuntimeError(
                f"Window sample mismatch for file index {win.file_idx}: expected {self.window_samples}, got {x.shape[1]}"
            )

        if self.normalize:
            mu = x.mean(axis=1, keepdims=True)
            sigma = x.std(axis=1, keepdims=True)
            x = (x - mu) / (sigma + 1e-6)

        rng = self._get_aug_rng()
        
        y_t = _mask_to_tensor(win.label_mask, n_labels=len(self.label_names))
        
        if rng is not None and bool(self.augment_mixup):
            if float(rng.random()) < float(self.augment_mixup_prob):
                noise_idx1 = int(rng.integers(0, len(self._index)))
                noise_idx2 = int(rng.integers(0, len(self._index)))
                
                noise_win1 = self._index[noise_idx1]
                noise_win2 = self._index[noise_idx2]
                
                noise_arr1 = self._load_avg17_array(noise_win1.file_idx)
                noise_arr2 = self._load_avg17_array(noise_win2.file_idx)
                
                x_noise1 = noise_arr1[:, noise_win1.start_sample : noise_win1.end_sample]
                x_noise2 = noise_arr2[:, noise_win2.start_sample : noise_win2.end_sample]

                if x_noise1.shape[1] != self.window_samples:
                    x_noise1 = np.pad(x_noise1, ((0, 0), (0, self.window_samples - x_noise1.shape[1])), mode="constant")
                if x_noise2.shape[1] != self.window_samples:
                    x_noise2 = np.pad(x_noise2, ((0, 0), (0, self.window_samples - x_noise2.shape[1])), mode="constant")
                
                if self.normalize:
                    mu1 = x_noise1.mean(axis=1, keepdims=True)
                    sigma1 = x_noise1.std(axis=1, keepdims=True)
                    x_noise1 = (x_noise1 - mu1) / (sigma1 + 1e-6)
                    
                    mu2 = x_noise2.mean(axis=1, keepdims=True)
                    sigma2 = x_noise2.std(axis=1, keepdims=True)
                    x_noise2 = (x_noise2 - mu2) / (sigma2 + 1e-6)
                
                x, y_t = _augment_with_noise_windows(
                    x,
                    y_t.numpy(),
                    x_noise1,
                    x_noise2,
                    rng,
                )
                y_t = torch.from_numpy(y_t)
        
        if rng is not None and bool(self.augment_segment_recombination):
            if float(rng.random()) < float(self.augment_segment_recombination_prob):
                x = _segment_recombination_phase_aware(
                    x,
                    rng,
                    n_segments=int(self.augment_segment_recombination_n_segments),
                )

        if rng is not None and bool(self.augment_time_domain):
            x = _time_domain_transform(
                x,
                rng,
                crop_frac=float(self.augment_time_domain_crop_frac),
                shift_frac=float(self.augment_time_domain_shift_frac),
            )

        if rng is not None and bool(self.augment_channel_dropout):
            if float(rng.random()) < float(self.augment_channel_dropout_prob):
                x = _channel_dropout(
                    x,
                    rng,
                    frac=float(self.augment_channel_dropout_frac),
                )

        if rng is not None and bool(self.augment_pink_noise) and float(self.augment_pink_noise_prob) > 0.0:
            if float(rng.random()) < float(self.augment_pink_noise_prob):
                if not self.normalize:
                    x = np.array(x, dtype=np.float32, copy=True)
                x = _add_pink_noise_snr_db(x, rng, snr_db=float(self.augment_pink_noise_snr_db))

        if rng is not None and bool(self.augment_gaussian_noise) and float(self.augment_gaussian_noise_prob) > 0.0:
            if float(rng.random()) < float(self.augment_gaussian_noise_prob):
                if not self.normalize:
                    x = np.array(x, dtype=np.float32, copy=True)
                x = _add_gaussian_noise_snr_db(x, rng, snr_db=float(self.augment_gaussian_noise_snr_db))

        x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return x_t, y_t


def build_file_list(split_dir: Path | str) -> List[Path]:
    split_path = Path(split_dir)
    return sorted(split_path.glob("**/*.edf"))


def make_dataloader(
    split_dir: Path | str,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    dataset_kwargs: dict | None = None,
) -> DataLoader:
    files = build_file_list(split_dir)
    if not files:
        raise ValueError(f"No EDF files found under {split_dir}")

    ds = EEGWindowDataset(files, **(dataset_kwargs or {}))

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def make_split_dataloaders(
    split_root: Path | str,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    train_shuffle: bool = True,
    dataset_kwargs: dict | None = None,
) -> Dict[str, DataLoader]:
    root = Path(split_root)
    loaders = {
        "train": make_dataloader(
            root / "train" / "01_tcp_ar",
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            dataset_kwargs=dataset_kwargs,
        ),
        "dev": make_dataloader(
            root / "dev" / "01_tcp_ar",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            dataset_kwargs=dataset_kwargs,
        ),
        "test": make_dataloader(
            root / "test" / "01_tcp_ar",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            dataset_kwargs=dataset_kwargs,
        ),
    }
    return loaders
