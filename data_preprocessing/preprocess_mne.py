#!/usr/bin/env python3
"""
TUSZ / TUEG preprocessing (lean)
--------------------------------
This streamlined pipeline:
  • Loads and cleans EDFs (EEG-only; finite samples).
  • Trims constant edges (head/tail) and keeps first/last non-constant sample.
  • Band-pass filters (HP/LP), optional notch via spectrum fit.
  • Optional resampling to `resample_freq` (default 250 Hz).
  • Builds selected montage (TCP / Avg / Cz / original subset).
  • Robust global scaling (middle-50% → µV).
  • Plots per-file amplitude histogram + GN fit and saves under stats/file_histograms/.
  • Hard-clips samples to a symmetric threshold (fitted or user-provided µV).
  • (Optional) Detects artifacts; attaches as MNE Annotations.
  • Loads seizure annotations from configured CSV roots; attaches as MNE Annotations.
  • Exports preprocessed EDF (with annotations) mirroring the original split/path.
  • Writes a single CSV with basic per-file preprocessing stats.

CSV inputs for seizures:
  - Required columns: label, start_time, stop_time  (case-insensitive).
  - Optional column:  channel
      If present → channel-specific mapping to the target montage.
      If missing  → apply intervals to all EEG channels in the target montage.

Annotation description format in EDF:
  - "{channel}:{label}"   e.g., "F7-T3_TCP:sz", "C3_AVG:Artifact", or "C3_AVG:Flatline".
  - Artifact detector (by default) emits "{channel}:Artifact".
    If `separate_annotations=true` in artifact_config, labels like "Flatline", "LowVariance", etc. are used.

NOTE:
  - No dataset-level statistics, no Venns/metrics; only per-file histograms and one CSV.
"""

from __future__ import annotations
import glob, logging, os, re, signal, argparse, yaml, warnings
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import fire, mne, numpy as np, pandas as pd
from scipy import stats
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from artifact_detection import ComprehensiveArtifactDetector
from histogram_plotter import HistogramPlotter

"""
Label Decomposition and Handling
================================
Base label vocabulary for multi-label classification. These are the fundamental
annotation types recognized by the system.
"""
_BASE_LABELS = {
    "artifact",
    "bad_acq_skip",
    "chew",
    "elec",
    "elpp",
    "eyem",
    "musc",
    "shiv",
}

"""
Labels containing underscores that should NOT be split into components.
"""
_NON_SPLIT_LABELS = {"bad_acq_skip"}


def _decompose_label(label: str) -> List[str]:
    """
    Decompose a label into its base components if it's a composite.
    
    Parameters
    ----------
    label : str
        Annotation label (possibly composite like 'eyem_musc')
    
    Returns
    -------
    list[str]
        List of base label components. Empty composites return original label.
    """
    normalized_label = str(label).strip().lower()
    if ("_" not in normalized_label) or (normalized_label in _NON_SPLIT_LABELS):
        return [normalized_label]

    label_parts = [p for p in normalized_label.split("_") if p]
    if len(label_parts) <= 1:
        return [normalized_label]

    if all(p in _BASE_LABELS for p in label_parts):
        return label_parts

    return [normalized_label]


def decompose_composite_labels(annotations: mne.Annotations) -> mne.Annotations:
    """
    Replace composite labels like 'eyem_musc' with overlapping base labels.

    Expands a single composite annotation into multiple annotations with the
    same (onset, duration, channel) but different base labels. Maintains
    EDF compatibility (one label per annotation row).
    
    Parameters
    ----------
    annotations : mne.Annotations
        Input annotations possibly containing composite labels
    
    Returns
    -------
    mne.Annotations
        Annotations with composite labels decomposed into base components
    """
    try:
        num_annotations = len(annotations)
    except Exception:
        return annotations

    if num_annotations <= 0:
        return annotations

    output_onsets: List[float] = []
    output_durations: List[float] = []
    output_descriptions: List[str] = []

    labels_were_decomposed = False

    for i in range(num_annotations):
        onset = float(annotations.onset[i])
        duration = float(annotations.duration[i])
        if duration <= 0:
            continue

        description = str(annotations.description[i])
        if ":" in description:
            channel, label = description.split(":", 1)
            channel = channel.strip()
            label = label.strip().lower()
        else:
            channel, label = "", description.strip().lower()

        decomposed_labels = _decompose_label(label)
        if len(decomposed_labels) != 1 or decomposed_labels[0] != label:
            labels_were_decomposed = True

        for output_label in decomposed_labels:
            output_onsets.append(onset)
            output_durations.append(duration)
            output_descriptions.append(f"{channel}:{output_label}" if channel else output_label)

    if not labels_were_decomposed:
        return annotations

    output_annotations = mne.Annotations(output_onsets, output_durations, output_descriptions, orig_time=annotations.orig_time)
    logger.info(f"Decomposed composite labels in annotations: {num_annotations} → {len(output_annotations)}")
    return output_annotations

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
def _init_worker(): signal.signal(signal.SIGINT, signal.SIG_IGN)

"""
TCP (Temporal-Central-Parietal) Bipolar Montage Definition
===========================================================
Standard 10-20 electrode pairs for bipolar montage (uppercase conventions).
"""
TCP_PAIRS: List[Tuple[str, str]] = [
    ("FP1","F7"),("F7","T3"),("T3","T5"),("T5","O1"),
    ("FP2","F8"),("F8","T4"),("T4","T6"),("T6","O2"),
    ("T3","C3"),("C3","CZ"),("CZ","C4"),("C4","T4"),
    ("FP1","F3"),("F3","C3"),("C3","P3"),("P3","O1"),
    ("FP2","F4"),("F4","C4"),("C4","P4"),("P4","O2"),
]
ALL_ELECS = {e for p in TCP_PAIRS for e in p}

"""
Maximum EDF physical field value (9,999,999 fits in 8 chars with sign).
"""
MAX_FIELD_UV = 9_999_999

"""
Laplacian Montage Definition (19-channel 10-20 System)
=======================================================
Standard electrode positions for Laplacian reference montage.
"""
LAPLACIAN_ELECS: List[str] = [
    "FP1","FP2","F7","F3","FZ","F4","F8","T3","C3","CZ","C4","T4","T5","P3","PZ","P4","T6","O1","O2"
]

"""
Laplacian Neighbor Mapping
==========================
For each electrode, lists the neighboring electrodes used in Laplacian calculation.
"""
LAPLACIAN_NEIGHBORS: Dict[str, List[str]] = {
    "FP1": ["FP2", "F7", "F3"],
    "FP2": ["FP1", "F4", "F8"],

    "F7":  ["FP1", "F3", "T3"],
    "F3":  ["FP1", "F7", "FZ", "C3"],
    "FZ":  ["F3", "F4", "CZ"],
    "F4":  ["FP2", "FZ", "F8", "C4"],
    "F8":  ["FP2", "F4", "T4"],

    "T3":  ["F7", "C3", "T5"],
    "C3":  ["F3", "CZ", "T3", "P3"],
    "CZ":  ["FZ", "C3", "C4", "PZ"],
    "C4":  ["F4", "CZ", "T4", "P4"],
    "T4":  ["F8", "C4", "T6"],

    "T5":  ["T3", "P3", "O1"],
    "P3":  ["C3", "PZ", "T5", "O1"],
    "PZ":  ["CZ", "P3", "P4"],
    "P4":  ["C4", "PZ", "T6", "O2"],
    "T6":  ["T4", "P4", "O2"],

    "O1":  ["T5", "P3", "O2"],
    "O2":  ["T6", "P4", "O1"],
}

"""
BENDR UI 10/20 Electrode Order
===============================
Fixed UI 10/20 electrode ordering (19 channels) commonly used in Deep1010-like mappings.
"""
BENDR_UI1020_ELECS: List[str] = [
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"
]
"""
TUH Legacy to UI 10/20 Channel Name Mapping
============================================
Legacy TUH electrode labels (T3/T4/T5/T6) to modern UI 10/20 equivalents (T7/T8/P7/P8).
"""
_TUH_LEGACY_TO_UI = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
_TUH_UI_TO_LEGACY = {v: k for k, v in _TUH_LEGACY_TO_UI.items()}

def _map_legacy_to_ui(ch: str) -> str:
    return _TUH_LEGACY_TO_UI.get(ch, ch)

def _map_ui_to_legacy(ch: str) -> str:
    return _TUH_UI_TO_LEGACY.get(ch, ch)

@dataclass
class FileStats:
    """Minimal statistics recorded per file."""
    filename: str
    subset: str = ""              
    orig_sfreq: int = 0
    rec_length_sec: float = 0.0
    trimmed_lead_sec: float = 0.0
    trimmed_tail_sec: float = 0.0
    mean_val: float = 0.0         
    std_val: float = 0.0         
    clip_threshold: float = 0.0   
    clipped_samples: int = 0
    total_samples: int = 0
    clipping_percent: float = 0.0
    annotations_present: List[str] = field(default_factory=list)

def sample_paths(paths: Sequence[str], pct: int, seed: int = 42) -> List[str]:
    """
    Sample a percentage of file paths, stratified by subset (train/dev/eval/test).
    
    Parameters
    ----------
    paths : sequence[str]
        File paths to sample from
    pct : int
        Percentage of files to keep (100 = all)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    list[str]
        Sampled file paths
    """
    if pct >= 100:
        return list(paths)
    rng = np.random.default_rng(seed)
    subset_buckets, output = {}, []
    for path in paths:
        for subset_name in ("train","dev","eval","test"):
            if f"{os.sep}{subset_name}{os.sep}" in path.lower():
                subset_buckets.setdefault(subset_name, []).append(path)
                break
    for files_in_subset in subset_buckets.values():
        sample_count = max(1, int(len(files_in_subset)*pct/100))
        output += rng.choice(files_in_subset, sample_count, False).tolist()
    return output

def clean(channel_name: str) -> str:
    """
    Canonicalize channel names to simple electrode labels (e.g., 'F7', 'CZ').
    
    Removes EEG prefix, standardizes case, removes reference indicators.
    """
    normalized_name = channel_name.upper()
    normalized_name = re.sub(r'\bEEG\b\s*', '', normalized_name)
    normalized_name = normalized_name.strip()
    normalized_name = re.sub(r'\s+', '', normalized_name)
    normalized_name = re.sub(r'[-._\s]?REF$', '', normalized_name)
    normalized_name = re.sub(r'[-._\s]?LE$', '', normalized_name)
    return normalized_name


def load_and_clean_raw(edf_path: str) -> Optional[mne.io.Raw]:
    """
    Load EDF file and perform basic channel cleaning.
    
    - Canonicalizes channel names
    - Extracts EEG channels only
    - Replaces non-finite values with 0
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    channel_mapping = {}
    seen_names = set()
    for ch in raw.ch_names:
        canonical_name = clean(ch)
        if canonical_name and canonical_name not in seen_names:
            channel_mapping[ch] = canonical_name
            seen_names.add(canonical_name)
    raw.rename_channels(channel_mapping)

    raw.pick(mne.pick_types(raw.info, eeg=True))
    if not raw.ch_names:
        logger.warning(f"{Path(edf_path).name}: no EEG channels, skipping")
        return None

    finite_mask = np.isfinite(raw._data)
    if not finite_mask.all():
        non_finite_count = (~finite_mask).sum()
        raw._data[~finite_mask] = 0.0
        logger.debug(f"{Path(edf_path).name}: replaced {int(non_finite_count)} non-finite samples with 0")

    return raw

def create_tcp_montage(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    """Create TCP montage from raw data."""
    missing = ALL_ELECS.difference(raw.ch_names)
    if missing:
        missing_list = sorted(missing)
        present_channels = sorted(raw.ch_names)
        logger.warning(
            f"{Path(raw.info['filenames'][0]).name}: missing TCP electrodes {missing_list}; EEG channels present: {present_channels}; skipping"
        )
        return None

    anodes, cathodes = zip(*TCP_PAIRS)
    ch_names = [f"{a}-{c}_TCP" for a, c in TCP_PAIRS]
    raw_tcp = mne.set_bipolar_reference(
        raw, anode=list(anodes), cathode=list(cathodes),
        ch_name=ch_names, drop_refs=True, copy=True, verbose=False
    ).pick(ch_names)

    return raw_tcp

def _create_avg_ref_montage(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    """Create Average-reference montage over ALL_ELECS present; channels suffixed _AVG."""
    picks = sorted(ALL_ELECS.intersection(raw.ch_names))
    if len(picks) < 2:
        logger.warning(f"{Path(raw.info['filenames'][0]).name}: not enough electrodes for Avg reference; skipping")
        return None
    r = raw.copy().pick(picks)
    mne.set_eeg_reference(r, ref_channels='average', projection=False, verbose=False)
    r.rename_channels({ch: f"{ch}_AVG" for ch in r.ch_names})
    return r

def _create_cz_ref_montage(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    """Create Cz-reference montage over ALL_ELECS present (requires CZ); channels suffixed _CZ."""
    if "CZ" not in raw.ch_names:
        logger.warning(f"{Path(raw.info['filenames'][0]).name}: CZ channel missing for Cz reference; skipping")
        return None
    picks = sorted(ALL_ELECS.intersection(raw.ch_names))
    if len(picks) < 2:
        logger.warning(f"{Path(raw.info['filenames'][0]).name}: not enough electrodes for Cz reference; skipping")
        return None
    r = raw.copy().pick(picks)
    mne.set_eeg_reference(r, ref_channels=["CZ"], projection=False, verbose=False)
    r.rename_channels({ch: f"{ch}_CZ" for ch in r.ch_names})
    return r

def _create_laplacian_montage(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    """Create Laplacian montage on a standard 19-channel 10–20 grid; channels suffixed _LAP."""
    present = [ch for ch in LAPLACIAN_ELECS if ch in raw.ch_names]
    if len(present) < 2:
        _fn = "unknown"
        try:
            if getattr(raw, "filenames", None) and len(raw.filenames) > 0 and raw.filenames[0]:
                _fn = Path(raw.filenames[0]).name
        except Exception:
            _fn = "unknown"
        logger.warning(f"{_fn}: not enough Laplacian electrodes present ({present}); skipping")
        return None

    r = raw.copy().pick_channels(present, ordered=True)

    idx = {ch: i for i, ch in enumerate(present)}
    n = len(present)
    L = np.zeros((n, n), dtype=float)

    kept = []
    for ch in present:
        nbrs_all = LAPLACIAN_NEIGHBORS.get(ch, [])
        nbrs = [nb for nb in nbrs_all if nb in idx]
        if len(nbrs) < 2:
            continue
        i = idx[ch]
        L[i, i] = 1.0
        w = 1.0 / float(len(nbrs))
        for nb in nbrs:
            j = idx[nb]
            L[i, j] -= w
        kept.append(ch)

    if not kept:
        _fn = "unknown"
        try:
            if getattr(raw, "filenames", None) and len(raw.filenames) > 0 and raw.filenames[0]:
                _fn = Path(raw.filenames[0]).name
        except Exception:
            _fn = "unknown"
        logger.warning(f"{_fn}: no channels had sufficient Laplacian neighbors; skipping")
        return None

    r._data = L @ r._data
    r = r.pick_channels(kept, ordered=True)

    r.rename_channels({ch: f"{ch}_LAP" for ch in r.ch_names})
    return r

def _create_bendr_ui1020_avg_montage(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    """
    BENDR-compatible: 19 UI10/20 channels in fixed order, average reference, suffix _AVG.
    TUH legacy names are supported as sources (T3/T4/T5/T6 used when T7/T8/P7/P8 are absent).
    Missing channels are zero-filled (so output always has 19 channels).
    """
    target_to_src: Dict[str, str] = {}
    for tgt in BENDR_UI1020_ELECS:
        if tgt in raw.ch_names:
            target_to_src[tgt] = tgt
            continue
        legacy = _map_ui_to_legacy(tgt)
        if legacy in raw.ch_names:
            target_to_src[tgt] = legacy

    src_picks = list(dict.fromkeys(target_to_src.values()))  
    if len(src_picks) < 2:
        logger.warning(f"{Path(raw.info['filenames'][0]).name}: not enough UI10/20 electrodes for BENDR_UI1020_AVG; skipping")
        return None

    r = raw.copy().pick_channels(src_picks, ordered=True)
    mne.set_eeg_reference(r, ref_channels="average", projection=False, verbose=False)

    rename_src_to_ui = {src: tgt for tgt, src in target_to_src.items() if src != tgt}
    if rename_src_to_ui:
        r.rename_channels(rename_src_to_ui)

    r.rename_channels({ch: f"{ch}_AVG" for ch in r.ch_names})

    desired = [f"{ch}_AVG" for ch in BENDR_UI1020_ELECS]
    present_set = set(r.ch_names)
    missing = [ch for ch in desired if ch not in present_set]
    if missing:
        info = mne.create_info(ch_names=missing, sfreq=r.info["sfreq"], ch_types=["eeg"] * len(missing))
        zeros = np.zeros((len(missing), r.n_times), dtype=r._data.dtype)
        r_missing = mne.io.RawArray(zeros, info, verbose=False)
        r.add_channels([r_missing], force_update_info=True)

    r = r.pick_channels(desired, ordered=True)
    return r


def _create_original_selection(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    """Keep original signals but restrict to ALL_ELECS present (no suffix)."""
    picks = sorted(ALL_ELECS.intersection(raw.ch_names))
    if len(picks) < 1:
        logger.warning(f"{Path(raw.info['filenames'][0]).name}: no electrodes from ALL_ELECS present; skipping")
        return None
    return raw.copy().pick(picks)

def create_selected_montage(raw: mne.io.Raw, montage: Optional[Union[str, List[str]]]) -> Optional[mne.io.Raw]:
    """
    Build the requested montage(s).
    - montage == None: keep original montage, but pick ALL_ELECS.
    - montage == "tcp" / "avg" / "cz" / "lap": build that montage.
    - montage == "bendr_ui1020_avg": build fixed UI10/20 19ch avg-ref montage (zero-fill missing).
    - montage is a list: concatenate the requested montages along the channel axis.
    """
    if montage is None:
        return _create_original_selection(raw)

    def _build_one(kind: str) -> Optional[mne.io.Raw]:
        k = kind.lower()
        if k == "tcp": return create_tcp_montage(raw)
        if k == "avg": return _create_avg_ref_montage(raw)
        if k == "cz":  return _create_cz_ref_montage(raw)
        if k == "lap" or k == "laplacian": return _create_laplacian_montage(raw)
        if k == "bendr_ui1020_avg": return _create_bendr_ui1020_avg_montage(raw)
        logger.warning(f"Unknown montage '{kind}' (accepted: TCP, Avg, Cz, Laplacian, BENDR_UI1020_AVG); skipping")
        return None

    if isinstance(montage, str):
        return _build_one(montage)

    outs: List[mne.io.Raw] = []
    for kind in montage:
        r = _build_one(kind)
        if r is not None:
            outs.append(r)

    if not outs:
        return None

    combined = outs[0]
    for r in outs[1:]:
        combined.add_channels([r], force_update_info=True)
    return combined

def apply_robust_scaling(raw: mne.io.Raw) -> float:
    """
    Apply global robust middle-50% scaling; convert to µV.
    
    Uses the middle 50% of absolute values to estimate a robust scale factor,
    avoiding outliers. Modifies raw data in place.
    
    Returns
    -------
    float
        Scale factor applied (in µV units)
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    middle_50_values = []
    for channel_idx in eeg_picks:
        absolute_values = np.abs(raw._data[channel_idx])
        percentile_25, percentile_75 = np.percentile(absolute_values, [25, 75])
        mask_middle = (absolute_values >= percentile_25) & (absolute_values <= percentile_75)
        if mask_middle.any():
            middle_50_values.append(absolute_values[mask_middle].mean())

    scale_factor = np.mean(np.clip(middle_50_values, *np.percentile(middle_50_values, [25, 75]))) if middle_50_values else 1.0
    if not np.isfinite(scale_factor) or scale_factor == 0:
        scale_factor = 1.0

    scale_factor *= 1e5
    raw._data /= scale_factor
    return scale_factor

def apply_clipping(raw: mne.io.Raw, clip_threshold: float) -> Tuple[int, int, float, float]:
    """
    Apply clipping based on clip_threshold (data already in µV).
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    clip_threshold : float
        Clipping threshold (in Volts internally)
    
    Returns
    -------
    tuple
        (clipped_sample_count, total_sample_count, min_threshold, max_threshold)
    """
    physical_max = min(clip_threshold, MAX_FIELD_UV)
    physical_min = -physical_max

    total_samples = raw._data.size
    clipping_mask = (raw._data > physical_max) | (raw._data < physical_min)
    clipped_count = int(clipping_mask.sum())

    if clipped_count:
        raw._data = np.clip(raw._data, physical_min, physical_max)

    return clipped_count, total_samples, physical_min, physical_max

def _detect_target_montage_from_raw(raw: mne.io.Raw) -> str:
    """
    Detect montage type from channel name suffixes in raw data.
    
    Returns montage name (TCP, AVG, CZ, LAP) or empty string if none detected.
    """
    channel_names = raw.ch_names
    if any(n.endswith("_TCP") for n in channel_names):
        return "TCP"
    if any(n.endswith("_AVG") for n in channel_names):
        return "AVG"
    if any(n.endswith("_CZ") for n in channel_names):
        return "CZ"
    if any(n.endswith("_LAP") for n in channel_names):
        return "LAP"
    return ""

def _parse_source_montage_from_csv_header(csv_path: str) -> str:
    """
    Extract source montage name from CSV file header comments.
    
    Looks for 'montage_file = ...' in comment lines.
    """
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.lstrip().startswith("#"):
                    break
                match = re.search(r"montage_file\s*=\s*(.+)", line, flags=re.I)
                if match:
                    return os.path.basename(match.group(1).strip())
    except Exception:
        pass
    return ""

"""
EAR Reference Electrode Labels
===============================
Common ear/mastoid electrode labels that may appear in recordings.
"""
_EAR_LABELS = {"A1", "A2", "M1", "M2"}

def _incident_tcp_channels(electrode: str, channel_names: Sequence[str]) -> List[str]:
    """
    Find all TCP bipolar pairs incident to the given electrode.
    
    Parameters
    ----------
    electrode : str
        Electrode to find incident pairs for
    channel_names : sequence[str]
        List of available channel names
    
    Returns
    -------
    list[str]
        TCP channel names containing this electrode
    """
    matching_channels = []
    prefix = f"{electrode}-"
    suffix = f"-{electrode}_TCP"
    for channel_name in channel_names:
        if not channel_name.endswith("_TCP"):
            continue
        if channel_name.startswith(prefix) or channel_name.endswith(suffix):
            matching_channels.append(channel_name)
    return matching_channels

def _map_channel_to_target(ch_clean: str, raw: mne.io.Raw, target: str) -> List[str]:
    """
    Returns a list of target channel names (existing in raw) corresponding to ch_clean.
    TERM is handled by caller. If no mapping is found, return [].
    """
    names = set(raw.ch_names)

    if target == "TCP" and ch_clean.endswith("_TCP") and ch_clean in names:
        return [ch_clean]

    if target == "TCP":
        if "-" in ch_clean:
            a, b = ch_clean.split("-", 1)
            cand1 = f"{a}-{b}_TCP"
            if cand1 in names:
                return [cand1]
            cand2 = f"{b}-{a}_TCP"
            if cand2 in names:
                return [cand2]

            if a in _EAR_LABELS and b not in _EAR_LABELS:
                inc = _incident_tcp_channels(b, names)
                if inc: return inc
            if b in _EAR_LABELS and a not in _EAR_LABELS:
                inc = _incident_tcp_channels(a, names)
                if inc: return inc

            inc_a = _incident_tcp_channels(a, names)
            inc_b = _incident_tcp_channels(b, names)
            if inc_a or inc_b:
                both = [n for n in set(inc_a) & set(inc_b)]
                return both if both else list(dict.fromkeys(inc_a + inc_b))  # keep order, dedupe

            return []  

        else:
            inc = _incident_tcp_channels(ch_clean, names)
            return inc

    suffix = f"_{target}" if target in ("AVG","CZ","LAP") else ""

    def _cands_one(e: str) -> List[str]:
        outs = []
        cand = f"{e}{suffix}"
        if cand in names:
            outs.append(cand)
            return outs
        e_ui = _map_legacy_to_ui(e)
        if e_ui != e:
            cand2 = f"{e_ui}{suffix}"
            if cand2 in names:
                outs.append(cand2)
        return outs

    if "-" in ch_clean:
        a, b = ch_clean.split("-", 1)
        if a in _EAR_LABELS and b not in _EAR_LABELS:
            outs = _cands_one(b)
            return outs
        if b in _EAR_LABELS and a not in _EAR_LABELS:
            outs = _cands_one(a)
            return outs
        outs = []
        outs += _cands_one(a)
        outs += _cands_one(b)
        return list(dict.fromkeys(outs))
    else:
        return _cands_one(ch_clean)

def load_seizure_annotations(
    edf_path: str, input_root: str,
    annotations_config: Dict[str, str],
    raw: mne.io.Raw,
    time_shift_sec: float = 0.0,
    binary: bool = False
) -> mne.Annotations:
    """Load seizure annotations from configured folders; returns one MNE Annotations object."""
    onsets: List[float]; durations: List[float]; descriptions: List[str]
    onsets, durations, descriptions = [], [], []

    if not annotations_config:
        return mne.Annotations([], [], [])

    rel = os.path.relpath(edf_path, input_root)
    rec_length_sec = len(raw.times) / raw.info['sfreq']
    target_mont = _detect_target_montage_from_raw(raw)
    src_mont_found = ""

    for _name, ann_root in annotations_config.items():
        csv_path = os.path.join(
            ann_root,
            re.sub(r"\.edf$", ".csv_bi" if binary else ".csv", rel, flags=re.I)
        )
        if not os.path.exists(csv_path):
            logger.warning(f"Missing annotation for {Path(edf_path).name} under '{_name}': {csv_path}")
            continue

        try:
            if not src_mont_found:
                src_mont_found = _parse_source_montage_from_csv_header(csv_path)

            df = pd.read_csv(csv_path, comment="#")
            colmap = {c.lower(): c for c in df.columns}
            required = ["label", "start_time", "stop_time"]
            if not all(r in colmap for r in required):
                logger.warning(f"{csv_path}: required columns {required} not all present; skipping file.")
                continue

            has_channel = "channel" in colmap

            for _, r in df.iterrows():
                label = str(r[colmap["label"]]).strip().lower()

                try:
                    start = float(r[colmap["start_time"]]) - time_shift_sec
                    stop  = float(r[colmap["stop_time"]])  - time_shift_sec
                except Exception:
                    logger.warning(f"Invalid times in {csv_path} row → skipping row.")
                    continue

                if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
                    continue

                if stop <= 0 or start >= rec_length_sec:
                    continue
                start = max(0.0, start)
                stop  = min(rec_length_sec, stop)
                dur = stop - start
                if dur <= 0:
                    continue

                if has_channel:
                    raw_ch = r[colmap["channel"]]
                    raw_ch_str = "" if pd.isna(raw_ch) else str(raw_ch)
                    ch_clean = clean(raw_ch_str) if raw_ch_str else "TERM"
                else:
                    ch_clean = "TERM"

                if ch_clean == "TERM":
                    target_channels = list(raw.ch_names)
                else:
                    mapped = _map_channel_to_target(ch_clean, raw, target_mont)
                    if mapped:
                        target_channels = mapped
                    else:
                        target_channels = list(raw.ch_names)
                        msg = (f"{Path(edf_path).name}: could not map CSV channel '{raw_ch_str or ''}' "
                               f"(cleaned '{ch_clean}', src_montage='{src_mont_found or '?'}' → target='{target_mont or 'plain'}'); "
                               f"applying annotation '{label}' to ALL channels.")
                        logger.warning(msg)
                        warnings.warn(msg)

                for ch in target_channels:
                    onsets.append(start)
                    durations.append(dur)
                    descriptions.append(f"{ch}:{label}")

        except Exception as e:
            logger.warning(f"Annotation {csv_path}: {e}")

    return mne.Annotations(onsets, durations, descriptions) if onsets else mne.Annotations([], [], [])

def export_edf(raw: mne.io.Raw, edf_path: str, input_root: str, out_root: str):
    """
    Export the processed EDF file, mirroring the input split/folders.

    Requirements satisfied:
    - Keep the split directory (train/dev/eval/test) — 'eval' routed as 'test' for layout.
    - Preserve the original subfolder structure UNDER that split.
    - Do NOT append suffixes; keep the original filename.
    """
    subset = next((s for s in ("train","dev","eval","test") if f"{os.sep}{s}{os.sep}" in edf_path.lower()), None)
    if subset is None:
        subset = "test"
    if subset == "eval":
        subset = "test"

    abs_input_root = os.path.abspath(input_root)
    abs_edf = os.path.abspath(edf_path)
    subset_root = os.path.join(abs_input_root, subset)

    if abs_edf.startswith(subset_root + os.sep):
        rel_under_subset = Path(os.path.relpath(abs_edf, subset_root)).parent
    else:
        rel_under_subset = Path(os.path.relpath(abs_edf, abs_input_root)).parent

    out_dir = Path(out_root) / subset / rel_under_subset
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / Path(edf_path).name
    raw.set_meas_date(None)  
    raw.export(str(out_file), fmt="edf", overwrite=True, verbose=False)

def trim_constant_edges(raw, tolerance=0.0):
    """
    Trim leading/trailing samples where ALL EEG channels are constant.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data to trim
    tolerance : float
        Tolerance for constant detection in Volts (0.0 = exact equality)
    
    Returns
    -------
    raw : mne.io.Raw
        Trimmed raw data
    lead_sec : float
        Duration of trimmed leading section (seconds)
    tail_sec : float
        Duration of trimmed trailing section (seconds)
    """
    data = _create_original_selection(raw).get_data(picks='eeg')
    sample_rate = raw.info['sfreq']
    num_samples = data.shape[1]

    is_start_constant = np.max(np.abs(data - data[:, :1]), axis=0) <= tolerance
    is_end_constant = np.max(np.abs(data - data[:, -1:]), axis=0) <= tolerance

    leading_constant_run = np.argmax(~is_start_constant) if (~is_start_constant).any() else num_samples
    leading_samples = max(0, leading_constant_run - 1)

    trailing_constant_run = np.argmax(~is_end_constant[::-1]) if (~is_end_constant).any() else num_samples
    trailing_samples = max(0, trailing_constant_run - 1)

    if leading_samples + trailing_samples >= num_samples:
        raise RuntimeError("Entire recording is constant at this tolerance.")

    leading_sec = leading_samples / sample_rate
    trailing_sec = trailing_samples / sample_rate

    if leading_samples or trailing_samples:
        raw.crop(tmin=leading_sec, tmax=(num_samples - trailing_samples - 1) / sample_rate)
    return raw, leading_sec, trailing_sec

def _coerce_fit_results_dict(obj):
    """
    Extract fit results dictionary from various return types.
    
    Handles dict, list, or tuple containers. Returns first dict found or None.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        for element in obj:
            if isinstance(element, dict):
                return element
    return None

def _has_annotations(annotations) -> bool:
    """
    Safely check if annotations object contains any entries.
    """
    try:
        return annotations is not None and len(annotations) > 0
    except Exception:
        return False

def process_one(arg) -> Optional[FileStats]:
    (
        edf_path, input_root, out_root, sample_rates, resample_freq, annotations,
        stats_dir, target_central_mass, do_artifact_detection, artifact_config,
        montage_spec, high_pass, low_pass, cfg_clip_threshold, robust_scaling,
        decompose_composite_labels_flag
    ) = arg

    file_stats = FileStats(filename=Path(edf_path).name)

    subset = next((s for s in ("train","dev","eval","test") if f"{os.sep}{s}{os.sep}" in edf_path.lower()), None)
    if subset is None:
        subset = "test"
    if subset == "eval":
        subset = "test"
    file_stats.subset = subset

    raw = load_and_clean_raw(edf_path)
    if raw is None:
        return None

    raw, trimmed_lead_sec, trimmed_tail_sec = trim_constant_edges(raw, tolerance=0.0)
    file_stats.trimmed_lead_sec = float(trimmed_lead_sec)
    file_stats.trimmed_tail_sec = float(trimmed_tail_sec)

    try:
        raw.filter(l_freq=high_pass, h_freq=low_pass, verbose=False)
    except Exception as e:
        logger.warning(f"{Path(edf_path).name}: filtering failed (l={high_pass}, h={low_pass}): {e}")

    raw.notch_filter(freqs=None, method='spectrum_fit')

    orig_sf = int(raw.info["sfreq"])
    file_stats.orig_sfreq = orig_sf

    if sample_rates is not None:
        sr_list = sample_rates.tolist() if hasattr(sample_rates, "tolist") else sample_rates
        if len(sr_list) > 0 and orig_sf not in set(sr_list):
            logger.warning(f"{Path(edf_path).name}: sfreq {orig_sf} not in {sr_list!r}, skipping")
            return None

    if orig_sf != resample_freq:
        raw.resample(resample_freq, npad="auto")

    raw = create_selected_montage(raw, montage_spec)
    if raw is None:
        return None

    rec_length_sec = len(raw.times) / raw.info['sfreq']
    file_stats.rec_length_sec = rec_length_sec

    if robust_scaling:
        _ = apply_robust_scaling(raw)

    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    plotter = HistogramPlotter(Path(stats_dir) / "file_histograms")
    amplitudes = raw._data.flatten()

    fit_results_raw = plotter.plot_amplitude_distribution(
        amplitudes=amplitudes,
        filename=Path(edf_path).stem,
        target_central_mass=target_central_mass
    )
    fit_results = _coerce_fit_results_dict(fit_results_raw)

    if cfg_clip_threshold is not None:
        try:
            user_uV = float(cfg_clip_threshold)
        except Exception:
            user_uV = 0.0
        clip_threshold_V = user_uV * 1e-6  
        file_stats.clip_threshold = user_uV  
    else:
        if (fit_results is not None) and ('clip_threshold' in fit_results):
            clip_threshold_V = float(fit_results['clip_threshold'])
            file_stats.clip_threshold = clip_threshold_V * 1e6  
        else:
            std_v = float(amplitudes.std())
            z = stats.norm.ppf(1 - (1 - target_central_mass) / 2)
            clip_threshold_V = std_v * z
            file_stats.clip_threshold = clip_threshold_V * 1e6  

    file_stats.mean_val = float(amplitudes.mean()) * 1e6  
    file_stats.std_val  = float(amplitudes.std())  * 1e6  

    clipped, before, _min, _max = apply_clipping(raw, clip_threshold_V)
    file_stats.clipped_samples = clipped
    file_stats.total_samples = before
    file_stats.clipping_percent = (clipped * 100 / before) if before > 0 else 0.0

    if do_artifact_detection:
        detector = ComprehensiveArtifactDetector(**artifact_config)
        artifact_annots, _summary = detector.detect_all_artifacts(raw)
    else:
        artifact_annots = mne.Annotations([], [], [])

    seizure_annots = load_seizure_annotations(
        edf_path, input_root, annotations or {}, raw,
        time_shift_sec=trimmed_lead_sec, binary=False
    )

    if decompose_composite_labels_flag and _has_annotations(seizure_annots):
        seizure_annots = decompose_composite_labels(seizure_annots)

    if _has_annotations(artifact_annots) and _has_annotations(seizure_annots):
        raw.set_annotations(artifact_annots + seizure_annots)
    elif _has_annotations(artifact_annots):
        raw.set_annotations(artifact_annots)
    elif _has_annotations(seizure_annots):
        raw.set_annotations(seizure_annots)

    if decompose_composite_labels_flag and _has_annotations(getattr(raw, "annotations", None)):
        try:
            raw.set_annotations(decompose_composite_labels(raw.annotations))
        except Exception as e:
            logger.warning(f"{Path(edf_path).name}: could not decompose final annotations: {e}")

    labels_present = set()

    def _acc(descs):
        if descs is None:
            return
        if hasattr(descs, "tolist"):  
            iterable = descs.tolist()
        elif isinstance(descs, (list, tuple)):
            iterable = descs
        else:
            iterable = [descs] 
        for d in iterable:
            s = str(d)
            if ":" in s:
                _, lab = s.split(":", 1)
                labels_present.add(lab.strip().lower())
            else:
                labels_present.add(s.strip().lower())

    try:
        _acc(getattr(raw.annotations, "description", []))
    except Exception:
        pass
    file_stats.annotations_present = sorted(labels_present)

    export_edf(raw, edf_path, input_root, out_root)

    return file_stats

def preprocess(
    input_path:    str,
    output_path:   str,
    percent:       int               = 100,
    sample_rates:  List[int] | None  = None,
    resample_freq: int               = 250,
    n_jobs:        int               = 4,
    seed:          int               = 42,
    annotations:   Dict[str, str] | None = None,
    stats_dir:     str               = "./stats",
    generate_plots: bool             = True,  
    target_central_mass: float       = 0.99999,
    detect_artifacts: bool           = True,
    artifact_config: dict            = {},
    montage: Optional[Union[str, List[str]]] = "TCP",
    high_pass: Optional[float]       = 0.5,
    low_pass: Optional[float]        = None,
    clip_threshold: Optional[float]  = None,
    robust_scaling: bool             = True,
    decompose_composite_labels: bool  = False,
):
    edfs = sample_paths(
        glob.glob(os.path.join(str(input_path), "**", "*.edf"), recursive=True),
        percent, seed=seed
    )
    args = [
        (p, input_path, output_path, sample_rates, resample_freq, annotations, stats_dir,
         target_central_mass, detect_artifacts, artifact_config, montage, high_pass, low_pass,
         clip_threshold, robust_scaling, decompose_composite_labels)
        for p in edfs
    ]

    results: List[FileStats] = []
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    if n_jobs and n_jobs > 1:
        with Pool(nodes=n_jobs, initializer=_init_worker) as pool:
            for res in tqdm(pool.uimap(process_one, args), total=len(args), desc="EDFs"):
                if res:
                    results.append(res)
    else:
        for a in tqdm(args, total=len(args), desc="EDFs"):
            res = process_one(a)
            if res:
                results.append(res)

    logger.info(f"Successfully processed: {len(results)} / {len(edfs)} files")

    if not results:
        logger.warning("No files were successfully processed")
        return

    df = pd.DataFrame([r.__dict__ for r in results])
    out_csv = Path(stats_dir) / "preprocessing_stats.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote per-file statistics to {out_csv}")

    total_clipped = int(df["clipped_samples"].sum())
    total_samples = int(df["total_samples"].sum())
    if total_samples:
        logger.info(f"Overall clipping: {(total_clipped * 100 / total_samples):.4f} % of {total_samples} samples")

    subsets_present = sorted(df["subset"].unique().tolist())
    if subsets_present:
        counts = Counter(df["subset"].tolist())
        logger.info("Files per subset: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    label_to_filecount: Dict[str, int] = defaultdict(int)
    for r in results:
        for lab in set(r.annotations_present):
            label_to_filecount[lab] += 1

    if label_to_filecount:
        logger.info("Annotation presence summary (files saved containing label):")
        for lab, cnt in sorted(label_to_filecount.items(), key=lambda x: (-x[1], x[0])):
            logger.info(f"  {lab}: {cnt} file(s)")
    else:
        logger.info("Annotation presence summary: none.")

def main(config: str):
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'high-pass' in cfg and 'high_pass' not in cfg:
        cfg['high_pass'] = cfg.pop('high-pass')
    if 'low-pass' in cfg and 'low_pass' not in cfg:
        cfg['low_pass'] = cfg.pop('low-pass')
    if 'clip-threshold' in cfg and 'clip_threshold' not in cfg:
        cfg['clip_threshold'] = cfg.pop('clip-threshold')

    preprocess(**cfg)

if __name__ == "__main__":
    fire.Fire(main)