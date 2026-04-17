#!/usr/bin/env python3
"""Patient-wise train/dev/test split for TUH-style EDFs (01_tcp_ar).

Creates an 80/10/10 split while ensuring each patient ID appears in exactly
one split. The split is *weighted by recording length*: each patient is treated
as an indivisible unit with weight equal to the sum of durations (seconds) of
all EDFs belonging to that patient.

By default it materializes the split as folders of symlinks under:
  dataset_split/01_tcp_ar_patientwise_lenweighted_seed<seed>/{train,dev,test}/01_tcp_ar/

Patient ID extraction (default): filename prefix before "_s".
Example: aaaaabbn_s010_t002.edf -> patient_id=aaaaabbn

Duration is computed from EDF headers using pyEDFlib when possible, with an
MNE fallback.

Optional label balancing:
    With --balance-labels (default), the split tries to avoid concentrating
    most of a label's annotated time into a single split by using EDF annotations
    (MNE) to compute per-patient label union durations and assigning patients to
    splits to match the target 80/10/10 ratios for both total recording time and
    per-label union time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional


_PATIENT_RE = re.compile(r"^(?P<pid>.+?)_s\d+_t\d+\.(?i:edf)$")


@dataclass(frozen=True)
class FileInfo:
    path: Path
    filename: str
    patient_id: str
    duration_sec: float


def _mne_available() -> Tuple[bool, Optional[str]]:
    """Check if MNE-Python is available.
    
    Returns:
        Tuple (is_available, error_message). If available, error_message is None.
    """
    try:
        import mne  

        return True, None
    except Exception as e:
        return False, str(e)


def _parse_label(description: str) -> str:
    """Parse annotation label from EDF description string.
    
    Extracts text after ':' if present, otherwise uses full description.
    Returns lowercase normalized label.
    """
    s = str(description)
    if ":" in s:
        return s.split(":", 1)[1].strip().lower()
    return s.strip().lower()


def _merge_intervals(intervals: List[Tuple[float, float]], tol: float = 1e-9) -> List[Tuple[float, float]]:
    """Merge overlapping time intervals.
    
    Args:
        intervals: List of (start, end) tuples in seconds.
        tol: Tolerance for overlap detection (default 1e-9).
    
    Returns:
        List of merged non-overlapping intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + tol:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _file_label_union_sec(edf_path: Path) -> Dict[str, float]:
    """Return union duration (sec) per label within a single EDF."""
    import mne  # type: ignore

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    try:
        label_to_intervals: Dict[str, List[Tuple[float, float]]] = {}
        for a in raw.annotations:
            onset = float(a["onset"])
            duration = float(a["duration"])
            if duration <= 0:
                continue
            label = _parse_label(a["description"])
            if not label:
                continue
            label_to_intervals.setdefault(label, []).append((onset, onset + duration))

        out: Dict[str, float] = {}
        for label, intervals in label_to_intervals.items():
            merged = _merge_intervals(intervals)
            out[label] = float(sum(max(0.0, e - s) for s, e in merged))
        return out
    finally:
        try:
            raw.close()
        except Exception:
            pass


def _iter_edf_files(root: Path) -> List[Path]:
    """Recursively find all EDF files in a directory.
    
    Args:
        root: Root directory to search.
    
    Returns:
        Sorted list of EDF file paths.
    
    Raises:
        FileNotFoundError: If root directory does not exist or is not a directory.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Source dir not found or not a directory: {root}")

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".edf"]
    files.sort()
    return files


def _extract_patient_id(filename: str) -> str:
    """Extract patient ID from EDF filename.
    
    Default pattern matches: <patient_id>_s<seq>_t<trial>.edf
    Falls back to splitting on '_s', then '_', then using filename stem.
    
    Args:
        filename: The EDF filename.
    
    Returns:
        Patient ID string.
    """
    m = _PATIENT_RE.match(filename)
    if m:
        return m.group("pid")

    if "_s" in filename:
        return filename.split("_s", 1)[0]
    if "_" in filename:
        return filename.split("_", 1)[0]
    return Path(filename).stem


def _duration_pyedflib(path: Path) -> float:
    """Extract EDF duration using pyEDFlib (optional dependency).
    
    Raises:
        ImportError: If pyEDFlib is not installed.
        ValueError: If duration cannot be determined from the file.
    """
    import pyedflib  # type: ignore

    reader = pyedflib.EdfReader(str(path))
    try:
        dur = float(reader.getFileDuration())
    finally:
        try:
            reader.close()
        except Exception:
            pass

    if not (dur > 0 and dur != float("inf")):
        raise ValueError(f"Non-positive or invalid duration from pyEDFlib: {dur}")
    return dur


def _duration_edf_header(path: Path) -> float:
    """Compute EDF duration from the file header only (no external deps).

    EDF fixed header fields (bytes):
      - number of data records: 236:244
      - duration of a data record (sec): 244:252
      - number of signals: 252:256

    If number of data records is -1/unknown, estimate it from file size using
    samples-per-record for each signal (EDF uses int16 => 2 bytes/sample).
    """

    with open(path, "rb") as f:
        header = f.read(256)
        if len(header) < 256:
            raise ValueError("File too small to be valid EDF")

        n_records_str = header[236:244].decode("ascii", errors="ignore").strip()
        record_dur_str = header[244:252].decode("ascii", errors="ignore").strip()
        n_signals_str = header[252:256].decode("ascii", errors="ignore").strip()

        try:
            n_records = int(n_records_str)
        except Exception as e:
            raise ValueError(f"Invalid number-of-records field: {n_records_str!r}") from e

        try:
            record_dur = float(record_dur_str)
        except Exception as e:
            raise ValueError(f"Invalid record-duration field: {record_dur_str!r}") from e

        if not (record_dur > 0 and math.isfinite(record_dur)):
            raise ValueError(f"Non-positive/invalid record duration: {record_dur}")

        if n_records > 0:
            return float(n_records) * record_dur

        try:
            n_signals = int(n_signals_str)
        except Exception as e:
            raise ValueError(f"Invalid number-of-signals field: {n_signals_str!r}") from e

        if n_signals <= 0:
            raise ValueError(f"Invalid number of signals: {n_signals}")

        header_bytes = 256 + n_signals * 256

        samples_block_offset = 256 + (n_signals * 216)
        f.seek(samples_block_offset)
        samples_block = f.read(n_signals * 8)
        if len(samples_block) < n_signals * 8:
            raise ValueError("Truncated EDF header while reading samples-per-record")

        samples_per_record: List[int] = []
        for i in range(n_signals):
            s = samples_block[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip()
            try:
                samples_per_record.append(int(s))
            except Exception as e:
                raise ValueError(f"Invalid samples-per-record for signal {i}: {s!r}") from e

        total_samples_per_record = sum(samples_per_record)
        if total_samples_per_record <= 0:
            raise ValueError("Invalid total samples per record")

    file_size = path.stat().st_size
    if file_size <= header_bytes:
        raise ValueError("EDF file smaller than header")

    bytes_per_record = total_samples_per_record * 2  
    est_records = int((file_size - header_bytes) // bytes_per_record)
    if est_records <= 0:
        raise ValueError("Could not estimate number of records from file size")

    return float(est_records) * record_dur


def _duration_mne(path: Path) -> float:
    """Extract EDF duration using MNE-Python.
    
    Args:
        path: Path to EDF file.
    
    Returns:
        Duration in seconds.
    
    Raises:
        ValueError: If duration cannot be determined from the file.
    """
    import mne  # type: ignore

    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        if sfreq <= 0 or n_times <= 0:
            raise ValueError(f"Invalid sfreq/n_times: sfreq={sfreq}, n_times={n_times}")
        return n_times / sfreq
    finally:
        try:
            raw.close()
        except Exception:
            pass


def get_duration_sec(path: Path) -> float:
    """Get EDF file duration in seconds.
    
    Attempts extraction in order of preference:
    1. Dependency-free header parsing (fastest, always available)
    2. pyEDFlib (if installed)
    3. MNE-Python (requires mne; used as fallback)
    
    Returns:
        Duration in seconds.
    """
    try:
        return _duration_edf_header(path)
    except Exception:
        pass

    try:
        return _duration_pyedflib(path)
    except Exception:
        pass

    return _duration_mne(path)


def _format_hms(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS string.
    
    Args:
        seconds: Duration in seconds.
    
    Returns:
        Formatted time string.
    """
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _choose_split(
    remaining: Dict[str, float],
    rng: random.Random,
) -> str:
    """Select split with largest remaining target duration.
    
    Breaks ties deterministically, then uses RNG for final selection.
    
    Args:
        remaining: Current remaining target duration per split.
        rng: Random number generator instance (seeded for reproducibility).
    
    Returns:
        Name of selected split ('train', 'dev', or 'test').
    """
    max_rem = max(remaining.values())
    ties = [k for k, v in remaining.items() if abs(v - max_rem) < 1e-9]
    if len(ties) == 1:
        return ties[0]
    ties.sort()
    return rng.choice(ties)


def split_patientwise_lenweighted(
    patient_to_files: Dict[str, List[FileInfo]],
    seed: int,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, List[str]]:
    """Create train/dev/test split weighted by total recording duration per patient.
    
    Ensures each patient appears in exactly one split. Patients are sorted by
    total duration (descending) and greedily assigned to splits to match target
    duration ratios.
    
    Args:
        patient_to_files: Mapping from patient ID to list of FileInfo objects.
        seed: Random seed for deterministic tie-breaking.
        ratios: Target (train, dev, test) ratios (default 80/10/10).
    
    Returns:
        Dictionary mapping split name to list of assigned patient IDs.
    
    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    train_r, dev_r, test_r = ratios
    if abs((train_r + dev_r + test_r) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    patient_seconds = {pid: sum(f.duration_sec for f in files) for pid, files in patient_to_files.items()}
    total_sec = sum(patient_seconds.values())
    targets = {
        "train": total_sec * train_r,
        "dev": total_sec * dev_r,
        "test": total_sec * test_r,
    }

    rng = random.Random(seed)

    pids = list(patient_seconds.keys())
    rng.shuffle(pids) 
    pids.sort(key=lambda pid: patient_seconds[pid], reverse=True)  

    assigned: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    used_sec = {k: 0.0 for k in assigned}

    for pid in pids:
        remaining = {k: targets[k] - used_sec[k] for k in assigned}
        split = _choose_split(remaining, rng)
        assigned[split].append(pid)
        used_sec[split] += patient_seconds[pid]

    return assigned


def split_patientwise_lenweighted_labelbalanced(
    patient_to_files: Dict[str, List[FileInfo]],
    patient_label_union_sec: Dict[str, Dict[str, float]],
    seed: int,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    *,
    label_weight: float = 1.0,
    time_weight: float = 1.0,
    drop_labels: Optional[Iterable[str]] = None,
    opt_swap_iters: int = 2000,
) -> Dict[str, List[str]]:
    """Length-weighted split with additional label-balance objective.

    Balances annotated union time per label across splits while keeping total
    recording time close to the desired ratios.
    """

    train_r, dev_r, test_r = ratios
    if abs((train_r + dev_r + test_r) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios}")

    drop = {str(x).strip().lower() for x in (drop_labels or []) if str(x).strip()}

    patient_seconds = {pid: sum(f.duration_sec for f in files) for pid, files in patient_to_files.items()}
    total_sec = sum(patient_seconds.values())
    targets_time = {
        "train": total_sec * train_r,
        "dev": total_sec * dev_r,
        "test": total_sec * test_r,
    }

    label_set = set()
    for pid, lbls in patient_label_union_sec.items():
        for lab in (lbls or {}).keys():
            lab_l = str(lab).strip().lower()
            if lab_l and lab_l not in drop:
                label_set.add(lab_l)
    labels = sorted(label_set)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    patient_label_vec: Dict[str, List[float]] = {}
    label_totals = [0.0 for _ in labels]
    for pid in patient_to_files.keys():
        vec = [0.0 for _ in labels]
        for lab, sec in (patient_label_union_sec.get(pid, {}) or {}).items():
            lab_l = str(lab).strip().lower()
            if lab_l in drop:
                continue
            idx = label_to_idx.get(lab_l)
            if idx is None:
                continue
            v = float(sec)
            if v <= 0:
                continue
            vec[idx] += v
        patient_label_vec[pid] = vec
        for i, v in enumerate(vec):
            label_totals[i] += float(v)

    ratios_by_split = {"train": train_r, "dev": dev_r, "test": test_r}

    target_label = {
        split: [label_totals[i] * ratios_by_split[split] for i in range(len(labels))]
        for split in ("train", "dev", "test")
    }

    eps = 1e-9
    denom_label = {
        split: [max(target_label[split][i], eps) for i in range(len(labels))]
        for split in ("train", "dev", "test")
    }

    rng = random.Random(seed)

    pids = list(patient_seconds.keys())
    rng.shuffle(pids) 
    pids.sort(key=lambda pid: patient_seconds[pid], reverse=True)  

    assigned: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    used_time = {"train": 0.0, "dev": 0.0, "test": 0.0}
    used_label = {"train": [0.0 for _ in labels], "dev": [0.0 for _ in labels], "test": [0.0 for _ in labels]}

    def split_cost(split: str) -> float:
        t_target = max(targets_time[split], eps)
        t_err = (used_time[split] - targets_time[split]) / t_target
        cost = time_weight * (t_err * t_err)
        if labels and label_weight != 0.0:
            l_cost = 0.0
            ul = used_label[split]
            tl = target_label[split]
            dl = denom_label[split]
            for i in range(len(labels)):
                e = (ul[i] - tl[i]) / dl[i]
                l_cost += e * e
            cost += label_weight * l_cost
        return float(cost)

    def split_cost_with_add(split: str, pid: str) -> float:
        t_target = max(targets_time[split], eps)
        new_time = used_time[split] + patient_seconds[pid]
        t_err = (new_time - targets_time[split]) / t_target
        cost = time_weight * (t_err * t_err)

        if labels and label_weight != 0.0:
            vec = patient_label_vec[pid]
            ul = used_label[split]
            tl = target_label[split]
            dl = denom_label[split]
            l_cost = 0.0
            for i in range(len(labels)):
                e = (ul[i] + vec[i] - tl[i]) / dl[i]
                l_cost += e * e
            cost += label_weight * l_cost
        return float(cost)

    for pid in pids:
        best_splits: List[str] = []
        best_score = None
        for split in ("train", "dev", "test"):
            old = split_cost(split)
            new = split_cost_with_add(split, pid)
            score = new - old
            if best_score is None or score < best_score - 1e-12:
                best_score = score
                best_splits = [split]
            elif abs(score - best_score) <= 1e-12:
                best_splits.append(split)

        pick = rng.choice(sorted(best_splits))
        assigned[pick].append(pid)
        used_time[pick] += patient_seconds[pid]
        if labels:
            vec = patient_label_vec[pid]
            ul = used_label[pick]
            for i in range(len(labels)):
                ul[i] += vec[i]

    if opt_swap_iters and opt_swap_iters > 0 and (len(pids) >= 2):
        split_of = {pid: split for split, xs in assigned.items() for pid in xs}

        def total_objective() -> float:
            return split_cost("train") + split_cost("dev") + split_cost("test")

        cur_obj = total_objective()
        for _ in range(int(opt_swap_iters)):
            a, b = rng.sample(pids, 2)
            sa = split_of[a]
            sb = split_of[b]
            if sa == sb:
                continue

            old_sa = split_cost(sa)
            old_sb = split_cost(sb)

            used_time[sa] -= patient_seconds[a]
            used_time[sb] -= patient_seconds[b]
            used_time[sa] += patient_seconds[b]
            used_time[sb] += patient_seconds[a]

            if labels:
                va = patient_label_vec[a]
                vb = patient_label_vec[b]
                ula = used_label[sa]
                ulb = used_label[sb]
                for i in range(len(labels)):
                    ula[i] = ula[i] - va[i] + vb[i]
                    ulb[i] = ulb[i] - vb[i] + va[i]

            new_sa = split_cost(sa)
            new_sb = split_cost(sb)

            new_obj = cur_obj + (new_sa + new_sb - old_sa - old_sb)
            if new_obj <= cur_obj - 1e-12:
                split_of[a], split_of[b] = sb, sa
                assigned[sa].remove(a)
                assigned[sb].remove(b)
                assigned[sa].append(b)
                assigned[sb].append(a)
                cur_obj = new_obj
            else:
                used_time[sa] -= patient_seconds[b]
                used_time[sb] -= patient_seconds[a]
                used_time[sa] += patient_seconds[a]
                used_time[sb] += patient_seconds[b]

                if labels:
                    va = patient_label_vec[a]
                    vb = patient_label_vec[b]
                    ula = used_label[sa]
                    ulb = used_label[sb]
                    for i in range(len(labels)):
                        ula[i] = ula[i] + va[i] - vb[i]
                        ulb[i] = ulb[i] + vb[i] - va[i]

    return assigned


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    """Create an empty directory, optionally removing existing content.
    
    Args:
        path: Directory path to create.
        overwrite: If True, remove existing directory. If False, raise error if exists.
    
    Raises:
        FileExistsError: If path exists and overwrite is False.
    """
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {path} (use --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _make_symlink(src: Path, dst: Path, relative: bool) -> None:
    """Create a symlink from destination to source.
    
    Args:
        src: Source file path.
        dst: Destination symlink path.
        relative: If True, use relative path; if False, use absolute path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    target = os.path.relpath(src, start=dst.parent) if relative else str(src)
    os.symlink(target, str(dst))


def materialize_split(
    src_dir: Path,
    out_dir: Path,
    patient_to_files: Dict[str, List[FileInfo]],
    assigned: Dict[str, List[str]],
    relative_symlinks: bool,
    dry_run: bool,
) -> Dict[str, Dict[str, float]]:
    """Create split directory structure with symlinks to source EDFs.
    
    Args:
        src_dir: Source directory name (typically '01_tcp_ar').
        out_dir: Output root directory.
        patient_to_files: Mapping from patient ID to file list.
        assigned: Split assignment from split function.
        relative_symlinks: If True, create relative symlinks (more portable).
        dry_run: If True, compute stats without writing files.
    
    Returns:
        Dictionary with per-split statistics (patient count, file count, duration).
    """
    split_stats: Dict[str, Dict[str, float]] = {}

    for split_name, pids in assigned.items():
        files: List[FileInfo] = []
        for pid in pids:
            files.extend(patient_to_files[pid])

        split_root = out_dir / split_name / src_dir.name

        total_sec = sum(f.duration_sec for f in files)
        split_stats[split_name] = {
            "patients": float(len(pids)),
            "files": float(len(files)),
            "total_sec": float(total_sec),
        }

        if dry_run:
            continue

        split_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = split_root / f.filename
            _make_symlink(f.path, dst, relative=relative_symlinks)

    return split_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-wise length-weighted split for 01_tcp_ar")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/nas/EEG-Seizures/data/TUH/preprocessed/TUAR_v3.0.1/test/01_tcp_ar"),
        help="Path to the processed 01_tcp_ar directory containing EDFs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root folder (will create train/dev/test under this)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic tie-breaking")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Compute split and print stats without writing")
    parser.add_argument(
        "--relative-symlinks",
        action="store_true",
        help="Create relative symlinks (more portable if moving the split folder)",
    )

    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument(
        "--balance-labels",
        dest="balance_labels",
        action="store_true",
        default=True,
        help="Balance per-label annotated union time across splits (default: on)",
    )
    balance_group.add_argument(
        "--no-balance-labels",
        dest="balance_labels",
        action="store_false",
        help="Disable label balancing and split by recording length only",
    )

    parser.add_argument(
        "--balance-label-weight",
        type=float,
        default=1.0,
        help="Weight of label-balance objective relative to time-balance objective",
    )
    parser.add_argument(
        "--balance-opt-swap-iters",
        type=int,
        default=2000,
        help="Number of random patient-swap optimization iterations (0 to disable)",
    )
    parser.add_argument(
        "--drop-label",
        action="append",
        default=None,
        help="Label to exclude from balancing (can be repeated; default: bad_acq_skip, artifact)",
    )

    args = parser.parse_args()

    src_dir: Path = args.source_dir
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "data" / "01_tcp_ar_split"
    out_dir: Path = args.out_dir or default_out

    edf_files = _iter_edf_files(src_dir)
    if not edf_files:
        raise RuntimeError(f"No EDF files found under: {src_dir}")

    file_infos: List[FileInfo] = []
    failures: List[str] = []

    for p in edf_files:
        fn = p.name
        pid = _extract_patient_id(fn)
        try:
            dur = get_duration_sec(p)
        except Exception as e:
            failures.append(f"{p}: {e}")
            continue
        file_infos.append(FileInfo(path=p, filename=fn, patient_id=pid, duration_sec=dur))

    if failures:
        print(f"Warning: failed to read duration for {len(failures)} files; they were skipped.")
        for line in failures[:5]:
            print(f"  - {line}")

    if not file_infos:
        raise RuntimeError("No valid EDFs with readable durations were found.")

    patient_to_files: Dict[str, List[FileInfo]] = {}
    for fi in file_infos:
        patient_to_files.setdefault(fi.patient_id, []).append(fi)

    drop_labels = args.drop_label if args.drop_label is not None else ["bad_acq_skip", "artifact"]

    patient_label_union_sec: Optional[Dict[str, Dict[str, float]]] = None
    if args.balance_labels:
        ok, err = _mne_available()
        if not ok:
            print(
                "Warning: --balance-labels requested but mne is not available; falling back to length-only split. "
                "Run inside the conda env or install mne to enable label balancing.\n"
                f"Import error: {err}"
            )
            args.balance_labels = False

    if args.balance_labels:
        patient_label_union_sec = {}

        for fi in file_infos:
            per_file = _file_label_union_sec(fi.path)
            pid_map = patient_label_union_sec.setdefault(fi.patient_id, {})
            for lab, sec in (per_file or {}).items():
                lab_l = str(lab).strip().lower()
                if not lab_l:
                    continue
                pid_map[lab_l] = float(pid_map.get(lab_l, 0.0) + float(sec))

        assigned = split_patientwise_lenweighted_labelbalanced(
            patient_to_files,
            patient_label_union_sec,
            seed=args.seed,
            label_weight=float(args.balance_label_weight),
            drop_labels=drop_labels,
            opt_swap_iters=int(args.balance_opt_swap_iters),
        )
    else:
        assigned = split_patientwise_lenweighted(patient_to_files, seed=args.seed)

    all_pids = sum((assigned[k] for k in ("train", "dev", "test")), [])
    if len(set(all_pids)) != len(all_pids):
        raise AssertionError("Patient overlap detected across splits")

    total_sec = sum(fi.duration_sec for fi in file_infos)

    if not args.dry_run:
        _ensure_empty_dir(out_dir, overwrite=args.overwrite)

    split_stats = materialize_split(
        src_dir=src_dir,
        out_dir=out_dir,
        patient_to_files=patient_to_files,
        assigned=assigned,
        relative_symlinks=args.relative_symlinks,
        dry_run=args.dry_run,
    )

    summary = {
        "source_dir": str(src_dir),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "total_files": int(len(file_infos)),
        "total_patients": int(len(patient_to_files)),
        "total_sec": float(total_sec),
        "balance_labels": bool(args.balance_labels),
        "balance_label_weight": float(args.balance_label_weight),
        "balance_opt_swap_iters": int(args.balance_opt_swap_iters),
        "drop_labels": list(drop_labels),
        "splits": {},
    }

    for split_name in ("train", "dev", "test"):
        s = split_stats.get(split_name, {"patients": 0.0, "files": 0.0, "total_sec": 0.0})
        pct = (s["total_sec"] / total_sec * 100.0) if total_sec > 0 else 0.0
        summary["splits"][split_name] = {
            "patients": int(s["patients"]),
            "files": int(s["files"]),
            "total_sec": float(s["total_sec"]),
            "pct": float(pct),
            "hms": _format_hms(s["total_sec"]),
        }

    if args.balance_labels and patient_label_union_sec is not None:
        try:
            label_totals: Dict[str, float] = {}
            per_split: Dict[str, Dict[str, float]] = {"train": {}, "dev": {}, "test": {}}

            drop_set = {str(x).strip().lower() for x in drop_labels}

            for split_name in ("train", "dev", "test"):
                for pid in assigned.get(split_name, []):
                    for lab, sec in (patient_label_union_sec.get(pid, {}) or {}).items():
                        lab_l = str(lab).strip().lower()
                        if lab_l in drop_set:
                            continue
                        per_split[split_name][lab_l] = float(per_split[split_name].get(lab_l, 0.0) + float(sec))
                        label_totals[lab_l] = float(label_totals.get(lab_l, 0.0) + float(sec))

            label_balance: Dict[str, Dict] = {}
            for lab in sorted(label_totals.keys()):
                tot = float(label_totals.get(lab, 0.0))
                row = {"total_union_sec": tot, "splits": {}}
                for split_name in ("train", "dev", "test"):
                    sec = float(per_split[split_name].get(lab, 0.0))
                    row["splits"][split_name] = {
                        "union_sec": sec,
                        "pct_of_label_total": (sec / tot * 100.0) if tot > 0 else 0.0,
                    }
                label_balance[lab] = row
            summary["label_balance"] = label_balance
        except Exception:
            pass

    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.dry_run:
        with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
