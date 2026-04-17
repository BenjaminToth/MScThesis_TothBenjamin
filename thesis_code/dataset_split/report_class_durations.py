#!/usr/bin/env python3
"""Report total annotation durations per class across train/dev/test splits.

This script is meant to be run on the output produced by
`dataset_split/justsplit_patientwise.py` (or any similar split layout).

It reads EDF annotations and aggregates per split:
  - Channel-summed duration: sums every annotation row (will overcount if the
    same interval is duplicated across channels).
  - Interval-union duration (recommended): for each label within a file, merges
    overlapping intervals and sums the union length; duplicates across channels
    (same onset/duration) do not inflate totals.

Annotation description is assumed to be in the form:
  "{channel}:{label}"
The label is the substring after the first colon.

Outputs:
  - class_durations.csv
  - class_durations.json

If split_summary.json is present in the split root (produced by
dataset_split/justsplit_patientwise.py), its split ratios/total seconds
are embedded into class_durations.json under:
    - out["split_time_pct"]: {train/dev/test: percent-of-total-recording-time}
    - out["split_time_ratio"]: same as fractions (pct / 100)

Requires: mne (run inside the EEG-Artifacts conda env if needed).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _require_mne():
    """Verify mne-python is installed, exit if not available.
    
    Raises:
        SystemExit: If mne cannot be imported.
    """
    try:
        import mne  
    except Exception as e:  
        raise SystemExit(
            "mne is required to read EDF annotations. "
            "Run inside the conda env (e.g., `conda run -n EEG-Artifacts python ...`).\n"
            f"Import error: {e}"
        )


def _read_json(path: Path) -> Dict:
    """Load JSON file from path.
    
    Args:
        path: Path to JSON file.
    
    Returns:
        Parsed JSON object as dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Ann:
    """Annotation record with timing and label information.
    
    Attributes:
        onset: Start time of annotation in seconds.
        duration: Duration of annotation in seconds.
        label: Annotation label (e.g., seizure type).
    """
    onset: float
    duration: float
    label: str


def _parse_label(description: str) -> str:
    """Extract label from annotation description string.
    
    Assumes format: '{channel}:{label}'
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


def _summarize_durations(durations: List[float]) -> Dict[str, float]:
    """Compute min, median, and max of duration list.
    
    Args:
        durations: List of duration values in seconds.
    
    Returns:
        Dictionary with 'min_sec', 'median_sec', and 'max_sec' keys.
    """
    if not durations:
        return {
            "min_sec": 0.0,
            "median_sec": 0.0,
            "max_sec": 0.0,
        }

    ordered = sorted(float(duration) for duration in durations)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    else:
        median = ordered[mid]

    return {
        "min_sec": ordered[0],
        "median_sec": median,
        "max_sec": ordered[-1],
    }


def _read_file_annotations(edf_path: Path) -> List[Ann]:
    """Read and parse annotations from an EDF file.
    
    Args:
        edf_path: Path to EDF file.
    
    Returns:
        List of Ann records with valid annotations.
    """
    import mne

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    anns = []
    for a in raw.annotations:
        onset = float(a["onset"])
        duration = float(a["duration"])
        if duration <= 0:
            continue
        label = _parse_label(a["description"])
        if not label:
            continue
        anns.append(Ann(onset=onset, duration=duration, label=label))
    try:
        raw.close()
    except Exception:
        pass
    return anns


def _iter_edfs(split_root: Path, data_subdir: Optional[str]) -> List[Path]:
    """Find all EDF files in split directory.
    
    Args:
        split_root: Root directory for a specific split (train/dev/test).
        data_subdir: Optional subdirectory name (e.g., '01_tcp_ar'). If None, search root directly.
    
    Returns:
        Sorted list of EDF file paths.
    """
    if data_subdir:
        root = split_root / data_subdir
    else:
        root = split_root
    if not root.exists():
        return []
    files = [p for p in root.rglob("*.edf") if p.is_file()]
    files.sort()
    return files


def report(split_root: Path, data_subdir: Optional[str], max_files: Optional[int] = None) -> Dict:
    """Generate annotation duration report across train/dev/test splits.
    
    Computes both channel-sum durations (overcounts overlaps) and interval-union
    durations (recommended; merges overlaps per label per file).
    Optionally enriches output with split time ratios from split_summary.json.
    
    Args:
        split_root: Root directory containing train/dev/test split folders.
        data_subdir: Optional subdirectory within each split (e.g., '01_tcp_ar').
        max_files: Optional limit on EDFs to process per split (for testing).
    
    Returns:
        Dictionary with per-split, per-label annotation statistics.
    """
    _require_mne()

    splits = ["train", "dev", "test"]

    out: Dict[str, Dict] = {"splits": {}}

    split_summary_path = split_root / "split_summary.json"
    split_summary: Optional[Dict] = None
    if split_summary_path.exists():
        try:
            split_summary = _read_json(split_summary_path)
            split_time_pct: Dict[str, float] = {}
            split_time_ratio: Dict[str, float] = {}
            for split in ("train", "dev", "test"):
                srow = (split_summary.get("splits", {}) or {}).get(split, {}) or {}
                pct = srow.get("pct", None)
                if pct is not None:
                    try:
                        split_time_pct[split] = float(pct)
                        split_time_ratio[split] = float(pct) / 100.0
                    except Exception:
                        pass

            if not split_time_pct:
                try:
                    total_all = float(split_summary.get("total_sec", 0.0) or 0.0)
                except Exception:
                    total_all = 0.0
                if total_all > 0:
                    for split in ("train", "dev", "test"):
                        srow = (split_summary.get("splits", {}) or {}).get(split, {}) or {}
                        sec = float(srow.get("total_sec", 0.0) or 0.0)
                        ratio = sec / total_all
                        split_time_ratio[split] = ratio
                        split_time_pct[split] = ratio * 100.0

            out["split_time_pct"] = split_time_pct
            out["split_time_ratio"] = split_time_ratio
        except Exception:
            split_summary = None
    all_labels = set()

    for split in splits:
        split_dir = split_root / split
        files = _iter_edfs(split_dir, data_subdir)
        if max_files is not None:
            files = files[: max_files]

        channel_sum_sec: Dict[str, float] = {}
        union_sec: Dict[str, float] = {}
        n_annotations: Dict[str, int] = {}
        files_with_label: Dict[str, int] = {}
        annotation_durations_sec: Dict[str, List[float]] = {}

        for edf in files:
            anns = _read_file_annotations(edf)
            if not anns:
                continue

            for a in anns:
                channel_sum_sec[a.label] = channel_sum_sec.get(a.label, 0.0) + a.duration
                n_annotations[a.label] = n_annotations.get(a.label, 0) + 1
                annotation_durations_sec.setdefault(a.label, []).append(a.duration)

            label_to_intervals: Dict[str, List[Tuple[float, float]]] = {}
            for a in anns:
                label_to_intervals.setdefault(a.label, []).append((a.onset, a.onset + a.duration))

            for label, intervals in label_to_intervals.items():
                merged = _merge_intervals(intervals)
                u = sum(max(0.0, e - s) for s, e in merged)
                union_sec[label] = union_sec.get(label, 0.0) + u
                files_with_label[label] = files_with_label.get(label, 0) + 1

        labels = sorted(set(channel_sum_sec) | set(union_sec) | set(n_annotations) | set(files_with_label) | set(annotation_durations_sec))
        all_labels.update(labels)

        total_union = sum(union_sec.values())
        total_channel_sum = sum(channel_sum_sec.values())

        out["splits"][split] = {
            "n_files": len(files),
            "total_labeled": {
                "interval_union_sec": float(total_union),
                "channel_sum_sec": float(total_channel_sum),
            },
            "labels": {
                label: {
                    "channel_sum_sec": channel_sum_sec.get(label, 0.0),
                    "interval_union_sec": union_sec.get(label, 0.0),
                    "pct_of_labeled_union": (union_sec.get(label, 0.0) / total_union * 100.0) if total_union > 0 else 0.0,
                    "pct_of_labeled_channel_sum": (channel_sum_sec.get(label, 0.0) / total_channel_sum * 100.0) if total_channel_sum > 0 else 0.0,
                    "n_annotations": n_annotations.get(label, 0),
                    "n_files_with_label": files_with_label.get(label, 0),
                    "annotation_durations_sec": [float(duration) for duration in annotation_durations_sec.get(label, [])],
                    "annotation_duration_summary_sec": _summarize_durations(annotation_durations_sec.get(label, [])),
                }
                for label in labels
            },
        }

    out["all_labels"] = sorted(all_labels)
    return out


def write_outputs(summary: Dict, out_dir: Path) -> None:
    """Write annotation report to CSV and JSON files.
    
    Args:
        summary: Report dictionary from report().
        out_dir: Output directory for CSV and JSON files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "class_durations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = out_dir / "class_durations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "label",
                "channel_sum_sec",
                "interval_union_sec",
                "pct_of_labeled_union",
                "pct_of_labeled_channel_sum",
                "n_annotations",
                "n_files_with_label",
            ]
        )
        for split, payload in summary.get("splits", {}).items():
            labels = payload.get("labels", {})
            for label in sorted(labels.keys()):
                row = labels[label]
                w.writerow(
                    [
                        split,
                        label,
                        f"{row.get('channel_sum_sec', 0.0):.6f}",
                        f"{row.get('interval_union_sec', 0.0):.6f}",
                        f"{row.get('pct_of_labeled_union', 0.0):.6f}",
                        f"{row.get('pct_of_labeled_channel_sum', 0.0):.6f}",
                        int(row.get("n_annotations", 0)),
                        int(row.get("n_files_with_label", 0)),
                    ]
                )


def main() -> None:
    """Parse arguments and generate annotation duration report."""
    p = argparse.ArgumentParser(description="Report total annotated durations per class across splits")
    p.add_argument(
        "--split-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "01_tcp_ar_split",
        help="Root directory containing train/dev/test subfolders",
    )
    p.add_argument(
        "--data-subdir",
        type=str,
        default="01_tcp_ar",
        help="Optional data folder under each split (set '' to scan split folder directly)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write class_durations.csv/json (default: split-root)",
    )
    p.add_argument("--max-files", type=int, default=None, help="Limit EDFs per split for quick runs")

    args = p.parse_args()
    split_root: Path = args.split_root
    data_subdir = args.data_subdir.strip() or None
    out_dir = args.out_dir or split_root

    summary = report(split_root=split_root, data_subdir=data_subdir, max_files=args.max_files)
    write_outputs(summary, out_dir=out_dir)

    print(f"Wrote: {out_dir / 'class_durations.csv'}")
    print(f"Wrote: {out_dir / 'class_durations.json'}")
    for split, payload in summary.get("splits", {}).items():
        labels = payload.get("labels", {})
        top = sorted(labels.items(), key=lambda kv: kv[1].get("interval_union_sec", 0.0), reverse=True)[:10]
        print(f"{split}: {len(labels)} labels")
        for label, row in top:
            print(f"  {label}: union_sec={row['interval_union_sec']:.1f}, channel_sum_sec={row['channel_sum_sec']:.1f}")


if __name__ == "__main__":
    main()
