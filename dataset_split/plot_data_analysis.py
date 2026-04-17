#!/usr/bin/env python3
"""Plot class distribution across train/dev/test.

Reads:
    - class_durations.json (from dataset_split/report_class_durations.py)
    - split_summary.json (optional; from dataset_split/justsplit_patientwise.py)

Writes:
    - class_distribution_pct_total_time.png
    - class_distribution_pct_total_time_whole_dataset.png
    - class_distribution_pct_total_time_combined.png
    - class_annotation_counts_total.png
    - class_annotation_length_distribution.png
    - class_annotation_length_distribution_2x3_middle95.png
    - class_annotation_length_distribution_3x2_middle95.png
    - class_annotation_length_distribution_6x1_middle95.png
    - artifact_eeg_window_4s_musc.png

Definition of percentage:
    - Preferred: pct = interval_union_sec(label, split) / total_sec(split) * 100
        where total_sec(split) comes from split_summary.json.
    - Fallback (if split_summary.json missing): pct_of_labeled_union from class_durations.json.

Usage:
    python dataset_split/plot_data_analysis.py \
        --split-root data/01_tcp_ar_split \
    --out-dir results/data_plots
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WHOLE_DATASET_BAR_COLOR = "#264653"
SPLIT_BAR_COLORS = {
    "train": "#264653",
    "dev": "#2A9D8F",
    "test": "#E76F51",
}
COMBINED_TOTAL_BAR_COLOR = "#E9C46A"
COMBINED_TOTAL_EDGE_COLOR = "#6B4F1D"
SPLIT_BAR_FALLBACK = ["#264653", "#2A9D8F", "#E76F51", "#E9C46A", "#577590"]
TEXT_COLOR = "#1F2933"
ANNOTATION_LENGTH_BOX_COLORS = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#E76F51",
    "#577590",
    "#8AB17D",
    "#BC6C25",
    "#6D597A",
]


def _require_matplotlib() -> None:
    """Verify matplotlib is installed, exit if not available.
    
    Raises:
        SystemExit: If matplotlib cannot be imported.
    """
    try:
        import matplotlib 
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Install it (e.g., `conda install matplotlib` or `pip install matplotlib`).\n"
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

def _get_splits_present(*payloads: Dict) -> List[str]:
    """Extract split names from payloads, ordered by preference.
    
    Args:
        *payloads: Dictionary payloads to search for 'splits' key.
    
    Returns:
        List of split names in preferred order (train, dev, test, then others).
    """
    preferred = ["train", "dev", "test"]
    present = set()
    for p in payloads:
        splits = p.get("splits", {}) if isinstance(p, dict) else {}
        present.update(splits.keys())
    ordered = [s for s in preferred if s in present]
    ordered += sorted([s for s in present if s not in preferred])
    return ordered

def _savefig(fig, out_path: Path) -> None:
    """Save matplotlib figure to file with standard settings.
    
    Args:
        fig: Matplotlib figure object.
        out_path: Output file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def _notitle_path(out_path: Path) -> Path:
    """Generate notitle variant of output path.
    
    Args:
        out_path: Original output path.
    
    Returns:
        Modified path with '_notitle' suffix before extension.
    """
    return out_path.with_name(f"{out_path.stem}_notitle{out_path.suffix}")


def _strip_titles(fig) -> None:
    """Remove figure and axis titles, preserving major/minor panel headings.
    
    Prepares figure for 'no-title' export while keeping organizational labels.
    """
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")
    keep_tokens = ("major classes", "minor classes")
    for ax in fig.get_axes():
        title = (ax.get_title() or "").strip()
        if title and any(token in title.lower() for token in keep_tokens):
            continue
        ax.set_title("")
    try:
        for legend in getattr(fig, "legends", []) or []:
            legend.set_bbox_to_anchor((0.965, 0.995))
            legend.set_in_layout(True)
    except Exception:
        pass


def _savefig_notitle(fig, out_path: Path) -> None:
    """Save figure without titles to notitle variant path.
    
    Args:
        fig: Matplotlib figure object.
        out_path: Base output file path (will add '_notitle' to filename).
    """
    _strip_titles(fig)
    try:
        if not bool(getattr(fig, "get_constrained_layout", lambda: False)()):
            fig.tight_layout()
    except Exception:
        pass
    _savefig(fig, _notitle_path(out_path))

def _split_major_minor(labels: List[str], pct: "np.ndarray", *, major_threshold: float) -> Tuple[List[str], "np.ndarray", List[str], "np.ndarray"]:
    """Split labels into major/minor based on max % across splits."""
    import numpy as np

    if len(labels) == 0:
        return [], np.zeros((0, 0)), [], np.zeros((0, 0))

    max_per_label = pct.max(axis=1) if pct.size else np.zeros(len(labels))
    major_idx = [i for i, m in enumerate(max_per_label) if float(m) >= major_threshold]
    minor_idx = [i for i, m in enumerate(max_per_label) if float(m) < major_threshold]

    major_labels = [labels[i] for i in major_idx]
    minor_labels = [labels[i] for i in minor_idx]
    major_pct = pct[major_idx] if major_idx else np.zeros((0, pct.shape[1]))
    minor_pct = pct[minor_idx] if minor_idx else np.zeros((0, pct.shape[1]))
    return major_labels, major_pct, minor_labels, minor_pct


def _get_plot_labels(class_durations: Dict, splits: List[str]) -> List[str]:
    """Get plottable labels, excluding non-target/meta classes.
    
    Args:
        class_durations: Class duration statistics dictionary.
        splits: List of split names to consider.
    
    Returns:
        Sorted list of label names (excluding bad_acq_skip, artifact, artf).
    """
    labels = class_durations.get("all_labels") or []
    if not labels:
        label_set = set()
        for split in splits:
            label_set.update((class_durations.get("splits", {}).get(split, {}).get("labels", {}) or {}).keys())
        labels = sorted(label_set)

    drop_labels = {"bad_acq_skip", "artifact", "artf"}
    return [label for label in labels if label.strip().lower() not in drop_labels]


def _get_union_sec_matrix(class_durations: Dict, labels: List[str], splits: List[str], *, np) -> "np.ndarray":
    """Extract union duration matrix from class_durations.
    
    Args:
        class_durations: Class duration statistics dictionary.
        labels: List of class labels.
        splits: List of split names.
        np: Numpy module reference.
    
    Returns:
        Numpy array of shape (labels, splits) with union durations in seconds.
    """
    union_sec = np.zeros((len(labels), len(splits)), dtype=float)
    for j, split in enumerate(splits):
        for i, label in enumerate(labels):
            union_sec[i, j] = float(
                class_durations.get("splits", {})
                .get(split, {})
                .get("labels", {})
                .get(label, {})
                .get("interval_union_sec", 0.0)
            )
    return union_sec


def _get_annotation_count_matrix(class_durations: Dict, labels: List[str], splits: List[str], *, np) -> "np.ndarray":
    """Extract annotation count matrix from class_durations.
    
    Args:
        class_durations: Class duration statistics dictionary.
        labels: List of class labels.
        splits: List of split names.
        np: Numpy module reference.
    
    Returns:
        Numpy array of shape (labels, splits) with annotation counts.
    """
    annotation_counts = np.zeros((len(labels), len(splits)), dtype=float)
    for j, split in enumerate(splits):
        for i, label in enumerate(labels):
            annotation_counts[i, j] = float(
                class_durations.get("splits", {})
                .get(split, {})
                .get("labels", {})
                .get(label, {})
                .get("n_annotations", 0)
            )
    return annotation_counts


def _get_annotation_durations_by_label(class_durations: Dict, labels: List[str], splits: List[str]) -> Dict[str, List[float]]:
    """Extract per-annotation durations grouped by label.
    
    Args:
        class_durations: Class duration statistics dictionary.
        labels: List of class labels.
        splits: List of split names.
    
    Returns:
        Dictionary mapping label to list of individual annotation durations.
    """
    durations_by_label: Dict[str, List[float]] = {label: [] for label in labels}
    for split in splits:
        split_labels = class_durations.get("splits", {}).get(split, {}).get("labels", {})
        for label in labels:
            durations = split_labels.get(label, {}).get("annotation_durations_sec", []) or []
            durations_by_label[label].extend(float(duration) for duration in durations)
    return durations_by_label


def _get_total_labeled_union_sec(class_durations: Dict, splits: List[str]) -> float:
    """Sum labeled union duration across all splits.
    
    Args:
        class_durations: Class duration statistics dictionary.
        splits: List of split names.
    
    Returns:
        Total labeled union duration in seconds across all splits.
    """
    return sum(
        float(
            class_durations.get("splits", {})
            .get(split, {})
            .get("total_labeled", {})
            .get("interval_union_sec", 0.0)
        )
        for split in splits
    )


def _style_barh_axis(ax, xlabel: str = "Percent (%)") -> None:
    """Apply consistent styling to horizontal bar chart axes.
    
    Args:
        ax: Matplotlib axis object.
        xlabel: Label for x-axis (default: "Percent (%)").
    """
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="semibold", color=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.5, zorder=0)
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=11)
    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("semibold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("semibold")


def _plot_grouped_barh_panel(
    ax,
    labs: List[str],
    vals: "np.ndarray",
    series_names: List[str],
    series_colors: List[str],
    title: str,
    *,
    highlight_series: Optional[List[str]] = None,
) -> None:
    """Plot grouped horizontal bar chart on given axis.
    
    Args:
        ax: Matplotlib axis object.
        labs: List of label names.
        vals: Numpy array of values (labels x series).
        series_names: Names of data series (columns).
        series_colors: Colors for each series.
        title: Axis title.
        highlight_series: Optional series names to highlight with hatching.
    """
    if not labs:
        ax.axis("off")
        return

    import numpy as np

    highlighted = set(highlight_series or [])

    y = np.arange(len(labs), dtype=float)
    group_h = 0.75
    bar_h = group_h / max(len(series_names), 1)

    for j, series_name in enumerate(series_names):
        y_off = y + (j - (len(series_names) - 1) / 2) * bar_h
        color = series_colors[j % len(series_colors)]
        bar_kwargs = {
            "height": bar_h,
            "label": series_name,
            "color": color,
            "zorder": 3,
        }
        if series_name in highlighted:
            bar_kwargs.update(
                {
                    "edgecolor": COMBINED_TOTAL_EDGE_COLOR,
                    "linewidth": 1.2,
                    "hatch": "//",
                }
            )
        ax.barh(y_off, vals[:, j], **bar_kwargs)

    ax.set_yticks(y, labels=labs)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="semibold", color=TEXT_COLOR)
    _style_barh_axis(ax)

    vmax = float(np.nanmax(vals)) if vals.size else 0.0
    ax.set_xlim(0.0, max(1e-6, vmax * 1.15))


def plot_class_distribution_whole_dataset(class_durations: Dict, split_summary: Optional[Dict], out_dir: Path) -> None:
    """Plot overall class distribution across all splits combined.
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    splits = _get_splits_present(class_durations, split_summary or {})

    labels = _get_plot_labels(class_durations, splits)

    union_sec_total = np.zeros(len(labels), dtype=float)
    for split in splits:
        for i, label in enumerate(labels):
            union_sec_total[i] += float(
                class_durations.get("splits", {})
                .get(split, {})
                .get("labels", {})
                .get(label, {})
                .get("interval_union_sec", 0.0)
            )

    denom_sec: float = 0.0
    subtitle: str
    if split_summary and isinstance(split_summary, dict):
        denom_sec = float(split_summary.get("total_sec", 0.0))
        if denom_sec > 0:
            subtitle = "Percent of total recording time (union / total_sec_all)"
        else:
            denom_sec = 0.0

    if denom_sec <= 0.0:
        denom_sec = _get_total_labeled_union_sec(class_durations, splits)
        subtitle = "Percent of labeled time (fallback: union / labeled_union_all)"

    pct_total = (union_sec_total / max(denom_sec, 1e-12)) * 100.0

    order = np.argsort(-union_sec_total) if len(labels) else np.array([], dtype=int)
    labels = [labels[i] for i in order] if len(labels) else []
    pct_total = pct_total[order] if len(order) else pct_total

    pct_mat = pct_total.reshape(-1, 1)
    major_labels, major_pct, minor_labels, minor_pct = _split_major_minor(labels, pct_mat, major_threshold=1.0)

    n_rows = len(labels)
    fig_h = max(5.2, min(0.50 * max(n_rows, 1) + 2.5, 13.5))
    if minor_labels:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(12.5, fig_h),
            gridspec_kw={"height_ratios": [max(1, len(major_labels)), max(1, len(minor_labels))]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12.5, fig_h))
        ax2 = None

    def _plot_barh(ax, labs: List[str], vals: "np.ndarray", title: str) -> None:
        if not labs:
            ax.axis("off")
            return

        y = np.arange(len(labs), dtype=float)
        ax.barh(y, vals[:, 0], height=0.65, label="all", color=WHOLE_DATASET_BAR_COLOR, zorder=3)
        ax.set_yticks(y, labels=labs)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="semibold", color=TEXT_COLOR)
        _style_barh_axis(ax)

        vmax = float(np.nanmax(vals)) if vals.size else 0.0
        ax.set_xlim(0.0, max(1e-6, vmax * 1.15))

    _plot_barh(ax1, major_labels, major_pct, "Whole-dataset class distribution (major classes)")
    if ax2 is not None:
        _plot_barh(ax2, minor_labels, minor_pct, "Whole-dataset class distribution (minor classes)")

    fig.suptitle(
        "Class distribution (train+dev+test combined)",
        y=0.995,
        fontsize=13,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])

    _savefig(fig, out_dir / "class_distribution_pct_total_time_whole_dataset.png")
    _savefig_notitle(fig, out_dir / "class_distribution_pct_total_time_whole_dataset.png")
    plt.close(fig)


def plot_class_distribution(class_durations: Dict, split_summary: Optional[Dict], out_dir: Path) -> None:
    """Plot class distribution by split.
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    splits = _get_splits_present(class_durations, split_summary or {})

    labels = _get_plot_labels(class_durations, splits)
    union_sec = _get_union_sec_matrix(class_durations, labels, splits, np=np)

    using_split_summary = bool(split_summary and isinstance(split_summary, dict))
    total_sec = np.zeros(len(splits), dtype=float)
    if using_split_summary:
        for j, split in enumerate(splits):
            total_sec[j] = float(split_summary.get("splits", {}).get(split, {}).get("total_sec", 0.0))
        using_split_summary = bool((total_sec > 0).all())

    if using_split_summary:
        pct = (union_sec / total_sec.reshape(1, -1)) * 100.0
        subtitle = "Percent of total recording time (union / total_sec)"
    else:
        pct = np.zeros_like(union_sec)
        for j, split in enumerate(splits):
            for i, label in enumerate(labels):
                pct[i, j] = float(
                    class_durations.get("splits", {})
                    .get(split, {})
                    .get("labels", {})
                    .get(label, {})
                    .get("pct_of_labeled_union", 0.0)
                )
        subtitle = "Percent of labeled time (fallback: pct_of_labeled_union)"

    order = np.argsort(-union_sec.sum(axis=1)) if len(labels) else np.array([], dtype=int)
    labels = [labels[i] for i in order] if len(labels) else []
    pct = pct[order] if len(order) else pct

    major_labels, major_pct, minor_labels, minor_pct = _split_major_minor(labels, pct, major_threshold=1.0)

    n_rows = (len(major_labels) if major_labels else 0) + (len(minor_labels) if minor_labels else 0)
    fig_h = max(5.2, min(0.50 * max(n_rows, 1) + 2.5, 13.5))

    if minor_labels:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(12.5, fig_h),
            gridspec_kw={"height_ratios": [max(1, len(major_labels)), max(1, len(minor_labels))]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12.5, fig_h))
        ax2 = None

    series_colors = [SPLIT_BAR_COLORS.get(split, SPLIT_BAR_FALLBACK[j % len(SPLIT_BAR_FALLBACK)]) for j, split in enumerate(splits)]

    _plot_grouped_barh_panel(ax1, major_labels, major_pct, splits, series_colors, "Class distribution (major classes)")
    if ax2 is not None:
        _plot_grouped_barh_panel(ax2, minor_labels, minor_pct, splits, series_colors, "Class distribution (minor classes)")

    handles, labels_leg = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(
            handles,
            labels_leg,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            frameon=False,
            borderaxespad=0.0,
            prop={"weight": "semibold", "size": 11},
        )

    fig.suptitle("Class distribution by split", y=0.995, fontsize=13, fontweight="semibold", color=TEXT_COLOR)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])

    _savefig(fig, out_dir / "class_distribution_pct_total_time.png")
    _savefig_notitle(fig, out_dir / "class_distribution_pct_total_time.png")
    plt.close(fig)


def plot_class_distribution_combined(class_durations: Dict, split_summary: Optional[Dict], out_dir: Path) -> None:
    """Plot class distribution by split with combined total column.
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    splits = _get_splits_present(class_durations, split_summary or {})
    labels = _get_plot_labels(class_durations, splits)
    union_sec = _get_union_sec_matrix(class_durations, labels, splits, np=np)
    union_sec_total = union_sec.sum(axis=1)

    using_split_summary = bool(split_summary and isinstance(split_summary, dict))
    total_sec = np.zeros(len(splits), dtype=float)
    denom_total_sec = 0.0
    if using_split_summary:
        for j, split in enumerate(splits):
            total_sec[j] = float(split_summary.get("splits", {}).get(split, {}).get("total_sec", 0.0))
        denom_total_sec = float(split_summary.get("total_sec", 0.0))
        using_split_summary = bool((total_sec > 0).all()) and denom_total_sec > 0.0

    if using_split_summary:
        pct_splits = (union_sec / total_sec.reshape(1, -1)) * 100.0
        pct_total = (union_sec_total / denom_total_sec) * 100.0
    else:
        pct_splits = np.zeros_like(union_sec)
        for j, split in enumerate(splits):
            for i, label in enumerate(labels):
                pct_splits[i, j] = float(
                    class_durations.get("splits", {})
                    .get(split, {})
                    .get("labels", {})
                    .get(label, {})
                    .get("pct_of_labeled_union", 0.0)
                )
        labeled_union_total = _get_total_labeled_union_sec(class_durations, splits)
        pct_total = (union_sec_total / max(labeled_union_total, 1e-12)) * 100.0

    pct_combined = np.concatenate([pct_splits, pct_total.reshape(-1, 1)], axis=1)

    order = np.argsort(-union_sec_total) if len(labels) else np.array([], dtype=int)
    labels = [labels[i] for i in order] if len(labels) else []
    pct_combined = pct_combined[order] if len(order) else pct_combined

    major_labels, major_pct, minor_labels, minor_pct = _split_major_minor(labels, pct_combined, major_threshold=1.0)

    n_rows = (len(major_labels) if major_labels else 0) + (len(minor_labels) if minor_labels else 0)
    fig_h = max(5.2, min(0.50 * max(n_rows, 1) + 2.5, 13.5))

    if minor_labels:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(12.5, fig_h),
            gridspec_kw={"height_ratios": [max(1, len(major_labels)), max(1, len(minor_labels))]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12.5, fig_h))
        ax2 = None

    series_names = [*splits, "total"]
    series_colors = [
        *[SPLIT_BAR_COLORS.get(split, SPLIT_BAR_FALLBACK[j % len(SPLIT_BAR_FALLBACK)]) for j, split in enumerate(splits)],
        COMBINED_TOTAL_BAR_COLOR,
    ]

    _plot_grouped_barh_panel(
        ax1,
        major_labels,
        major_pct,
        series_names,
        series_colors,
        "Combined class distribution (major classes)",
        highlight_series=["total"],
    )
    if ax2 is not None:
        _plot_grouped_barh_panel(
            ax2,
            minor_labels,
            minor_pct,
            series_names,
            series_colors,
            "Combined class distribution (minor classes)",
            highlight_series=["total"],
        )

    handles, labels_leg = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(
            handles,
            labels_leg,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            frameon=False,
            borderaxespad=0.0,
            prop={"weight": "semibold", "size": 11},
        )

    fig.suptitle(
        "Class distribution by split and total",
        y=0.995,
        fontsize=13,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])

    _savefig(fig, out_dir / "class_distribution_pct_total_time_combined.png")
    _savefig_notitle(fig, out_dir / "class_distribution_pct_total_time_combined.png")
    plt.close(fig)


def plot_total_annotation_counts(class_durations: Dict, out_dir: Path) -> None:
    """Plot total annotation counts per class across all splits.
    
    Args:
        class_durations: Class duration statistics dictionary.
        out_dir: Output directory for PNG files.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    splits = _get_splits_present(class_durations)
    labels = _get_plot_labels(class_durations, splits)
    annotation_counts = _get_annotation_count_matrix(class_durations, labels, splits, np=np)
    annotation_total = annotation_counts.sum(axis=1)
    annotation_pct = (annotation_total / max(float(annotation_total.sum()), 1e-12)) * 100.0

    order = np.argsort(-annotation_total) if len(labels) else np.array([], dtype=int)
    labels = [labels[i] for i in order] if len(labels) else []
    annotation_total = annotation_total[order] if len(order) else annotation_total
    annotation_pct = annotation_pct[order] if len(order) else annotation_pct

    pct_mat = annotation_pct.reshape(-1, 1)
    major_labels, _, minor_labels, _ = _split_major_minor(labels, pct_mat, major_threshold=1.0)

    major_count = annotation_total[: len(major_labels)] if major_labels else np.zeros(0, dtype=float)
    minor_count = annotation_total[len(major_labels) :] if minor_labels else np.zeros(0, dtype=float)

    n_rows = max(len(major_labels), len(minor_labels), 1)
    fig_h = max(6.0, min(0.60 * n_rows + 2.8, 14.5))
    if minor_labels:
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(12.5, fig_h),
            gridspec_kw={"width_ratios": [max(1, len(major_labels)), max(1, len(minor_labels))]},
        )
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12.5, fig_h))
        ax2 = None

    def _plot_vertical_count_panel(
        ax,
        labs: List[str],
        vals: "np.ndarray",
        title: str,
        *,
        y_ticks: Optional[List[float]] = None,
        y_ticklabels: Optional[List[str]] = None,
    ) -> None:
        """Plot vertical bar chart with annotation count labels.
        
        Args:
            ax: Matplotlib axis object.
            labs: List of bar labels.
            vals: Numpy array of bar heights (annotation counts).
            title: Axis title.
            y_ticks: Optional custom y-axis tick positions.
            y_ticklabels: Optional custom y-axis tick labels.
        """
        if not labs:
            ax.axis("off")
            return

        x = np.arange(len(labs), dtype=float)
        ax.bar(x, vals, width=0.65, color=WHOLE_DATASET_BAR_COLOR, zorder=3)
        ax.set_xticks(x, labels=labs)
        ax.set_title(title, fontsize=12, fontweight="semibold", color=TEXT_COLOR)
        ax.set_ylabel("Annotation count", fontsize=11, fontweight="semibold", color=TEXT_COLOR)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.5, zorder=0)
        ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=11, rotation=0)
        ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("semibold")
            tick.set_ha("center")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("semibold")

        vmax = float(np.nanmax(vals)) if vals.size else 0.0
        if y_ticks is not None:
            top = float(max(y_ticks)) if y_ticks else 1.0
            ax.set_ylim(0.0, max(top * 1.14, vmax * 1.10, 1.0))
            ax.set_yticks(y_ticks)
            if y_ticklabels is not None:
                ax.set_yticklabels(y_ticklabels)
        else:
            ax.set_ylim(0.0, max(1.0, vmax * 1.18))

        for xi, value in zip(x, vals):
            ax.text(
                xi,
                value + max(1.0, vmax * 0.015),
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color=TEXT_COLOR,
                rotation=0,
            )

    _plot_vertical_count_panel(
        ax1,
        major_labels,
        major_count,
        "Total annotations per label (major classes)",
        y_ticks=[0, 50000, 100000, 150000],
        y_ticklabels=["0", "50k", "100k", "150k"],
    )
    if ax2 is not None:
        _plot_vertical_count_panel(
            ax2,
            minor_labels,
            minor_count,
            "Total annotations per label (minor classes)",
            y_ticks=[0, 200, 400],
            y_ticklabels=["0", "200", "400"],
        )

    fig.suptitle(
        "Total annotation counts (train+dev+test combined)",
        y=0.995,
        fontsize=13,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])

    _savefig(fig, out_dir / "class_annotation_counts_total.png")
    _savefig_notitle(fig, out_dir / "class_annotation_counts_total.png")
    plt.close(fig)


def plot_annotation_length_distribution(class_durations: Dict, split_summary: Optional[Dict], out_dir: Path) -> None:
    """Plot boxplot of annotation duration distribution by class.
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    import matplotlib.pyplot as plt
    import statistics

    splits = _get_splits_present(class_durations, split_summary or {})
    labels = _get_plot_labels(class_durations, splits)
    durations_by_label = _get_annotation_durations_by_label(class_durations, labels, splits)
    labels = [label for label in labels if durations_by_label.get(label)]
    if not labels:
        raise SystemExit(
            "class_durations.json is missing per-annotation duration lists required for "
            "class_annotation_length_distribution.png. Regenerate it with "
            "dataset_split/report_class_durations.py and rerun plotting."
        )

    labels = sorted(
        labels,
        key=lambda label: (
            statistics.median(durations_by_label[label]),
            statistics.fmean(durations_by_label[label]),
            label,
        ),
        reverse=True,
    )
    duration_series = [durations_by_label[label] for label in labels]

    fig_w = max(11.5, 1.15 * len(labels) + 4.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 7.2))

    box = ax.boxplot(
        duration_series,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COMBINED_TOTAL_EDGE_COLOR, "linewidth": 2.0},
        whiskerprops={"color": TEXT_COLOR, "linewidth": 1.4},
        capprops={"color": TEXT_COLOR, "linewidth": 1.4},
        boxprops={"edgecolor": TEXT_COLOR, "linewidth": 1.25},
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(ANNOTATION_LENGTH_BOX_COLORS[idx % len(ANNOTATION_LENGTH_BOX_COLORS)])
        patch.set_alpha(0.58)
        patch.set_zorder(3)

    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5, zorder=0)
    ax.set_ylabel("Annotation duration (seconds)", fontsize=11, fontweight="semibold", color=TEXT_COLOR)
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=11, rotation=0)
    ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=11)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("semibold")
        tick.set_ha("center")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("semibold")

    visible_upper = max(
        (max(float(value) for value in whisker.get_ydata()) for whisker in box["whiskers"]),
        default=0.0,
    )
    ax.set_ylim(0.0, max(1.0, visible_upper * 1.10))
    ax.set_title(
        "Annotation length distribution by class (train+dev+test combined)",
        fontsize=13,
        fontweight="semibold",
        color=TEXT_COLOR,
    )

    fig.tight_layout()
    _savefig(fig, out_dir / "class_annotation_length_distribution.png")
    _savefig_notitle(fig, out_dir / "class_annotation_length_distribution.png")
    plt.close(fig)


def _plot_annotation_length_distribution_grid_middle95(
    class_durations: Dict,
    split_summary: Optional[Dict],
    out_dir: Path,
    *,
    nrows: int,
    ncols: int,
    fig_size: Tuple[float, float],
    out_name: str,
    title_suffix: str,
) -> None:
    """Plot histogram grid of annotation durations (middle 95% per class).
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
        nrows: Number of rows in subplot grid.
        ncols: Number of columns in subplot grid.
        fig_size: Figure size tuple (width, height).
        out_name: Output filename.
        title_suffix: Suffix for figure title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    splits = _get_splits_present(class_durations, split_summary or {})
    labels_available = set(_get_plot_labels(class_durations, splits))
    plot_labels = [label for label in TARGET_LABELS if label in labels_available]
    durations_by_label = _get_annotation_durations_by_label(class_durations, plot_labels, splits)

    labels_by_sample_size = sorted(
        TARGET_LABELS,
        key=lambda label: len(durations_by_label.get(label, [])),
        reverse=True,
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, constrained_layout=True)
    flat_axes = list(np.atleast_1d(axes).ravel())

    for idx, label in enumerate(labels_by_sample_size):
        ax = flat_axes[idx]
        color = ANNOTATION_LENGTH_BOX_COLORS[idx % len(ANNOTATION_LENGTH_BOX_COLORS)]
        raw_durations = durations_by_label.get(label, [])

        if not raw_durations:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold",
                color=TEXT_COLOR,
                transform=ax.transAxes,
            )
            ax.set_title(label, fontsize=12, fontweight="semibold", color=TEXT_COLOR)
            ax.set_xlabel("Duration (s)", fontsize=10, fontweight="semibold", color=TEXT_COLOR)
            ax.set_ylabel("Count", fontsize=10, fontweight="semibold", color=TEXT_COLOR)
            ax.grid(axis="y", alpha=0.35)
            ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=10)
            ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=10)
            continue

        data = np.asarray(raw_durations, dtype=float)
        p_low, p_high = np.percentile(data, [2.5, 97.5]) 
        middle95 = data[(data >= p_low) & (data <= p_high)]

        if middle95.size == 0:
            ax.text(
                0.5,
                0.5,
                "No data after filtering",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold",
                color=TEXT_COLOR,
                transform=ax.transAxes,
            )
            ax.set_title(f"{label} (middle 95%)", fontsize=12, fontweight="semibold", color=TEXT_COLOR)
            ax.set_xlabel("Duration (s)", fontsize=10, fontweight="semibold", color=TEXT_COLOR)
            ax.set_ylabel("Count", fontsize=10, fontweight="semibold", color=TEXT_COLOR)
            ax.grid(axis="y", alpha=0.35)
            ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=10)
            ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=10)
            continue

        bins = int(np.clip(np.sqrt(middle95.size), 15, 80))
        if label in {"shiv", "elpp"}:
            bins = int(np.clip(bins * 2, 20, 160)) 
        ax.hist(middle95, bins=bins, color=color, edgecolor=TEXT_COLOR, linewidth=0.8, alpha=0.8)
        ax.set_title(
            f"{label} (n={middle95.size:,})",
            fontsize=12,
            fontweight="semibold",
            color=TEXT_COLOR,
        )
        ax.set_xlabel("Duration (s)", fontsize=10, fontweight="semibold", color=TEXT_COLOR)
        ax.set_ylabel("Count", fontsize=10, fontweight="semibold", color=TEXT_COLOR)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=10)
        ax.tick_params(axis="y", colors=TEXT_COLOR, labelsize=10)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("semibold")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("semibold")

    for ax in flat_axes[len(labels_by_sample_size) :]:
        ax.axis("off")

    fig.suptitle(f"Class annotation length distributions (middle 95% per class, {title_suffix})", fontsize=14, fontweight="semibold", color=TEXT_COLOR)

    out_path = out_dir / out_name
    _savefig(fig, out_path)
    _savefig_notitle(fig, out_path)
    plt.close(fig)


def plot_annotation_length_distribution_2x3_middle95(
    class_durations: Dict,
    split_summary: Optional[Dict],
    out_dir: Path,
) -> None:
    """Plot 2x3 grid of class annotation length distributions (middle 95%).
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    _plot_annotation_length_distribution_grid_middle95(
        class_durations,
        split_summary,
        out_dir,
        nrows=2,
        ncols=3,
        fig_size=(15.0, 8.5),
        out_name="class_annotation_length_distribution_2x3_middle95.png",
        title_suffix="2x3 by sample size",
    )


def plot_annotation_length_distribution_3x2_middle95(
    class_durations: Dict,
    split_summary: Optional[Dict],
    out_dir: Path,
) -> None:
    """Plot 3x2 grid of class annotation length distributions (middle 95%).
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    _plot_annotation_length_distribution_grid_middle95(
        class_durations,
        split_summary,
        out_dir,
        nrows=3,
        ncols=2,
        fig_size=(12.5, 11.0),
        out_name="class_annotation_length_distribution_3x2_middle95.png",
        title_suffix="3x2 by sample size",
    )


def plot_annotation_length_distribution_6x1_middle95(
    class_durations: Dict,
    split_summary: Optional[Dict],
    out_dir: Path,
) -> None:
    """Plot 6x1 grid of class annotation length distributions (middle 95%).
    
    Args:
        class_durations: Class duration statistics dictionary.
        split_summary: Optional split summary with total recording times.
        out_dir: Output directory for PNG files.
    """
    _plot_annotation_length_distribution_grid_middle95(
        class_durations,
        split_summary,
        out_dir,
        nrows=6,
        ncols=1,
        fig_size=(8.5, 17.5),
        out_name="class_annotation_length_distribution_6x1_middle95.png",
        title_suffix="6x1 by sample size",
    )


TARGET_LABELS = ["chew", "elec", "elpp", "eyem", "musc", "shiv"]


def _is_artifact_annotation(description: str) -> bool:
    """Check if an annotation description indicates an artifact.
    
    Args:
        description: Annotation description string.
    
    Returns:
        True if description matches artifact-related tokens.
    """
    desc = (description or "").strip().lower()
    if not desc:
        return False
    artifact_tokens = set(TARGET_LABELS) | {"artifact", "artf"}
    parts = [token for token in re.split(r"[^a-z0-9]+", desc) if token]
    return any(part in artifact_tokens for part in parts)


def _iter_split_edf_files(split_root: Path) -> List[Tuple[str, Path]]:
    """Find all EDF files across train/dev/test splits.
    
    Args:
        split_root: Root directory containing split subdirectories.
    
    Returns:
        List of (split_name, edf_path) tuples in sorted order.
    """
    split_names = ["train", "dev", "test"]
    files: List[Tuple[str, Path]] = []
    for split in split_names:
        split_dir = split_root / split
        if not split_dir.exists():
            continue
        for edf_path in sorted(split_dir.rglob("*.edf")):
            files.append((split, edf_path))
    return files


def plot_artifact_eeg_window(
    split_root: Path,
    out_dir: Path,
    *,
    window_sec: float = 4.0,
    max_channels: Optional[int] = None,
    preferred_label: Optional[str] = None,
    prefer_shortest_match: bool = False,
    preferred_match_rank: int = 1,
) -> Optional[Path]:
    """Plot a classic multi-channel EEG window containing an artifact annotation.
    
    Args:
        split_root: Root directory containing split subdirectories with EDFs.
        out_dir: Output directory for PNG file.
        window_sec: Time window duration in seconds (default 4.0).
        max_channels: Maximum number of EEG channels to plot (None for all).
        preferred_label: Preferred artifact label to search for.
        prefer_shortest_match: If True, find shortest matching artifact within class.
        preferred_match_rank: Rank of preferred match to use (1-indexed).
    
    Returns:
        Path to output PNG file, or None if no artifact found or mne unavailable.
    """
    try:
        import mne
    except Exception as e:
        print(f"Skipping artifact window plot: mne not available ({e})")
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    preferred = (preferred_label or "").strip().lower()
    selected = None
    preferred_candidates: List[Dict[str, object]] = []
    fallback_selected = None
    for split, edf_path in _iter_split_edf_files(split_root):
        try:
            raw_info = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        except Exception:
            continue

        for onset, duration, description in zip(
            raw_info.annotations.onset,
            raw_info.annotations.duration,
            raw_info.annotations.description,
        ):
            label = str(description)
            if _is_artifact_annotation(label):
                candidate = {
                    "split": split,
                    "path": edf_path,
                    "onset": float(onset),
                    "duration": max(float(duration), 0.0),
                    "label": label,
                }
                if fallback_selected is None:
                    fallback_selected = candidate

                if preferred:
                    desc_tokens = [token for token in re.split(r"[^a-z0-9]+", label.strip().lower()) if token]
                    if preferred in desc_tokens:
                        if prefer_shortest_match:
                            preferred_candidates.append(candidate)
                        else:
                            selected = candidate
                            break
                else:
                    selected = candidate
                    break
        if selected is not None:
            break

    if selected is None and preferred and preferred_candidates:
        preferred_candidates = sorted(
            preferred_candidates,
            key=lambda cand: (
                float(cand.get("duration", 0.0)),
                str(cand.get("path", "")),
                float(cand.get("onset", 0.0)),
            ),
        )
        rank_idx = max(0, int(preferred_match_rank) - 1)
        rank_idx = min(rank_idx, len(preferred_candidates) - 1)
        selected = preferred_candidates[rank_idx]

    if selected is None and fallback_selected is not None:
        selected = fallback_selected

    if selected is None:
        print("Skipping artifact window plot: no artifact annotations found in split EDF files.")
        return None

    raw = mne.io.read_raw_edf(str(selected["path"]), preload=True, verbose=False)
    data_duration = float(raw.times[-1]) if raw.n_times > 0 else 0.0
    if data_duration <= 0.0:
        print(f"Skipping artifact window plot: empty recording in {selected['path'].name}.")
        return None

    eeg_indices = [
        idx
        for idx, name in enumerate(raw.ch_names)
        if name.strip().upper() != "ANOT" and "EEG" in name.upper()
    ]
    if not eeg_indices:
        eeg_indices = [idx for idx, name in enumerate(raw.ch_names) if name.strip().upper() != "ANOT"]
    if not eeg_indices:
        print(f"Skipping artifact window plot: no plottable channels in {selected['path'].name}.")
        return None

    clinical_order = [
        "FP1", "FP2",
        "F7", "F3", "FZ", "F4", "F8",
        "T3", "C3", "CZ", "C4", "T4",
        "T5", "P3", "PZ", "P4", "T6",
        "O1", "O2",
    ]

    def _canonical_ch_name(name: str) -> str:
        """Normalize channel name to standard form for ordering."""
        canon = name.strip().upper()
        canon = canon.replace("EEG ", "")
        for suffix in ("_AVG", "_TCP", "-REF", "-LE"):
            if canon.endswith(suffix):
                canon = canon[: -len(suffix)]
        return canon

    clinical_rank = {label: idx for idx, label in enumerate(clinical_order)}
    ordered_eeg_indices = sorted(
        eeg_indices,
        key=lambda idx: (
            clinical_rank.get(_canonical_ch_name(raw.ch_names[idx]), len(clinical_order)),
            _canonical_ch_name(raw.ch_names[idx]),
        ),
    )

    if max_channels is None or max_channels <= 0:
        picked_indices = ordered_eeg_indices
    else:
        picked_indices = ordered_eeg_indices[:max_channels]
    raw = raw.copy().pick(picked_indices)
    sfreq = float(raw.info["sfreq"])

    artifact_onset = float(selected["onset"])
    artifact_duration = float(selected["duration"])
    artifact_end = artifact_onset + artifact_duration
    artifact_mid = artifact_onset + max(artifact_duration, 0.0) / 2.0

    win_start = max(0.0, artifact_mid - window_sec / 2.0)
    win_end = min(data_duration, win_start + window_sec)
    win_start = max(0.0, win_end - window_sec)

    start_samp = int(np.floor(win_start * sfreq))
    stop_samp = int(np.ceil(win_end * sfreq))
    if stop_samp <= start_samp:
        print(f"Skipping artifact window plot: invalid sample range in {selected['path'].name}.")
        return None

    segment = raw.get_data(start=start_samp, stop=stop_samp)
    segment = segment - segment.mean(axis=1, keepdims=True)
    n_times = segment.shape[1]
    times = np.arange(n_times, dtype=float) / sfreq + win_start

    amp_ref = float(np.percentile(np.abs(segment), 95)) if segment.size else 1.0
    if amp_ref <= 0.0:
        amp_ref = 1.0
    spacing = amp_ref * 4.0  

    fig_h = max(6.0, 0.44 * len(raw.ch_names) + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(15.5, fig_h))

    offsets = []
    for idx, ch_name in enumerate(raw.ch_names):
        y_offset = (len(raw.ch_names) - 1 - idx) * spacing
        offsets.append(y_offset)
        ax.plot(times, segment[idx] + y_offset, color="#1F2933", linewidth=0.9)

    ax.set_yticks(offsets)
    ax.set_yticklabels(raw.ch_names, fontsize=9, fontweight="semibold", color=TEXT_COLOR)
    ax.set_xlim(win_start, win_end)
    ax.set_xlabel("Time (s)", fontsize=11, fontweight="semibold", color=TEXT_COLOR)
    ax.set_ylabel("Channels", fontsize=11, fontweight="semibold", color=TEXT_COLOR)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("semibold")

    title_label = preferred if preferred else str(selected["label"])
    ax.set_title(
        f"Classic EEG window ({window_sec:.1f}s) containing artifact: {title_label}",
        fontsize=12,
        fontweight="semibold",
        color=TEXT_COLOR,
    )

    fig.tight_layout()
    label_suffix = preferred if preferred else "artifact"
    window_label = f"{int(round(window_sec))}s" if abs(window_sec - round(window_sec)) < 1e-9 else f"{window_sec:g}s"
    out_path = out_dir / f"artifact_eeg_window_{window_label}_{label_suffix}.png"
    _savefig(fig, out_path)
    _savefig_notitle(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Generate all plots from class_durations.json and split_summary.json."""
    _require_matplotlib()

    p = argparse.ArgumentParser(description="Plot split comparisons from class_durations.json and split_summary.json")
    p.add_argument(
        "--split-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "01_tcp_ar_split",
        help="Directory containing class_durations.json (and optionally split_summary.json)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "data_plots",
        help="Where to save PNG plots",
    )
    args = p.parse_args()
    split_root: Path = args.split_root
    out_dir: Path = args.out_dir

    class_durations_path = split_root / "class_durations.json"
    if not class_durations_path.exists():
        raise SystemExit(f"Missing: {class_durations_path}")

    class_durations = _read_json(class_durations_path)

    split_summary_path = split_root / "split_summary.json"
    split_summary = _read_json(split_summary_path) if split_summary_path.exists() else None

    plot_class_distribution(class_durations, split_summary, out_dir)
    plot_class_distribution_whole_dataset(class_durations, split_summary, out_dir)
    plot_class_distribution_combined(class_durations, split_summary, out_dir)
    plot_total_annotation_counts(class_durations, out_dir)
    plot_annotation_length_distribution(class_durations, split_summary, out_dir)
    plot_annotation_length_distribution_2x3_middle95(class_durations, split_summary, out_dir)
    plot_annotation_length_distribution_3x2_middle95(class_durations, split_summary, out_dir)
    plot_annotation_length_distribution_6x1_middle95(class_durations, split_summary, out_dir)
    artifact_plot_path = plot_artifact_eeg_window(
        split_root,
        out_dir,
        window_sec=4.0,
        preferred_label="musc",
        prefer_shortest_match=True,
        preferred_match_rank=1,
    )

    print(f"Wrote: {out_dir / 'class_distribution_pct_total_time.png'}")
    print(f"Wrote: {out_dir / 'class_distribution_pct_total_time_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_distribution_pct_total_time_whole_dataset.png'}")
    print(f"Wrote: {out_dir / 'class_distribution_pct_total_time_whole_dataset_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_distribution_pct_total_time_combined.png'}")
    print(f"Wrote: {out_dir / 'class_distribution_pct_total_time_combined_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_counts_total.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_counts_total_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_2x3_middle95.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_2x3_middle95_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_3x2_middle95.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_3x2_middle95_notitle.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_6x1_middle95.png'}")
    print(f"Wrote: {out_dir / 'class_annotation_length_distribution_6x1_middle95_notitle.png'}")
    if artifact_plot_path is not None:
        print(f"Wrote: {artifact_plot_path}")
        print(f"Wrote: {_notitle_path(artifact_plot_path)}")


if __name__ == "__main__":
    main()
