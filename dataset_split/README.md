# Dataset Split & Analysis Tools

Quick start for train/dev/test splitting, duration reporting, and visualization.

## Overview

This folder contains tools for:
- **Patient-wise splitting** with length weighting and optional label balancing
- **Duration reporting** to analyze annotation statistics across splits
- **Data visualization** for class distribution and annotation patterns

## Split Data

Create patient-wise train/dev/test splits with length-weighted balancing:

```bash
python justsplit_patientwise.py \
  --source-dir ./data/processed \
  --out-dir ./splits \
  --seed 42 \
  --balance-labels
```

Common options:
- `--balance-labels`: Balance per-label annotated time across splits (default: on)
- `--balance-label-weight`: Weight of label-balance vs. time-balance (default: 1.0)
- `--balance-opt-swap-iters`: Random patient-swap iterations for optimization
- `--drop-label`: Labels to exclude from balancing (e.g., `artifact`, `bad_acq_skip`)
- `--dry-run`: Preview split statistics without writing files
- `--relative-symlinks`: Create portable relative symlinks

## Report Durations

Generate per-class annotation duration statistics:

```bash
python report_class_durations.py \
  --split-root ./splits \
  --out-dir ./splits
```

Outputs:
- `class_durations.csv` — Duration summary table
- `class_durations.json` — Structured duration data by split

## Plot Analysis

Visualize class distribution and annotation patterns:

```bash
python plot_data_analysis.py \
  --split-root ./splits \
  --out-dir ./plots
```

Generates:
- Class distribution comparisons across splits
- Annotation count distributions
- Annotation length distributions with percentile ranges
- Artifact EEG window examples

## Order of Execution

1. `justsplit_patientwise.py` — Create splits
2. `report_class_durations.py` — Analyze durations
3. `plot_data_analysis.py` — Visualize results
