# Data Preprocessing Tools

Quick start for EEG signal preprocessing and visualization.

## Setup

1. Edit the config file: `preprocess_mne_cfg/TUAR_100_Avg.yaml`
   - Set `input_path` and `output_path` for your data folders

## Preprocess Data

```bash
python preprocess_mne.py --config preprocess_mne_cfg/TUAR_100_Avg.yaml
```

Optional: Set `decompose_composite_labels: true` in the config to split multi-labels (e.g., `eyem_musc` → separate `eyem` and `musc` annotations).

## Plot EEG Signals

```bash
python plot_EEG.py --inputs ./data/preprocessed/ --output ./plots/
```

Generates interactive HTML viewers for each EDF file with annotations overlaid.

## Utilities

- `artifact_detection.py`: Detect and flag artifact regions in signals
- `histogram_plotter.py`: Visualize amplitude distributions and statistical analysis
- `rename_annotations.py`: Batch find-and-replace in annotation CSVs

## Attribution

This work was developed as part of an MSc thesis. The code foundation was built by the AI Research Team at the Institute of Mathematics, ELTE, and has been extended and adapted for this project.