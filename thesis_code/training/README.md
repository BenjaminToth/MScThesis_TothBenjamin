# Training Scripts

This directory contains training scripts for EEG artifact detection models and synthetic data generators.

## Overview

- **train.py** - Train classification models for EEG artifact detection
- **train_ldm.py** - Train per-class Latent Diffusion Models (VAE + DDPM) for synthetic data generation
- **train_wgangp.py** - Train per-class WGAN-GP generators for synthetic data generation

---

## 1. train.py - EEG Artifact Classification

Train a classification model to detect multiple artifact types in EEG signals.

### Basic Usage

```bash
python training/train.py \
  --split-root data/01_tcp_ar_split \
  --model eegnet \
  --epochs 30 \
  --batch-size 32 \
  --lr 3e-4
```

### Output

Saves a checkpoint file with:
- Trained model weights
- Model configuration
- Run settings
- Performance metrics on dev set

### Parameters

#### Data Loading
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--split-root` | str | `data/01_tcp_ar_split` | Root directory containing train/dev/test splits |
| `--data-subdir` | str | `01_tcp_ar` | Subdirectory name within each split |
| `--batch-size` | int | `32` | Batch size for training |
| `--num-workers` | int | `4` | Number of workers for data loading |

#### Signal Processing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sfreq` | float | `250.0` | Sampling frequency (Hz) |
| `--window-sec` | float | `8.0` | Window length in seconds |
| `--stride-sec` | float | `0.5` | Stride between windows in seconds |
| `--min-overlap-sec` | float | `0.25` | Minimum overlap with artifact labels (seconds) |
| `--min-overlap-frac-artifact` | float | `0.25` | Minimum fraction of artifact window that must overlap |
| `--no-normalize` | flag | `False` | Disable z-score normalization |

#### Data Augmentation - Standard
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--aug-pink-noise` | flag | `False` | Enable pink-noise (1/f) augmentation |
| `--aug-pink-noise-prob` | float | `0.5` | Probability of pink-noise application |
| `--aug-pink-noise-snr-db` | float | `20.0` | Signal-to-noise ratio (dB) for pink noise |
| `--aug-time-domain` | flag | `False` | Enable time-domain cropping + jitter |
| `--aug-time-domain-crop-frac` | float | `0.9` | Fraction of window to keep during cropping |
| `--aug-time-domain-shift-frac` | float | `0.05` | Max positional jitter as fraction of window |
| `--aug-segment-recombination` | flag | `False` | Enable segment recombination with phase-aware alignment |
| `--aug-segment-recombination-prob` | float | `0.5` | Probability of segment recombination |
| `--aug-segment-recombination-n-segments` | int | `8` | Number of segments for recombination |
| `--aug-channel-dropout` | flag | `False` | Enable channel dropout augmentation |
| `--aug-channel-dropout-prob` | float | `0.5` | Probability of channel dropout |
| `--aug-channel-dropout-frac` | float | `0.1` | Fraction of channels to drop |
| `--aug-mixup` | flag | `False` | Enable Mixup (blend 90% target + 10% noise) |
| `--aug-mixup-prob` | float | `0.5` | Probability of Mixup |
| `--aug-gaussian-noise` | flag | `False` | Enable Gaussian (white) noise augmentation |
| `--aug-gaussian-noise-prob` | float | `0.5` | Probability of Gaussian noise |
| `--aug-gaussian-noise-snr-db` | float | `20.0` | Signal-to-noise ratio (dB) for Gaussian noise |

#### Data Augmentation - GAN (WGAN-GP)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--aug-wgangp-major` | flag | `False` | Augment with WGAN-GP synthetics for major classes (chew/elec/eyem/musc) |
| `--aug-wgangp-n-per-class` | int | `10000` | Number of synthetic windows per major class |
| `--aug-wgangp-root` | str | `results/saved_wgangp` | Root folder containing per-class generators |

#### Data Augmentation - LDM (Diffusion)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--aug-ldm-major` | flag | `False` | Augment with LDM synthetics for major classes (chew/elec/eyem/musc) |
| `--aug-ldm-n-per-class` | int | `10000` | Number of synthetic windows per major class |
| `--aug-ldm-root` | str | `results/saved_ldm` | Root folder containing per-class LDM checkpoints |
| `--aug-ldm-num-inference-steps` | int | `-1` | Override LDM inference steps (>0). Use -1 for checkpoint setting |

#### Model
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | `eegnet` | Model architecture: `eegnet`, `eegnex`, `eegconformer`, `cnnlstm`, `eeginceptionerp` |

#### Training
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | `30` | Number of training epochs |
| `--lr` | float | `3e-4` | Learning rate |
| `--weight-decay` | float | `0.0` | L2 weight decay |
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda`, or `cuda:N` |
| `--gpu` | int | `0` | GPU index when using `auto` or `cuda` |
| `--seed` | int | `42` | Random seed (>=0 fixed; <0 random) |

#### Data Sampling
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-samples-per-epoch` | int | `50000` | Max training windows per epoch (0 = no cap) |
| `--sampler-warmup-epochs` | int | `1` | Epochs of uniform sampling before repeat-factor sampling |
| `--sampler` | str | `repeat_factor` | Sampling strategy: `repeat_factor`, `major_only`, `none` |
| `--repeat-threshold` | float | `0.1` | Threshold for repeat-factor sampling |

#### Loss Function (Asymmetric Sigmoid Loss)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gamma-pos` | float | `1.0` | Focus loss exponent for positive labels |
| `--gamma-neg` | float | `2.0` | Focus loss exponent for negative labels |
| `--clip` | float | `0.05` | Clipping margin for loss stability |

#### Inference/Evaluation
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pred-threshold` | float | `0.5` | Sigmoid probability threshold for predictions |
| `--no-tune-thresholds` | flag | `False` | Disable per-class threshold tuning on dev set |
| `--threshold-grid-size` | int | `101` | Number of thresholds to sweep per class |

#### Miscellaneous
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no-progress` | flag | `False` | Disable tqdm progress bars |
| `--run-name` | str | `""` | Custom run name (timestamp used if empty) |

---

## 2. train_ldm.py - Latent Diffusion Model (VAE + DDPM)

Train a per-class Latent Diffusion Model for generating synthetic EEG artifact windows.

### Basic Usage

```bash
python training/train_ldm.py \
  --split-root data/01_tcp_ar_split \
  --class-name chew \
  --vae-epochs 10 \
  --diffusion-epochs 20
```

### Output

For each class, saves:
- `vae_last.pt` - Trained VAE weights
- `unet_last.pt` - Trained diffusion UNet weights
- `settings.json` - Training configuration
- `history.json` - Loss history

### Parameters

#### Data Loading
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--split-root` | str | `data/01_tcp_ar_split` | Root directory containing train/dev/test splits |
| `--data-subdir` | str | `01_tcp_ar` | Subdirectory name within each split |
| `--batch-size` | int | `32` | Batch size for training |
| `--num-workers` | int | `4` | Number of workers for data loading |

#### Signal Processing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sfreq` | float | `250.0` | Sampling frequency (Hz) |
| `--window-sec` | float | `8.0` | Window length in seconds |
| `--stride-sec` | float | `0.5` | Stride between windows in seconds |
| `--min-overlap-sec` | float | `0.25` | Minimum overlap with artifact labels (seconds) |
| `--min-overlap-frac-artifact` | float | `0.25` | Minimum fraction of artifact window that must overlap |
| `--no-normalize` | flag | `False` | Disable z-score normalization |

#### VAE Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--latent-channels` | int | `32` | Number of latent channels in VAE |
| `--vae-base-channels` | int | `128` | Base number of channels in VAE encoder/decoder |
| `--vae-downsample-factor` | int | `16` | Downsampling factor (2^n; e.g., 16 = 4 steps) |
| `--vae-epochs` | int | `10` | Number of VAE training epochs |
| `--vae-kl-weight` | float | `1e-4` | Weight for KL divergence term |
| `--vae-recon-loss` | str | `l1` | Reconstruction loss: `l1` or `mse` |
| `--vae-lr` | float | `1e-3` | VAE learning rate |

#### Diffusion Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--diffusion-timesteps` | int | `1000` | Number of diffusion steps |
| `--diffusion-beta-schedule` | str | `cosine` | Beta schedule: `cosine` or `linear` |
| `--unet-base-channels` | int | `128` | Base number of channels in diffusion UNet |
| `--diffusion-epochs` | int | `20` | Number of diffusion training epochs |
| `--diffusion-lr` | float | `2e-4` | Diffusion learning rate |
| `--num-inference-steps` | int | `50` | Number of inference steps for generation |

#### Latent Scaling
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--latent-scale` | float | `0.0` | Latent scaling multiplier (0=auto-estimate, <0=disable/use 1.0, >0=fixed) |
| `--latent-scale-estimate-batches` | int | `50` | Batches used for auto-estimating latent std |

#### Training
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--train-samples-per-epoch` | int | `50000` | Max training windows per epoch (0 = no cap) |
| `--grad-clip` | float | `1.0` | Gradient clipping threshold |
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda`, or `cuda:N` |
| `--gpu` | int | `0` | GPU index when using `auto` or `cuda` |
| `--seed` | int | `42` | Random seed (>=0 fixed; <0 random) |

#### Class Selection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--class-name` | str | `""` | Artifact class to train (chew, elec, eyem, musc, shiv, elpp) |
| `--all-classes` | flag | `False` | Train all classes sequentially |
| `--exclusive` | flag | `False` | Only use windows where this class is the only positive label |

#### Output
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--run-name` | str | `""` | Custom run name (timestamp used if empty) |

---

## 3. train_wgangp.py - WGAN-GP Generator

Train per-class WGAN-GP generators for synthesizing artifact windows.

### Basic Usage

```bash
python training/train_wgangp.py \
  --split-root data/01_tcp_ar_split \
  --class-name chew \
  --epochs 50 \
  --batch-size 32
```

### Output

For each class, saves:
- `generator_last.pt` - Trained generator weights
- `settings.json` - Training configuration
- `history.json` - Loss history

### Parameters

#### Data Loading
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--split-root` | str | `data/01_tcp_ar_split` | Root directory containing train/dev/test splits |
| `--data-subdir` | str | `01_tcp_ar` | Subdirectory name within each split |
| `--batch-size` | int | `32` | Batch size for training |
| `--num-workers` | int | `4` | Number of workers for data loading |

#### Signal Processing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sfreq` | float | `250.0` | Sampling frequency (Hz) |
| `--window-sec` | float | `8.0` | Window length in seconds |
| `--stride-sec` | float | `0.5` | Stride between windows in seconds |
| `--min-overlap-sec` | float | `0.25` | Minimum overlap with artifact labels (seconds) |
| `--min-overlap-frac-artifact` | float | `0.25` | Minimum fraction of artifact window that must overlap |
| `--no-normalize` | flag | `False` | Disable z-score normalization |

#### Model Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--latent-dim` | int | `128` | Latent vector dimension for generator |
| `--lambda-gp` | float | `10.0` | Gradient penalty coefficient |
| `--output-activation` | str | `linear` | Generator output activation: `linear` or `tanh` |

#### Optimizer Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lr-g` | float | `2e-4` | Generator learning rate |
| `--lr-d` | float | `2e-4` | Critic (discriminator) learning rate |
| `--beta1` | float | `0.0` | Adam beta1 (momentum) |
| `--beta2` | float | `0.9` | Adam beta2 (running average) |

#### Training
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | `50` | Number of training epochs |
| `--n-critic` | int | `5` | Number of critic updates per generator update |
| `--train-samples-per-epoch` | int | `50000` | Max training windows per epoch (0 = no cap) |
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda`, or `cuda:N` |
| `--gpu` | int | `0` | GPU index when using `auto` or `cuda` |
| `--seed` | int | `42` | Random seed (>=0 fixed; <0 random) |

#### Class Selection
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--class-name` | str | `""` | Artifact class to train (chew, elec, eyem, musc, shiv, elpp) |
| `--all-classes` | flag | `False` | Train all classes sequentially |
| `--exclusive` | flag | `False` | Only use windows where this class is the only positive label |

#### Output
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--run-name` | str | `""` | Custom run name (timestamp used if empty) |

---

## Supported Models (for train.py)

| Model | Description |
|-------|-------------|
| `eegnet` | Standard ConvNet for EEG with depthwise separable convolutions |
| `eegnex` | EEGNeX: efficient ConvNet variant |
| `eegconformer` | Transformer-based with self-attention on the entire signal |
| `cnnlstm` | CNN backbone + bidirectional LSTM encoder |
| `eeginceptionerp` | Inception-style architecture optimized for ERPs |

---

## Artifact Classes

The following artifact classes are used across all scripts:
- `chew` - Jaw clenching/chewing
- `elec` - Electrode-related
- `eyem` - Eye movement
- `musc` - Muscle activity
- `shiv` - Shivering
- `elpp` - Electrode pop

---

## Notes

- **Seed Management**: Use `--seed 42` for reproducibility or `--seed -1` for a random seed
- **GPU/CPU**: Use `--device auto` to auto-detect, `--device cpu` for CPU-only, or `--device cuda:0` for specific GPU
- **LDM Augmentation**: LDM augmentation supports only `--num-workers 0` in the current version. When using `--aug-ldm-major`, always set `--num-workers 0` to avoid CUDA issues
- **Model Checkpoints**: Classification models are saved in `results/experiments/` with unique IDs based on model name and seed
- **GAN/LDM Models**: Generators saved in `results/saved_wgangp/<class>/` and `results/saved_ldm/<class>/`
- **Threshold Tuning**: By default, per-class thresholds are tuned on the dev set to maximize F1. Disable with `--no-tune-thresholds`
