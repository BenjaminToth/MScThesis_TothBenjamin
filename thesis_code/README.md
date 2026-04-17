## Workflow

### 1. Preprocessing
Start by preprocessing the raw EEG data. This step involves loading EDF files, cleaning the signals, and preparing them for further analysis. The preprocessing scripts handle tasks like filtering, normalization, and annotation processing. See the `data_preprocessing/` directory for available tools.

### 2. Dataset Splitting
After preprocessing, organize the data into train, development, and test splits. This ensures proper evaluation and prevents data leakage. The splitting can be done patient-wise to maintain independence between splits. Refer to the `dataset_split/` directory for splitting scripts and data analysis tools.

### 3. Training & Experiments
Train classification models to detect artifacts. You have three main options:

**Option A: Train Classification Models**
- Train neural networks (EEGNet, EEGConformer, CNNLSTM, etc.) to classify artifact types
- Apply various data augmentations (noise, channel dropout, mixup, etc.)
- Tune hyperparameters and evaluate on validation/test sets
- See `training/train.py` for details

**Option B: Generate Synthetic Data with WGAN-GP**
- Train per-class WGAN-GP generators to synthesize realistic artifact windows
- Use the trained generators to augment classifier training data
- See `training/train_wgangp.py` for details

**Option C: Generate Synthetic Data with Latent Diffusion Models**
- Train per-class VAE + DDPM models to learn the latent distribution of artifacts
- Use the diffusion models to generate high-quality synthetic samples
- Use these synthetics to improve classifier performance
- See `training/train_ldm.py` for details

You can combine approaches—for example, train synthetic generators first, then use them to augment classifier training.

All training scripts save results, configurations, and performance metrics for analysis. See `training/README.md` for complete documentation of parameters and workflows.


