# CardioLatentNet

> **Unsupervised ECG representation learning via deep autoencoders for heartbeat clustering and rhythm analysis**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-PTB--XL-green)](https://physionet.org/content/ptb-xl/1.0.3/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

---

## Overview

CardioLatentNet explores unsupervised representation learning for 12-lead ECG signals using three distinct deep autoencoder architectures. The core idea is simple but powerful: **compress raw ECG signals into compact latent vectors that preserve clinically meaningful structure**, then cluster those vectors to discover rhythm patterns without any labels.

The project runs two parallel experimental tracks — one operating on **individual heartbeats** (60-sample windows centered on R-peaks), and one operating on **full 10-second ECG recordings** (1000 samples at 100 Hz). Both tracks evaluate reconstruction fidelity and latent space quality independently, allowing direct comparison of how different inductive biases handle short versus long temporal signals.

---

## Motivation

Standard ECG analysis pipelines rely heavily on hand-crafted features or supervised classifiers trained on expensive expert annotations. This project takes a different angle: **can we learn a general-purpose ECG embedding purely from reconstruction signal, without any diagnostic labels?**

A good latent space should satisfy two properties simultaneously:

1. **Reconstruction quality** — the decoder can faithfully reproduce the original signal from the compressed representation (measured via MSE, MAE, RMSE, SNR)
2. **Cluster separability** — similar rhythms cluster together in latent space (measured via Silhouette Score and Davies-Bouldin Index on K-Means with K=18 diagnostic classes)

These two objectives are often in tension. A model that memorises every detail reconstructs well but may scatter latents arbitrarily. A model that compresses aggressively may cluster cleanly but lose clinically relevant waveform features. The experiments systematically explore this trade-off.

---

## Dataset

**PTB-XL** — a large publicly available 12-lead ECG dataset from Physikalisch-Technische Bundesanstalt.

| Split | Records |
|---|---|
| Train | ~17,000 recordings |
| Validation | ~2,200 recordings |
| Test | ~2,200 recordings |

Each recording is 10 seconds at 100 Hz → 1000 time steps × 12 leads. For the heartbeat track, R-peaks are detected using NeuroKit2 on Lead II, and 60-sample windows (200ms pre-peak, 400ms post-peak) are extracted, yielding approximately **200,000 individual heartbeats** for training.

Labels are multi-label SCP diagnostic codes collapsed to 5 diagnostic superclasses. Labels are used **only for evaluation** (latent space visualisation and cluster quality scoring) — never for training.

---

## Architecture

Three autoencoder families are evaluated, each with multiple capacity and regularisation variants:

### CNN Autoencoder (`CnnAec`)

A purely convolutional autoencoder with a learnable bottleneck. Encoder uses strided `Conv1d` blocks with BatchNorm and GELU activations; decoder mirrors with `ConvTranspose1d`. The bottleneck uses **Attention Pooling** — a learned weighted sum across the temporal dimension — to produce a single fixed-size latent vector regardless of sequence length.

```
Input (B, T, 12)
  → [Conv1d stride=2 + BN + GELU] × blocks          # encoder, halves T each block
  → Attention Pooling → Linear                        # (B, T', C) → (B, latent_dim)
  → Linear → reshape                                  # latent_dim → (B, C, T')
  → [ConvTranspose1d stride=2 + BN + GELU] × blocks  # decoder, doubles T each block
  → Output (B, T, 12)
```

Key design choices: `enc_dec_ratio` controls how many processing layers run between each downsampling step, allowing depth without aggressive spatial compression. Residual connections within stride-1 blocks improve gradient flow in deeper variants.

### LSTM-CNN VAE (`LstmVae`)

A hybrid recurrent-convolutional variational autoencoder. Each encoder block applies multi-scale convolutions (kernel sizes 3, 5, 7 in parallel) merged via a 1×1 conv, followed by a bidirectional LSTM with a learnable residual gate (`alpha` parameter). The variational bottleneck samples `z ~ N(mu, sigma)` and is regularised with **Maximum Mean Discrepancy (MMD)** instead of KL divergence, avoiding posterior collapse.

```
Input (B, T, 12)
  → [MultiScale Conv + BiLSTM + α-residual] × blocks  # encoder
  → Flatten → Linear → (mu, logvar)                    # variational bottleneck
  → Sample z, project back                             # reparametrisation
  → [Interpolate + BiLSTM + ConvTranspose] × blocks    # decoder
  → Output (B, T, 12)
```

MMD regularisation weight `mmd_weight` is the primary hyperparameter controlling the reconstruction vs. latent structure trade-off. Empirically, low values (0.1–0.3) yield better cluster separability than high values by allowing the model to organise the latent space semantically rather than forcing it into a unit Gaussian.

### Transformer U-AEC (`TransformerAec`)

A U-Net-style transformer autoencoder with skip connections. The encoder applies multi-head self-attention blocks interleaved with learned 1D downsampling layers. Skip connections from each encoder level are passed to the corresponding decoder level via cross-attention. The bottleneck uses Attention Pooling to compress the spatial dimension before projecting to `latent_dim`, then expands back via `expand` before the decoder.

```
Input (B, T, 12) → Linear embedding + Positional Encoding
  → [Self-Attention × enc_ratio + Downsampler] × blocks   # encoder + skip connections
  → AttentionPool → latent_proj → latent (B, latent_dim)  # bottleneck
  → latent_unproj → expand → ConnectorDecoder             # re-expansion
  → [Upsampler + Cross-Attention(enc_skip) × dec_ratio] × blocks  # decoder
  → Linear projection → Output (B, T, 12)
```

`enc_dec_ratio=(e, d)` controls attention depth per scale level independently for encoder and decoder. Flash Attention is used where available. Gradient checkpointing is supported for memory-constrained environments.

---

## Regularisation Strategy

| Model | Regulariser | Notes |
|---|---|---|
| CNN AEC | None | Pure reconstruction objective |
| LSTM VAE | MMD (RBF kernel) | `mmd_weight` ∈ [0.1, 1.0] across experiments |
| Transformer AEC | None | Relies on depth and dropout |

MMD with a Radial Basis Function kernel measures the distance between the empirical latent distribution and a standard Gaussian prior. Unlike KL divergence, it does not suffer from posterior collapse and produces smoother, more continuous latent spaces suitable for clustering.

---

## Training Infrastructure

All trainers share a common design pattern:

- **Step-based training** with cosine LR decay and linear warmup
- **Gradient accumulation** with configurable `accumulation_step` — all logging, evaluation, and checkpointing intervals are automatically aligned to accumulation boundaries via `__post_init__`
- **Early stopping** on validation reconstruction loss with configurable patience, saving `{model}_best.pt` on every improvement
- **Three checkpoint files** per run: `{model}_step_{N}.pt` (periodic), `{model}_newest.pt` (latest), `{model}_best.pt` (best val loss), `{model}_model.pt` (weights only)
- **AMP support** for both `bf16` and `fp16` with automatic GradScaler for fp16
- **MPS support** for Apple Silicon training

Training configuration is fully serialisable to JSON, enabling reproducible experiment management without code changes.

---

## Experiment Tracks

### Heartbeat Track (`seq_len=60`)

Variants sweep over architecture depth (`blocks` 3–6), latent dimensionality (16–192), regularisation strength (`mmd_weight` 0.1–1.0), and processing depth per scale level (`enc_dec_ratio` 1–3). Full experiment list in `experiment_heartbeat.json` and `experiment_heartbeat2.json`.

### Full ECG Track (`seq_len=1000`)

Longer sequences introduce the quadratic attention bottleneck for transformers. Experiments use aggressive downsampling ratios (`enc_dec_ratio=[2,1]` to `[3,2]`), gradient checkpointing, and gradient accumulation to maintain effective batch sizes. Full experiment list in `experiment_full.json`.

---

## Evaluation Metrics

### Reconstruction

| Metric | Description |
|---|---|
| MSE | Mean squared error across all timesteps and leads |
| MAE | Mean absolute error |
| RMSE | Root mean squared error |
| SNR (dB) | Signal-to-noise ratio — primary reconstruction metric for reporting |

### Latent Space Quality

| Metric | Description | Optimum |
|---|---|---|
| Silhouette Score | Cluster cohesion and separation (K=18) | ↑ higher |
| Davies-Bouldin Index | Average cluster similarity | ↓ lower |
| Active Dimensions | Number of latent dims with std > 0.1 | model-dependent |

Latent spaces are visualised with PCA, t-SNE (perplexity=50, 10k sample), and UMAP (n\_neighbors=30) coloured by diagnostic superclass.

### Composite Ranking

For cross-model comparison, a composite score balances both objectives:

```
score = 0.4 × SNR_dB + 0.6 × Silhouette × 100
```

Weights reflect the primary project goal of clustering quality over reconstruction fidelity.

---

## Results

> *Section to be completed after full experimental runs.*

### Heartbeat Track — Summary

| Model | MSE | SNR (dB) | Silhouette | Davies-Bouldin | Active Dims |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

### Full ECG Track — Summary

| Model | MSE | SNR (dB) | Silhouette | Davies-Bouldin | Active Dims |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

### Key Findings

> *To be filled in after analysis.*

---

## Project Structure

```text
CardioLatentNet/
│
├── src/
│   │
│   ├── cnn_aec.py
│   │   └── CNN Autoencoder architecture
│   │
│   ├── lstmcnn_aec.py
│   │   └── LSTM-CNN VAE architecture
│   │
│   ├── transformer_uaec.py
│   │   └── Transformer U-AEC architecture
│   │
│   ├── layers/
│   │   │
│   │   ├── mlp.py
│   │   │   └── Multi-Layer Perceptron helper layers
│   │   │
│   │   ├── attention_pooling.py
│   │   │   └── Attention Pooling mechanism
│   │   │
│   │   ├── blocks/
│   │   │   ├── conv_encoder_block.py
│   │   │   │   └── Convolutional encoder blocks
│   │   │   │
│   │   │   ├── conv_decoder_block.py
│   │   │   │   └── Convolutional decoder blocks
│   │   │   │
│   │   │   ├── lstmconv_encoder_block.py
│   │   │   │   └── Coupled LSTM-Conv encoder layers
│   │   │   │
│   │   │   ├── lstmconv_decoder_block.py
│   │   │   │   └── Coupled LSTM-Conv decoder layers
│   │   │   │
│   │   │   ├── lstmconv_proccess_block.py
│   │   │   │   └── Latent-space processing for LSTM-Conv models
│   │   │   │
│   │   │   ├── transformer_encoder_block.py
│   │   │   │   └── Transformer-based encoding blocks
│   │   │   │
│   │   │   ├── transformer_decoder_block.py
│   │   │   │   └── Transformer-based decoding blocks
│   │   │   │
│   │   │   └── variational_block.py
│   │   │       └── KL-Divergence and latent sampling for VAEs
│   │   │
│   │   ├── encoding/
│   │   │   └── positional_encoding.py
│   │   │       └── Positional Encoding for sequence data
│   │   │
│   │   └── dimension/
│   │       ├── downsampler.py
│   │       │   └── Dimensionality reduction utility
│   │       └── upsampler.py
│   │           └── Dimensionality expansion utility
│   │
│   ├── utils/
│   │   │
│   │   ├── trainers/
│   │   │   └── Core training engines:
│   │   │       • CnnTrainer
│   │   │       • LstmTrainer
│   │   │       • TransformerTrainer
│   │   │
│   │   └── config/
│   │       │
│   │       ├── model/
│   │       │   └── Architecture specifications:
│   │       │       • cnn_model_config
│   │       │       • lstm_model_config
│   │       │       • transformer_model_config
│   │       │
│   │       └── trainer/
│   │           └── Training hyperparameter specifications:
│   │               • cnn_trainer_config
│   │               • lstm_trainer_config
│   │               • transformer_trainer_config
│   │
│   ├── data/
│   │   ├── heartbeat_ecg_ds.py
│   │   │   └── Heartbeat_ECG_DataSet (R-peak segmentation)
│   │   │
│   │   └── full_ecg_ds.py
│   │       └── Full_ECG_DataSet (complete ECG recordings)
│   │
│   └── visualize/
│       ├── plot_history.py
│       │   └── Training metrics and loss curves
│       │
│       ├── plot_heartbeats.py
│       │   └── Original vs reconstructed heartbeat signals
│       │
│       ├── plot_clusters.py
│       │   └── Latent-space visualizations and clustering
│       │
│       └── plot_channels.py
│           └── Multi-channel ECG signal visualization
│
├── experiments/
│   ├── experiment_heartbeat.json
│   │   └── Baseline heartbeat experiment configuration
│   │
│   ├── experiment_heartbeat2.json
│   │   └── Extended heartbeat experiment configuration
│   │
│   └── experiment_full.json
│       └── Full ECG experiment configuration
│
├── training.py
│   └── CLI entry point for model training
│
├── process.py
│   └── Data processing and inference pipeline orchestration
│
├── report.ipynb
│   └── Jupyter notebook for prototyping and result presentation
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Project documentation
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/yourname/CardioLatentNet
cd CardioLatentNet
pip install -r requirements.txt
```

### Download Dataset

```python
import kagglehub
path = kagglehub.dataset_download("khyeh0719/ptb-xl-dataset")
```

### Run Training

```python
from src.utils.config_loader import load_configs

configs  = load_configs("experiments/experiment_heartbeat.json",  "./checkpoints/heartbeat")
configs += load_configs("experiments/experiment_heartbeat2.json", "./checkpoints/heartbeat")
configs += load_configs("experiments/experiment_full.json",       "./checkpoints/full")

for cfg in configs:
    run_training(
        train_ds, val_ds, test_ds,
        model_cls=cfg['model_cls'],
        trainer_cls=cfg['trainer_cls'],
        model_cfg=cfg['model_cfg'],
        trainer_cfg=cfg['trainer_cfg'],
        batch_sizes=cfg['batch_sizes'],
    )
```

### Run Evaluation

```python
python proccess.py
```

Results are saved per model to `results/{heartbeat|full}/{model_name}/`:
- `{model_name}_latent_pca.png`
- `{model_name}_latent_tsne.png`
- `{model_name}_latent_umap.png`
- `{model_name}_history.png`
- `{model_name}_results.txt`

---

## Requirements

```
torch>=2.0
torchaudio
numpy
pandas
scikit-learn
neurokit2
wfdb
umap-learn
matplotlib
seaborn
kagglehub
```

---

## License

MIT License. Dataset (PTB-XL) is subject to its own [PhysioNet license](https://physionet.org/content/ptb-xl/1.0.3/).