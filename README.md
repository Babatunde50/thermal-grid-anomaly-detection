# Infrared Grid Anomaly Detection

End-to-end experiments for anomalous human activity detection on **low-resolution (8×8) thermal infrared grids** using the Coventry‑2018 dataset.  
We provide:
- **Exp 1 – Supervised CNN baseline** with scarce anomaly labels
- **Exp 2 – Unsupervised Autoencoder** with reconstruction‑error thresholding
- **Exp 3 – Semi‑Supervised Hybrid (AE + K‑Means + FNN)**

> ⚠️ **Data not included.** You must supply the licensed Coventry‑2018 CSV files locally (see below).

---

## Repository Layout

```
.
├── data/
│   └── raw/
│       └── Coventry2018/
│           ├── Sensor1_BigArea_Single/   # single-person sequences (CSV)
│           └── Sensor1_BigArea_Double/   # two-person sequences (CSV)   [only for Exp 2/3 variants that use it]
├── outputs/
│   ├── exp1_cnn_supervised/
│   ├── exp2_autoencoder_unsup/
│   └── exp3_hybrid_semi_supervised/
├── requirements.txt
├── README.md
└── src/
    ├── utils/               # shared helpers (I/O, plotting, metrics)
    ├── exp1_cnn_supervised/
    │   ├── train.py
    │   └── config.yaml
    ├── exp2_autoencoder_unsup/
    │   ├── train.py
    │   └── config.yaml
    └── exp3_hybrid_semi_supervised/
        ├── train.py
        └── config.yaml
```

---

## Data (Not Included)

This project uses the **Infrared Human Activity Recognition Dataset (Coventry‑2018)** available via **IEEE DataPort (subscription)**.  
Place CSVs under the following structure (do **not** commit data):

```
data/raw/Coventry2018/Sensor1_BigArea_Single/*.csv
data/raw/Coventry2018/Sensor1_BigArea_Double/*.csv    # only needed for runs that use the double-person set
```

### Activity Labels in Exp 1
Inside `Sensor1_BigArea_Single`, files containing **“SwingLR”** or **“SwingFB”** in their **filename** are treated as **anomalies**. All other activities are treated as **normal**.  
If your file naming differs, update `anomaly_patterns` in the config (see below).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

We recommend Python **3.9–3.11** and TensorFlow **2.13+** (Apple Silicon works fine on CPU).

---

## Run Experiments

### Exp 1 — Supervised CNN Baseline
Trains a small CNN on activity-labeled sequences from `Sensor1_BigArea_Single`, varying the fraction of anomaly labels (10–70%). Threshold is chosen via ROC‑Youden or F1‑grid on an internal validation split.

```bash
python -m src.exp1_cnn_supervised.train --config src/exp1_cnn_supervised/config.yaml
```

**Outputs** → `outputs/exp1_cnn_supervised/<timestamp>/`
- `summary_exp1_cnn.csv` — mean±std per anomaly fraction (Accuracy, F1)
- `accuracy_vs_anomaly_frac_meanstd.png`
- `f1_vs_anomaly_frac_meanstd.png`
- `used_config.json`

**Key config fields** (YAML):
```yaml
seed: 42
data:
  dataset_root: "data/raw/Coventry2018"
  single_folder: "Sensor1_BigArea_Single"
  anomaly_patterns: ["SwingLR", "SwingFB"]
  T: 40
  F: 64
  C: 1
  normal_test_ratio: 0.2
  anomaly_test_ratio: 0.3
train:
  epochs: 100
  batch_size: 32
  l2: 0.0005
  dropout_conv: 0.3
  dropout_dense: 0.5
  early_stopping: {enabled: true, monitor: val_loss, patience: 10}
  reduce_lr: {enabled: true, patience: 5, factor: 0.5}
cv:
  folds: 5
  threshold_method: youden  # or: f1grid
experiment:
  anomaly_fractions: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  runs_per_fraction:
    default: 10
    for_fraction: {"0.4": 20}
output:
  dir: "outputs/exp1_cnn_supervised"
```

---

### Exp 2 — Unsupervised Autoencoder (Reconstruction Thresholding)
Trains a conv‑autoencoder on **normal** sequences only (from `Sensor1_BigArea_Single`). The anomaly threshold is set from held‑out normal validation errors (`median + 3σ`).

```bash
python -m src.exp2_autoencoder_unsup.train --config src/exp2_autoencoder_unsup/config.yaml
```

**Outputs** → `outputs/exp2_autoencoder_unsup/<timestamp>/`
- `heldout_error_hist.png` — MSE on held‑out normals
- `confusion_matrix.png` — evaluation on a normal+anomaly test mix
- `classification_metrics.png` — Accuracy / Precision / Recall / F1
- `rmse_hist.png` — RMSE distributions for normal vs anomaly
- `statistics.csv` — numeric metrics
- `used_config.json`

**Key config fields** (YAML):
```yaml
seed: 42
data:
  dataset_root: "data/raw/Coventry2018"
  normal_folder: "Sensor1_BigArea_Single"
  anomaly_folder: "Sensor1_BigArea_Double"  # optional depending on your variant
  T: 40
  F: 64
  C: 1
train:
  epochs: 50
  batch_size: 10
model:
  conv_filters: [32, 32]
  bottleneck_filters: 32
output:
  dir: "outputs/exp2_autoencoder_unsup"
```

---

### Exp 3 — Semi‑Supervised Hybrid (AE + K‑Means + FNN)
1) Train AE on normals (like Exp 2).  
2) Fit K‑Means on **raw, flattened normals** to get centroids.  
3) Build hybrid features = **[AE latent | distances to K‑Means centroids]**.  
4) Train a small FNN with **limited anomaly labels** (10–70%) and evaluate.

```bash
python -m src.exp3_hybrid_semi_supervised.train --config src/exp3_hybrid_semi_supervised/config.yaml
```

**Outputs** → `outputs/exp3_hybrid_semi_supervised/<timestamp>/`
- `f1_vs_frac_meanstd.png`
- `accuracy_vs_frac_meanstd.png`
- `summary_exp3_hybrid.csv`
- `used_config.json`

**Key config fields** (YAML):
```yaml
seed: 42
data:
  dataset_root: "data/raw/Coventry2018"
  normal_folder: "Sensor1_BigArea_Single"
  anomaly_folder: "Sensor1_BigArea_Double"
  T: 40
  F: 64
  C: 1
ae:
  epochs: 50
  batch_size: 16
  latent_filters: 16
kmeans:
  n_clusters: 5
fnn:
  epochs: 30
  batch_size: 16
experiment:
  anomaly_fractions: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  runs:
    default: 10
    override: {0.5: 50}
output:
  dir: "outputs/exp3_hybrid_semi_supervised"
```

---

## Reproducibility

- Every run writes `used_config.json` in its output folder.
- Set `seed` in the config to fix RNG for NumPy / TF.
- We avoid data leakage by holding out a fixed test split and sampling anomalies for training only from the **remaining** pool (Exp 1 & 3).

---

## Troubleshooting

- **“No anomalies found by filename patterns” (Exp 1)**  
  Check `anomaly_patterns` and verify filenames contain those tokens (e.g., `SwingLR`, `SwingFB`).

- **TensorFlow retracing warnings**  
  Harmless here. For speed, keep model creation outside tight loops if you heavily customize.

- **Empty plots / all-zeros metrics**  
  Verify your CSVs are parsed correctly (some files have a leading timestamp column). Our loader tries both `[64]` and `[1+64]` shapes and pads/truncates to T=40.

---

## License

- **Code**: MIT (or your preferred OSI license).  
- **Data**: *Not redistributed*. Obtain Coventry‑2018 via IEEE DataPort and follow the provider’s license.

---

## Citation

If you use this code, please cite the accompanying manuscript:

> Babatunde, O., Hasan, R., *Human Anomalous Activity Detection System Based on Infrared Grid Sensor*, 2025. (Under review)

