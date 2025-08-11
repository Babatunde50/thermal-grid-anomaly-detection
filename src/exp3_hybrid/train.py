from __future__ import annotations
import os, csv, glob, json, argparse, random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

# Optional YAML
try:
    import yaml
except Exception:
    yaml = None


# =============================
# Utilities
# =============================

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def load_csv_sequence(file_path: Path, T: int, F: int) -> np.ndarray:
    """
    Load one CSV to (T,F). Robust to optional first timestamp/index col, headers.
    Truncate/pad to T. Normalize as (x - 10)/12.5.
    """
    rows = []
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    frames = []
    for r in rows:
        if not r:
            continue
        if len(r) >= F and all(is_float(v) for v in r[:F]):
            vals = [float(v) for v in r[:F]]
        elif len(r) >= F + 1 and all(is_float(v) for v in r[1:1+F]):
            vals = [float(v) for v in r[1:1+F]]
        else:
            continue
        frames.append(vals)

    if len(frames) == 0:
        arr = np.zeros((T, F), dtype=np.float32)
        return (arr - 10.0) / 12.5

    frames = frames[:T]
    if len(frames) < T:
        last = frames[-1]
        frames = frames + [last] * (T - len(frames))
    arr = np.asarray(frames, dtype=np.float32)
    arr = (arr - 10.0) / 12.5
    return arr

def load_folder_to_array(folder: Path, T: int, F: int, C: int) -> np.ndarray:
    paths = sorted(glob.glob(str(folder / "*.csv")))
    if len(paths) == 0:
        return np.zeros((0, T, F, C), dtype=np.float32)
    X = np.zeros((len(paths), T, F, C), dtype=np.float32)
    for i, p in enumerate(paths):
        seq = load_csv_sequence(Path(p), T=T, F=F)
        X[i, :, :, 0] = seq
    return X

def build_autoencoder(input_shape, enc_filters=(32, 16), dec_filters=(16, 32), use_batchnorm=False):
    def conv_block(x, f):
        x = layers.Conv2D(f, 3, padding="same")(x)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    inp = layers.Input(shape=input_shape)
    x = conv_block(inp, enc_filters[0])
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = conv_block(x, enc_filters[1])
    bottleneck = layers.MaxPooling2D(2, padding="same", name="bottleneck")(x)

    x = layers.Conv2DTranspose(dec_filters[0], 3, strides=2, activation="relu", padding="same")(bottleneck)
    x = layers.Conv2DTranspose(dec_filters[1], 3, strides=2, activation="relu", padding="same")(x)
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    ae = models.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    enc = models.Model(inp, bottleneck)
    return ae, enc

def build_fnn(input_dim: int, hidden_units=64, dropout=0.3) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_units, activation="relu")(inp)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


# =============================
# Core pipeline
# =============================

def run(cfg: dict) -> None:
    set_seeds(int(cfg.get("seed", 42)))

    # ---- Config
    data_cfg = cfg.get("data", {})
    dataset_root = Path(data_cfg.get("dataset_root", "./"))
    normal_folder = data_cfg.get("normal_folder", "Sensor1_BigArea_Single")
    anomaly_folder = data_cfg.get("anomaly_folder", "Sensor1_BigArea_Double")
    T = int(data_cfg.get("T", 40))
    F = int(data_cfg.get("F", 64))
    C = int(data_cfg.get("C", 1))
    normal_train_ratio = float(data_cfg.get("normal_train_ratio", 0.8))
    split_seed = int(data_cfg.get("split_seed", 42))

    ae_cfg = cfg.get("ae", {})
    ae_epochs = int(ae_cfg.get("epochs", 50))
    ae_bs = int(ae_cfg.get("batch_size", 16))
    enc_filters = tuple(ae_cfg.get("encoder_filters", [32, 16]))
    dec_filters = tuple(ae_cfg.get("decoder_filters", [16, 32]))
    use_bn = bool(ae_cfg.get("use_batchnorm", False))

    km_cfg = cfg.get("kmeans", {})
    n_clusters = int(km_cfg.get("n_clusters", 5))
    km_seed = int(km_cfg.get("random_state", 0))

    fnn_cfg = cfg.get("fnn", {})
    fnn_epochs = int(fnn_cfg.get("epochs", 30))
    fnn_bs = int(fnn_cfg.get("batch_size", 16))
    fnn_hidden = int(fnn_cfg.get("hidden_units", 64))
    fnn_dropout = float(fnn_cfg.get("dropout", 0.3))
    val_split = float(fnn_cfg.get("val_split", 0.2))

    exp_cfg = cfg.get("experiment", {})
    anomaly_fracs: List[float] = exp_cfg.get("anomaly_fractions", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    runs_cfg: Dict = exp_cfg.get("runs_per_fraction", {"default": 10, "overrides": {}})
    runs_default = int(runs_cfg.get("default", 10))
    runs_over = {float(k): int(v) for k, v in runs_cfg.get("overrides", {}).items()}

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "output/exp3"))
    ensure_dir(out_dir)

    print("Loading data... (dataset path NOT included in repo; set in config)")
    norm_path = dataset_root / normal_folder
    anom_path = dataset_root / anomaly_folder

    X_norm = load_folder_to_array(norm_path, T=T, F=F, C=C)
    X_anom = load_folder_to_array(anom_path, T=T, F=F, C=C)

    if X_norm.shape[0] == 0:
        raise RuntimeError(f"No CSVs found in normals folder: {norm_path}")
    if X_anom.shape[0] == 0:
        raise RuntimeError(f"No CSVs found in anomalies folder: {anom_path}")

    # ---- AE train/val on normals
    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(X_norm))
    rng.shuffle(idx)
    n_train = max(1, int(normal_train_ratio * len(idx)))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:] if n_train < len(idx) else idx[:1]

    x_ae_train = X_norm[train_idx]
    x_ae_val   = X_norm[val_idx]

    # ---- Build & train AE
    ae, encoder = build_autoencoder((T, F, C), enc_filters=enc_filters, dec_filters=dec_filters, use_batchnorm=use_bn)
    ae.fit(x_ae_train, x_ae_train,
           epochs=ae_epochs,
           batch_size=ae_bs,
           shuffle=True,
           validation_data=(x_ae_val, x_ae_val),
           verbose=0)

    # ---- K-Means on raw normals
    raw_norm_flat = X_norm.reshape(len(X_norm), -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=km_seed).fit(raw_norm_flat)
    centroids = kmeans.cluster_centers_

    # ---- Build hybrid features (latent + centroid distances)
    all_data = np.concatenate([X_norm, X_anom], axis=0)
    y_all = np.concatenate([np.zeros(len(X_norm), dtype=int),
                            np.ones(len(X_anom), dtype=int)], axis=0)

    latent_all = encoder.predict(all_data, verbose=0).reshape(len(all_data), -1)
    raw_all = all_data.reshape(len(all_data), -1)
    dists_all = np.linalg.norm(raw_all[:, None, :] - centroids[None, :, :], axis=2)
    X_fnn = np.concatenate([latent_all, dists_all], axis=1)

    norm_idx = np.where(y_all == 0)[0]
    anom_idx = np.where(y_all == 1)[0]

    # ---- Experiment loop (multi-run per fraction)
    results_rows = []
    acc_means, acc_stds, f1_means, f1_stds = [], [], [], []
    xs_pct = []

    for frac in anomaly_fracs:
        runs = runs_over.get(float(frac), runs_default)
        print(f"\n===== {int(frac*100)}% anomalies in TRAIN ({runs} runs) =====")

        f1_list, acc_list = [], []

        for run in range(runs):
            # per-run seeds for reproducibility variance
            set_seeds(int(cfg.get("seed", 42)) + run)

            # how many anomalies in train
            n_anom_train = int(frac * len(anom_idx))
            if frac > 0 and n_anom_train == 0:
                n_anom_train = 1
            n_anom_train = min(n_anom_train, len(anom_idx))

            # sample anomalies for train
            train_anom = np.random.choice(anom_idx, size=n_anom_train, replace=False)
            test_anom = np.setdiff1d(anom_idx, train_anom)

            # match normals for train (cap if too few normals)
            n_norm_train = min(n_anom_train, len(norm_idx))
            train_norm = np.random.choice(norm_idx, size=n_norm_train, replace=False)
            test_norm = np.setdiff1d(norm_idx, train_norm)

            train_ids = np.concatenate([train_anom, train_norm])
            test_ids = np.concatenate([test_anom, test_norm])

            X_train, y_train = X_fnn[train_ids], y_all[train_ids]
            X_test,  y_test  = X_fnn[test_ids],  y_all[test_ids]

            # shuffle train
            perm = np.random.permutation(len(y_train))
            X_train, y_train = X_train[perm], y_train[perm]

            # ensure both classes present; if not, skip this run
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                # fallback: skip run to avoid degenerate training/eval
                continue

            # class weights
            cw_vals = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_train)
            cw = {0: cw_vals[0], 1: cw_vals[1]}

            # FNN
            fnn = build_fnn(X_train.shape[1], hidden_units=fnn_hidden, dropout=fnn_dropout)
            fnn.fit(X_train, y_train,
                    epochs=fnn_epochs,
                    batch_size=fnn_bs,
                    validation_split=val_split,
                    class_weight=cw,
                    verbose=0)

            prob_test = fnn.predict(X_test, verbose=0).ravel()
            best_thr = youden_threshold(y_test, prob_test)
            y_pred = (prob_test > best_thr).astype(int)

            acc_list.append(accuracy_score(y_test, y_pred))
            f1_list.append(f1_score(y_test, y_pred, zero_division=0))

        # summarize this fraction
        if len(f1_list) == 0:
            acc_mean, acc_std = float("nan"), float("nan")
            f1_mean, f1_std = float("nan"), float("nan")
        else:
            acc_mean, acc_std = float(np.mean(acc_list)), float(np.std(acc_list))
            f1_mean,  f1_std  = float(np.mean(f1_list)),  float(np.std(f1_list))

        print(f"{int(frac*100)}% → F1: {f1_mean:.4f} ± {f1_std:.4f} | Acc: {acc_mean:.4f} ± {acc_std:.4f}")

        results_rows.append({
            "anomaly_fraction": frac,
            "runs": runs,
            "acc_mean": acc_mean, "acc_std": acc_std,
            "f1_mean": f1_mean,   "f1_std": f1_std
        })
        xs_pct.append(int(frac * 100))
        acc_means.append(acc_mean); acc_stds.append(acc_std)
        f1_means.append(f1_mean);   f1_stds.append(f1_std)

    # ---- Save CSV + plots + config snapshot
    import pandas as pd
    ensure_dir(out_dir)
    pd.DataFrame(results_rows).to_csv(out_dir / "summary_exp3_hybrid.csv", index=False)

    plt.figure(figsize=(6,4))
    plt.errorbar(xs_pct, f1_means, yerr=f1_stds, marker="o", capsize=5)
    plt.title("Hybrid (AE + K-Means + FNN) — F1 vs % Anomalies")
    plt.xlabel("% Anomalies in Training")
    plt.ylabel("Test F1-Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_vs_frac_meanstd.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.errorbar(xs_pct, acc_means, yerr=acc_stds, marker="o", capsize=5)
    plt.title("Hybrid (AE + K-Means + FNN) — Accuracy vs % Anomalies")
    plt.xlabel("% Anomalies in Training")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_frac_meanstd.png")
    plt.close()

    with open(out_dir / "used_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nDone.")
    print(f"Saved CSV → {out_dir/'summary_exp3_hybrid.csv'}")
    print(f"Saved plots → {out_dir/'f1_vs_frac_meanstd.png'}, {out_dir/'accuracy_vs_frac_meanstd.png'}")


# =============================
# Entry
# =============================

def load_config(path: str | None) -> dict:
    defaults = {
        "seed": 42,
        "data": {
            "dataset_root": "./",  # change me
            "normal_folder": "Sensor1_BigArea_Single",
            "anomaly_folder": "Sensor1_BigArea_Double",
            "T": 40, "F": 64, "C": 1,
            "normal_train_ratio": 0.8,
            "split_seed": 42
        },
        "ae": {
            "epochs": 50, "batch_size": 16,
            "encoder_filters": [32, 16],
            "decoder_filters": [16, 32],
            "use_batchnorm": False
        },
        "kmeans": {"n_clusters": 5, "random_state": 0},
        "fnn": {"epochs": 30, "batch_size": 16, "hidden_units": 64, "dropout": 0.3, "val_split": 0.2},
        "experiment": {
            "anomaly_fractions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "runs_per_fraction": {"default": 10, "overrides": {"0.5": 50}}
        },
        "output": {"dir": "output/exp3"}
    }
    if path is None:
        return defaults
    if yaml is None:
        print("[warn] PyYAML not installed; using defaults.")
        return defaults
    with open(path, "r") as f:
        user = yaml.safe_load(f) or {}

    def deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    return deep_merge(defaults, user)

def main(config_path: str | None) -> None:
    cfg = load_config(config_path)
    run(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (optional).")
    args = parser.parse_args()
    main(args.config)