# src/exp1_cnn_supervised/train.py

from __future__ import annotations
import os
import csv
import glob
import argparse
import json
import time
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# Sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import class_weight

# Optional YAML (config)
try:
    import yaml
except Exception:
    yaml = None


# ==============================
# Utility functions
# ==============================

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def load_csv_sequence(file_path: Path, T: int, F: int) -> np.ndarray:
    """
    Load a single CSV sequence into shape (T, F).
    Handles CSVs where the first column might be a timestamp/index.
    Applies truncation/padding to T frames.
    Normalization: (x - 10) / 12.5
    """
    rows = []
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    frames = []
    for row in rows:
        # Skip empty rows
        if not row:
            continue

        # Try to parse with 64 features directly
        if len(row) >= F and all(is_float(v) for v in row[:F]):
            vals = [float(v) for v in row[:F]]
        # Else try skipping first column (timestamp/index)
        elif len(row) >= (F + 1) and all(is_float(v) for v in row[1:1 + F]):
            vals = [float(v) for v in row[1:1 + F]]
        else:
            # If even that fails (e.g., header), skip this row
            continue

        frames.append(vals)

    if len(frames) == 0:
        # If file is malformed, return zeros
        arr = np.zeros((T, F), dtype=np.float32)
        return (arr - 10.0) / 12.5

    # Truncate or pad with last frame to length T
    frames = frames[:T]
    if len(frames) < T:
        last = frames[-1]
        frames = frames + [last] * (T - len(frames))

    arr = np.asarray(frames, dtype=np.float32)  # (T, F)
    arr = (arr - 10.0) / 12.5
    return arr


def load_and_label_from_single_folder(
    folder: Path,
    T: int,
    F: int,
    C: int,
    anomaly_patterns: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads all CSV files in `folder`. Labels as anomaly=1 if filename contains
    any of the given patterns (e.g., "SwingLR","SwingFB"), else normal=0.
    Returns X in (N,T,F,C), y in (N,), and list of file basenames.
    """
    paths = sorted(glob.glob(str(folder / "*.csv")))
    if len(paths) == 0:
        raise RuntimeError(
            f"No CSV files found in: {folder}. Check data.dataset_root and data.single_folder."
        )
    N = len(paths)
    X = np.zeros((N, T, F, C), dtype=np.float32)
    y = np.zeros((N,), dtype=np.int32)
    names = []

    for i, p in enumerate(paths):
        pth = Path(p)
        names.append(pth.name)
        # Assign label by filename pattern (case-insensitive)
        fn_lower = pth.name.lower()
        pat_lower = [k.lower() for k in anomaly_patterns]
        y[i] = 1 if any(k in fn_lower for k in pat_lower) else 0

        seq = load_csv_sequence(pth, T=T, F=F)  # (T,F)
        X[i, :, :, 0] = seq

    return X, y, names


def build_cnn(
    input_shape: Tuple[int, int, int],
    l2: float = 5e-4,
    dr_conv: float = 0.30,
    dr_dense: float = 0.50
) -> tf.keras.Model:
    """
    2-block CNN with BN+ReLU, MaxPool, Dropouts and Dense(64) head.
    """
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None

    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dr_conv)(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dr_conv)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dr_dense)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold by maximizing Youden's J: max(TPR - FPR)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    k = np.argmax(j)
    return float(thr[k])


def choose_threshold_from_validation(
    y_val: np.ndarray,
    val_prob: np.ndarray,
    method: str = "youden",
    grid_step: float = 0.005
) -> Tuple[float, float]:
    """
    Returns (best_threshold, score_used_for_selection).
    If method='youden': score is J (TPR-FPR).
    If method='f1grid': score is F1.
    """
    if method == "youden":
        thr = youden_threshold(y_val, val_prob)
        fpr, tpr, _ = roc_curve(y_val, val_prob)
        score = np.max(tpr - fpr)
        return float(thr), float(score)

    # F1 grid search
    best_f1, best_thr = -1.0, 0.5
    for t in np.arange(0.0, 1.0 + 1e-9, grid_step):
        preds = (val_prob > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1


# ==============================
# Main training / evaluation
# ==============================

def run_experiment(cfg: dict) -> None:
    # ---- Config with defaults ----
    seed = int(cfg.get("seed", 42))
    set_seeds(seed)

    data_cfg = cfg.get("data", {})
    T = int(data_cfg.get("T", 40))
    F = int(data_cfg.get("F", 64))
    C = int(data_cfg.get("C", 1))
    dataset_root = Path(data_cfg.get("dataset_root", "./"))  # user must set
    single_folder = data_cfg.get("single_folder", "Sensor1_BigArea_Single")
    anomaly_patterns = data_cfg.get("anomaly_patterns", ["SwingLR", "SwingFB"])
    normal_test_ratio = float(data_cfg.get("normal_test_ratio", 0.2))
    # keep 30% anomalies for test (no leakage)
    anomaly_test_ratio = float(data_cfg.get("anomaly_test_ratio", 0.3))
    split_seed = int(data_cfg.get("split_seed", seed))

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 100))
    batch_size = int(train_cfg.get("batch_size", 32))
    l2 = float(train_cfg.get("l2", 5e-4))
    dropout_conv = float(train_cfg.get("dropout_conv", 0.3))
    dropout_dense = float(train_cfg.get("dropout_dense", 0.5))

    es_cfg = train_cfg.get("early_stopping", {"enabled": True, "monitor": "val_loss", "patience": 10})
    rl_cfg = train_cfg.get("reduce_lr", {"enabled": True, "patience": 5, "factor": 0.5})

    cv_cfg = cfg.get("cv", {})
    kfolds = int(cv_cfg.get("folds", 5))
    # 'youden' or 'f1grid'
    thr_method = cv_cfg.get("threshold_method", "youden")

    exp_cfg = cfg.get("experiment", {})
    anomaly_fracs = exp_cfg.get("anomaly_fractions", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    runs_cfg = exp_cfg.get("runs_per_fraction", {"default": 10, "for_fraction": {"0.4": 20}})
    default_runs = int(runs_cfg.get("default", 10))
    runs_map = {float(k): int(v) for k, v in runs_cfg.get("for_fraction", {}).items()}

    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "output/exp1"))
    ensure_dir(out_dir)

    # ---- Load data ----
    print("Loading data... (dataset path NOT included; set in config)")
    folder = dataset_root / single_folder
    X, y, names = load_and_label_from_single_folder(
        folder=folder, T=T, F=F, C=C, anomaly_patterns=anomaly_patterns
    )

    # indices
    all_idx = np.arange(len(y))
    normal_idx = all_idx[y == 0]
    anomaly_idx = all_idx[y == 1]

    if len(anomaly_idx) == 0:
        raise RuntimeError("No anomalies found by filename patterns. Check 'anomaly_patterns' and data path.")

    rng = np.random.default_rng(split_seed)
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    # Fixed test split (no leakage): hold out a portion of normals and anomalies
    n_test_norm = max(1, int(normal_test_ratio * len(normal_idx)))
    n_test_anom = max(1, int(anomaly_test_ratio * len(anomaly_idx)))

    test_norm = normal_idx[-n_test_norm:]
    train_norm = normal_idx[:-n_test_norm]

    test_anom = anomaly_idx[-n_test_anom:]
    train_anom_pool = anomaly_idx[:-n_test_anom]

    # If user forces anomaly_test_ratio=1.0, warn about no anomalies left for training
    if len(train_anom_pool) == 0:
        print("[warn] No anomalies left for training (anomaly_test_ratio=1.0). "
              "Training will run with 0 anomaly samples; results will be degenerate.")

    X_test = np.concatenate([X[test_norm], X[test_anom]], axis=0)
    y_test = np.concatenate([y[test_norm], y[test_anom]], axis=0)

    print(f"Total samples: {len(y)}")
    print(f" - Normals: {len(normal_idx)} (train {len(train_norm)}, test {len(test_norm)})")
    print(f" - Anomalies: {len(anomaly_idx)} (train pool {len(train_anom_pool)}, test {len(test_anom)})")

    # ---- Experiment loop ----
    acc_mean, acc_std, f1_mean, f1_std = [], [], [], []
    per_fraction_rows = []

    for frac in anomaly_fracs:
        runs = runs_map.get(float(frac), default_runs)
        print(f"\n===== {int(frac*100)}% anomalies for TRAIN ({runs} runs) =====")

        accs, f1s = [], []

        for run in range(runs):
            set_seeds(seed + run)

            # how many anomalies in training
            n_anom_train = int(frac * len(train_anom_pool))
            if len(train_anom_pool) > 0 and n_anom_train == 0 and frac > 0:
                n_anom_train = 1  # at least 1 if we asked for >0%

            if n_anom_train > len(train_anom_pool):
                n_anom_train = len(train_anom_pool)

            if n_anom_train > 0:
                selected_anom = np.random.choice(train_anom_pool, size=n_anom_train, replace=False)
            else:
                selected_anom = np.array([], dtype=int)

            # Training set = all train normals + selected anomalies (class weights will balance)
            train_idx = np.concatenate([train_norm, selected_anom]) if len(selected_anom) else train_norm
            X_train, y_train = X[train_idx], y[train_idx]

            # Log train counts
            n_tr_norm = int(np.sum(y_train == 0))
            n_tr_anom = int(np.sum(y_train == 1))
            print(f"  [run {run+1:02d}/{runs}] train counts → normal={n_tr_norm}, anomaly={n_tr_anom}")

            # Compute class weights (helps with imbalance) — robust to single-class case
            uniq = np.unique(y_train)
            cw_vals = class_weight.compute_class_weight(class_weight="balanced", classes=uniq, y=y_train)
            cw_dict = {0: 1.0, 1: 1.0}
            for cls, w in zip(uniq, cw_vals):
                cw_dict[int(cls)] = float(w)

            # --------- Adaptive CV to pick threshold ---------
            best_thr = 0.5
            best_score = -1.0  # use J or F1 depending on method

            both_classes = (len(np.unique(y_train)) == 2)
            min_per_class = int(np.bincount(y_train).min()) if both_classes else 0
            n_splits_eff = min(kfolds, max(1, min_per_class))

            # Callbacks
            cb = []
            if es_cfg.get("enabled", True):
                cb.append(callbacks.EarlyStopping(
                    monitor=es_cfg.get("monitor", "val_loss"),
                    patience=int(es_cfg.get("patience", 10)),
                    restore_best_weights=True
                ))
            if rl_cfg.get("enabled", True):
                cb.append(callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    patience=int(rl_cfg.get("patience", 5)),
                    factor=float(rl_cfg.get("factor", 0.5)),
                    min_lr=1e-6
                ))

            if both_classes and n_splits_eff >= 2:
                skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed + run)
                for tr_idx, val_idx in skf.split(X_train, y_train):
                    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                    tf.keras.backend.clear_session()
                    model = build_cnn(
                        input_shape=(T, F, C),
                        l2=l2, dr_conv=dropout_conv, dr_dense=dropout_dense
                    )
                    model.fit(
                        X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        class_weight=cw_dict,
                        callbacks=cb
                    )

                    val_prob = model.predict(X_val, batch_size=batch_size, verbose=0).flatten()
                    thr, score = choose_threshold_from_validation(
                        y_val, val_prob, method=thr_method, grid_step=0.005
                    )
                    if score > best_score:
                        best_score = score
                        best_thr = float(thr)
            else:
                # Not enough samples per class for CV
                if both_classes and min_per_class >= 2:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + run)
                    tr_idx, val_idx = next(sss.split(X_train, y_train))
                    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                    tf.keras.backend.clear_session()
                    model = build_cnn(
                        input_shape=(T, F, C),
                        l2=l2, dr_conv=dropout_conv, dr_dense=dropout_dense
                    )
                    model.fit(
                        X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        class_weight=cw_dict,
                        callbacks=cb
                    )

                    val_prob = model.predict(X_val, batch_size=batch_size, verbose=0).flatten()
                    thr, score = choose_threshold_from_validation(
                        y_val, val_prob, method=thr_method, grid_step=0.005
                    )
                    best_thr = float(thr)
                    best_score = float(score)
                else:
                    print(f"  [warn] Too few samples per class for validation (min={min_per_class}). "
                          f"Using default threshold {best_thr:.2f}.")

            print(f"  tuned threshold={best_thr:.2f} (score={best_score:.3f})")

            # --------- Retrain on full train set, evaluate on fixed test ---------
            tf.keras.backend.clear_session()
            final_model = build_cnn(
                input_shape=(T, F, C),
                l2=l2, dr_conv=dropout_conv, dr_dense=dropout_dense
            )

            # Use a small validation hold-out so callbacks have `val_loss`
            n_train = len(X_train)
            n_val = max(1, int(0.1 * n_train))
            perm = np.random.permutation(n_train)
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]

            X_tr_final, y_tr_final = X_train[tr_idx], y_train[tr_idx]
            X_val_final, y_val_final = X_train[val_idx], y_train[val_idx]

            final_model.fit(
                X_tr_final, y_tr_final,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                class_weight=cw_dict,
                callbacks=cb,
                validation_data=(X_val_final, y_val_final)
            )

            test_prob = final_model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
            test_pred = (test_prob > best_thr).astype(int)

            acc = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred, zero_division=0)

            accs.append(acc)
            f1s.append(f1)

        # Aggregate
        acc_mean.append(float(np.mean(accs)))
        acc_std.append(float(np.std(accs)))
        f1_mean.append(float(np.mean(f1s)))
        f1_std.append(float(np.std(f1s)))

        per_fraction_rows.append({
            "anomaly_fraction": frac,
            "runs": runs,
            "acc_mean": acc_mean[-1],
            "acc_std": acc_std[-1],
            "f1_mean": f1_mean[-1],
            "f1_std": f1_std[-1],
        })

    # ---- Save results ----
    ensure_dir(out_dir)

    # CSV summary
    import pandas as pd
    df = pd.DataFrame(per_fraction_rows)
    df.to_csv(out_dir / "summary_exp1_cnn.csv", index=False)

    # Plots (mean ± std)
    xs = [int(f * 100) for f in anomaly_fracs]

    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, acc_mean, yerr=acc_std, fmt='-o', capsize=5)
    plt.title("Supervised CNN Accuracy ± Std vs. % Anomalies (Train)")
    plt.xlabel("% of anomalies in training set")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_anomaly_frac_meanstd.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.errorbar(xs, f1_mean, yerr=f1_std, fmt='-o', capsize=5)
    plt.title("Supervised CNN F1-Score ± Std vs. % Anomalies (Train)")
    plt.xlabel("% of anomalies in training set")
    plt.ylabel("Test F1-score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_vs_anomaly_frac_meanstd.png")
    plt.close()

    # Also dump the effective config for provenance
    with open(out_dir / "used_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nDone.")
    print(f"Saved CSV → {out_dir/'summary_exp1_cnn.csv'}")
    print(f"Saved plots → {out_dir/'accuracy_vs_anomaly_frac_meanstd.png'}, {out_dir/'f1_vs_anomaly_frac_meanstd.png'}")


# ==============================
# Entry point
# ==============================

def load_config(path: str | None) -> dict:
    """
    Load YAML config if provided; otherwise return defaults.
    """
    base_defaults = {
        "seed": 42,
        "data": {
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # IMPORTANT: set this to your local dataset root (not included)
            # e.g., "/path/to/Coventry2018" (folder that contains Sensor1_BigArea_Single)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            "dataset_root": "./",  # change me before running
            "single_folder": "Sensor1_BigArea_Single",
            "anomaly_patterns": ["SwingLR", "SwingFB"],
            "T": 40,
            "F": 64,
            "C": 1,
            "normal_test_ratio": 0.2,
            "anomaly_test_ratio": 0.3,
            "split_seed": 42,
        },
        "train": {
            "epochs": 100,
            "batch_size": 32,
            "l2": 5e-4,
            "dropout_conv": 0.3,
            "dropout_dense": 0.5,
            "early_stopping": {"enabled": True, "monitor": "val_loss", "patience": 10},
            "reduce_lr": {"enabled": True, "patience": 5, "factor": 0.5},
        },
        "cv": {
            "folds": 5,
            "threshold_method": "youden"  # or "f1grid"
        },
        "experiment": {
            "anomaly_fractions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "runs_per_fraction": {
                "default": 10,
                "for_fraction": {"0.4": 20}
            }
        },
        "output": {
            "dir": "output/exp1"
        }
    }
    if path is None:
        return base_defaults

    if yaml is None:
        print("[warn] PyYAML not installed; using defaults.")
        return base_defaults

    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    # merge (shallow) user_cfg into defaults
    def deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    return deep_merge(base_defaults, user_cfg)


def main(config_path: str | None) -> None:
    cfg = load_config(config_path)
    # sanity note if user tries to emulate the “all anomalies in test” paper split:
    if cfg.get("data", {}).get("anomaly_test_ratio", 0.3) >= 0.999:
        print("[note] anomaly_test_ratio is ~1.0. This leaves 0 anomalies for training. "
              "If you *also* sample anomalies into training, you'll leak test data. "
              "This script prevents leakage by holding out test anomalies and sampling "
              "only from the remaining pool.")

    run_experiment(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional).")
    args = parser.parse_args()
    main(args.config)