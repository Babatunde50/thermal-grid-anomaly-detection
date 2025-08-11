# -*- coding: utf-8 -*-
"""
Experiment 1: Supervised CNN baseline (balanced training, CV thresholding).
- Fixed test split: 20% normals + 30% anomalies (no leakage).
- For each anomaly fraction, sample that % of remaining anomalies for TRAIN
  and match with equal # of normals (balanced).
- 5-fold Stratified CV to choose decision threshold via ROC (Youden's J).
- Retrain on full training set with best threshold; evaluate on fixed test set.
- Save per-run metrics, summary, and plots.

Run:
  python -m src.exp1_cnn_supervised.train --config src/exp1_cnn_supervised/config.yaml
"""

from __future__ import annotations
import os
import csv
import json
import argparse
import random
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# -------------------- utils --------------------


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _parse_numeric_row(row):
    out = []
    for x in row:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out


def load_single_folder_labeled(folder: Path, T=40, F=64, norm_shift=10.0, norm_scale=12.5,
                               anomaly_substrings=None):
    paths = sorted(folder.glob("*.csv"))
    X_list, y_list, files = [], [], []

    for p in paths:
        with open(p, newline="") as f:
            reader = csv.reader(f)
            seq = []
            for row in reader:
                parsed = _parse_numeric_row(row)
                if parsed is None:
                    continue
                if len(parsed) >= F + 1:
                    vals = parsed[1:1+F]
                else:
                    vals = parsed[:F]
                    if len(vals) < F:
                        vals = vals + [vals[-1]] * \
                            (F - len(vals)) if vals else [0.0]*F
                seq.append(vals)

        if len(seq) == 0:
            seq = [[0.0]*F]

        if len(seq) >= T:
            seq = seq[:T]
        else:
            last = seq[-1]
            seq = seq + [last]*(T - len(seq))

        arr = (np.array(seq, dtype=np.float32) - norm_shift) / norm_scale
        arr = arr.reshape(T, F, 1)
        X_list.append(arr)

        fname = p.name
        label = 0
        if anomaly_substrings:
            for s in anomaly_substrings:
                if s in fname:
                    label = 1
                    break
        y_list.append(label)
        files.append(fname)

    if not X_list:
        return np.zeros((0, T, F, 1), dtype=np.float32), np.array([], dtype=np.int32), []

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y, files


def build_cnn(input_shape=(40, 64, 1), l2=5e-4, dr_conv=0.3, dr_dense=0.5):
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Dropout(dr_conv),

        layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(2),
        layers.Dropout(dr_conv),

        layers.Flatten(),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dr_dense),
        layers.Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

# -------------------- main --------------------


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_base = int(cfg.get("seed_base", 1234))
    seed_everything(seed_base)

    T, F, C = cfg["input_shape"]
    data_root = Path(cfg["data_root"])
    single_dir = data_root / cfg["single_dir"]
    anom_keys = cfg.get("anomaly_substrings", ["SwingLR", "SwingFB"])

    # outputs
    save_root = Path("outputs") / cfg["experiment_name"] / \
        datetime.now().strftime("%Y%m%d-%H%M%S")
    ensure_dir(save_root)
    ensure_dir(save_root/"figures")
    ensure_dir(save_root/"metrics")
    with open(save_root/"config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # load all labeled samples from Single
    X, y, files = load_single_folder_labeled(
        single_dir, T=T, F=F, anomaly_substrings=anom_keys)
    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]

    # fixed test split
    rng = np.random.RandomState(seed_base)
    rng.shuffle(normal_idx)
    rng.shuffle(anomaly_idx)

    n_test_norm = max(
        1, int(np.ceil(len(normal_idx)*cfg["test_split"]["normals_frac"])))
    n_test_anom = max(
        1, int(np.ceil(len(anomaly_idx)*cfg["test_split"]["anomalies_frac"])))

    test_norm_idx = normal_idx[:n_test_norm]
    test_anom_idx = anomaly_idx[:n_test_anom]

    train_norm_pool = normal_idx[n_test_norm:]
    train_anom_pool = anomaly_idx[n_test_anom:]

    # fixed test sets
    X_test = np.concatenate([X[test_norm_idx], X[test_anom_idx]], axis=0)
    y_test = np.concatenate([y[test_norm_idx], y[test_anom_idx]], axis=0)

    # record splits (by filename for traceability)
    save_json({
        "test_norm_files": [files[i] for i in test_norm_idx],
        "test_anom_files": [files[i] for i in test_anom_idx],
        "train_norm_pool": [files[i] for i in train_norm_pool],
        "train_anom_pool": [files[i] for i in train_anom_pool],
    }, save_root/"splits.json")

    # config bits
    frac_list = [float(f) for f in cfg["anomaly_fractions"]]
    runs_default = int(cfg.get("runs_default", 10))
    runs_override = {float(k): int(v) for k, v in (
        cfg.get("runs_override", {}) or {}).items()}
    kfolds = int(cfg.get("cv_folds", 5))
    thr_method = cfg["thresholding"]["method"]

    # results aggregators
    mean_acc, std_acc, mean_f1, std_f1 = [], [], [], []
    per_run_rows = []

    for frac in frac_list:
        runs = runs_override.get(frac, runs_default)
        print(
            f"\n===== {int(frac*100)}% anomalies for TRAIN ({runs} runs) =====")

        accs, f1s = [], []

        # training pool sizes
        n_train_anom_max = len(train_anom_pool)
        n_train_anom = max(1, int(round(frac*n_train_anom_max)))

        for run in range(runs):
            seed = seed_base + run + int(frac*1000)
            seed_everything(seed)
            rng = np.random.RandomState(seed)

            # sample anomalies for train from pool (no leakage)
            chosen_anom = rng.choice(
                train_anom_pool, size=n_train_anom, replace=False)

            # balanced: match same # normals
            chosen_norm = rng.choice(
                train_norm_pool, size=n_train_anom, replace=False)

            train_idx = np.concatenate([chosen_norm, chosen_anom])
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_train, y_train = shuffle(X_train, y_train, random_state=seed)

            # CV to pick threshold
            best_thr = 0.5
            best_j = -1.0

            skf = StratifiedKFold(
                n_splits=kfolds, shuffle=True, random_state=seed)
            for tr_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                model = build_cnn(input_shape=(T, F, C),
                                  l2=cfg["train"]["l2"],
                                  dr_conv=cfg["train"]["dropout_conv"],
                                  dr_dense=cfg["train"]["dropout_dense"])

                cb = []
                es = cfg["train"]["early_stopping"]
                if es.get("enabled", True):
                    cb.append(callbacks.EarlyStopping(
                        monitor=es.get("monitor", "val_loss"),
                        patience=es.get("patience", 10),
                        restore_best_weights=True))
                rl = cfg["train"]["reduce_lr"]
                if rl.get("enabled", True):
                    cb.append(callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        patience=rl.get("patience", 5),
                        factor=rl.get("factor", 0.5),
                        min_lr=1e-6))

                model.fit(X_tr, y_tr,
                          validation_data=(X_val, y_val),
                          epochs=cfg["train"]["epochs"],
                          batch_size=cfg["train"]["batch_size"],
                          verbose=0, callbacks=cb)

                val_prob = model.predict(X_val, verbose=0).flatten()

                if thr_method == "youden":
                    thr = youden_threshold(y_val, val_prob)
                    fpr, tpr, _ = roc_curve(y_val, val_prob)
                    j = np.max(tpr - fpr)
                    if j > best_j:
                        best_j = j
                        best_thr = float(thr)
                else:
                    # fallback: maximize F1 on a fine grid
                    best_local_f1, best_local_thr = -1, 0.5
                    for t in np.arange(0.0, 1.01, 0.005):
                        preds = (val_prob > t).astype(int)
                        f1 = f1_score(y_val, preds, zero_division=0)
                        if f1 > best_local_f1:
                            best_local_f1, best_local_thr = f1, float(t)
                    if best_local_f1 > best_j:
                        best_j = best_local_f1
                        best_thr = best_local_thr

            # retrain on full train set
            final_model = build_cnn(input_shape=(T, F, C),
                                    l2=cfg["train"]["l2"],
                                    dr_conv=cfg["train"]["dropout_conv"],
                                    dr_dense=cfg["train"]["dropout_dense"])
            cb_final = []
            es = cfg["train"]["early_stopping"]
            if es.get("enabled", True):
                cb_final.append(callbacks.EarlyStopping(
                    monitor=es.get("monitor", "val_loss"),
                    patience=es.get("patience", 10),
                    restore_best_weights=True))
            rl = cfg["train"]["reduce_lr"]
            if rl.get("enabled", True):
                cb_final.append(callbacks.ReduceLROnPlateau(
                    monitor="val_loss", patience=rl.get("patience", 5),
                    factor=rl.get("factor", 0.5), min_lr=1e-6))
            # use a small val split to drive callbacks
            final_model.fit(X_train, y_train,
                            validation_split=0.2,
                            epochs=cfg["train"]["epochs"],
                            batch_size=cfg["train"]["batch_size"],
                            verbose=0, callbacks=cb_final)

            # test
            test_prob = final_model.predict(X_test, verbose=0).flatten()
            test_pred = (test_prob > best_thr).astype(int)

            acc = accuracy_score(y_test, test_pred)
            f1 = f1_score(y_test, test_pred, zero_division=0)

            accs.append(acc)
            f1s.append(f1)
            per_run_rows.append({
                "fraction": frac, "run": run,
                "n_train_anom": int(n_train_anom),
                "threshold": float(best_thr),
                "accuracy": float(acc),
                "f1": float(f1)
            })

        mean_acc.append(float(np.mean(accs)))
        std_acc.append(float(np.std(accs)))
        mean_f1.append(float(np.mean(f1s)))
        std_f1.append(float(np.std(f1s)))

        print(
            f"→ Acc {np.mean(accs):.4f} ± {np.std(accs):.4f} | F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # save per-run CSV and summary
    import pandas as pd
    df = pd.DataFrame(per_run_rows)
    df.to_csv(save_root/"metrics"/"per_run_metrics.csv", index=False)

    summary = {
        "fractions": frac_list,
        "accuracy_mean": mean_acc, "accuracy_std": std_acc,
        "f1_mean": mean_f1, "f1_std": std_f1
    }
    save_json(summary, save_root/"metrics"/"summary.json")

    # plots
    x = [int(f*100) for f in frac_list]

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, mean_acc, yerr=std_acc, fmt='-o', capsize=5)
    plt.title("Supervised CNN Accuracy ± Std vs. % Anomalies")
    plt.xlabel("% anomalies in training set")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root/"figures"/"accuracy_vs_anomaly_frac_meanstd.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.errorbar(x, mean_f1, yerr=std_f1, fmt='-o', capsize=5)
    plt.title("Supervised CNN F1-Score ± Std vs. % Anomalies")
    plt.xlabel("% anomalies in training set")
    plt.ylabel("Test F1-score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_root/"figures"/"f1_vs_anomaly_frac_meanstd.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
