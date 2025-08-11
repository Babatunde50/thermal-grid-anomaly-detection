# Unsupervised AE (Exp 2) — saves four plots matching manuscript:
# - output/exp2/heldout_error_hist.png
# - output/exp2/confusion_matrix.png
# - output/exp2/classification_metrics.png
# - output/exp2/rmse_hist.png
from __future__ import annotations
import os, glob, csv, random, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -----------------------------
# Helpers
# -----------------------------
def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def load_folder(folder: Path, label: int, img_shape=(40, 64, 1)):
    """
    Load every CSV in `folder` into X (N,T,F,C) and y (N,) with value `label`.
    Robust to an optional first timestamp/index column and headers.
    Normalization: (x - 10)/12.5
    """
    paths = sorted(glob.glob(str(folder / "*.csv")))
    N = len(paths)
    T, F, C = img_shape

    X = np.zeros((N, T, F, C), dtype=np.float32)
    y = np.full((N,), label, dtype=np.int32)

    for i, p in enumerate(paths):
        rows = []
        with open(p, "r", newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                rows.append(r)

        frames = []
        for r in rows:
            if not r:
                continue
            # try first F values
            if len(r) >= F and all(is_float(v) for v in r[:F]):
                vals = [float(v) for v in r[:F]]
            # else try skipping the first col
            elif len(r) >= F + 1 and all(is_float(v) for v in r[1:1+F]):
                vals = [float(v) for v in r[1:1+F]]
            else:
                continue
            frames.append(vals)

        # if empty/malformed: keep zeros
        if len(frames) == 0:
            continue

        # truncate/pad to T
        frames = frames[:T]
        if len(frames) < T:
            last = frames[-1]
            frames = frames + [last] * (T - len(frames))

        arr = np.asarray(frames, dtype=np.float32)  # (T,F)
        arr = (arr - 10.0) / 12.5
        X[i, :, :, 0] = arr

    return X, y

def build_autoencoder(input_shape):
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x   = layers.MaxPooling2D(2, padding="same")(x)
    x   = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x   = layers.MaxPooling2D(2, padding="same")(x)
    x   = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    x   = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

# -----------------------------
# Main
# -----------------------------
def main(dataset_root: str):
    # config (kept same as your original)
    batch_size  = 10
    epochs      = 50
    T, F, C     = 40, 64, 1
    input_shape = (T, F, C)

    # folders (join with dataset_root)
    root = Path(dataset_root)
    normal_dir = root / "Sensor1_BigArea_Single"
    anom_dir   = root / "Sensor1_BigArea_Double"

    print("Loading data…")
    x_norm, _  = load_folder(normal_dir, label=0, img_shape=input_shape)
    x_anom, _  = load_folder(anom_dir,   label=1, img_shape=input_shape)

    if x_norm.shape[0] == 0:
        raise RuntimeError(f"No normal CSVs found in {normal_dir}")
    if x_anom.shape[0] == 0:
        raise RuntimeError(f"No anomaly CSVs found in {anom_dir}")

    # 80/20 split on normals
    indices = list(range(len(x_norm)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    x_train = x_norm[indices[:split]]
    x_val   = x_norm[indices[split:]]

    # build + train AE (normals only)
    autoencoder = build_autoencoder(input_shape)
    autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, x_val),
        verbose=0
    )

    # threshold from held-out normals
    x_val_rec = autoencoder.predict(x_val, verbose=0)
    err_val   = np.mean(
        (x_val.reshape(len(x_val), -1) - x_val_rec.reshape(len(x_val_rec), -1))**2,
        axis=1
    )
    threshold = np.median(err_val) + 3 * np.std(err_val)
    print(f"Anomaly threshold: {threshold:.6f}")

    # evaluate on true anomalies
    x_anom_rec = autoencoder.predict(x_anom, verbose=0)
    err_anom   = np.mean(
        (x_anom.reshape(len(x_anom), -1) - x_anom_rec.reshape(len(x_anom_rec), -1))**2,
        axis=1
    )

    y_true = np.concatenate([np.zeros_like(err_val), np.ones_like(err_anom)])
    y_pred = np.concatenate([
        (err_val  > threshold).astype(int),
        (err_anom > threshold).astype(int)
    ])

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    print("Classification report:\n",
          classification_report(y_true, y_pred, target_names=["Normal","Anomaly"], zero_division=0))

    # -----------------------------
    # Save the FOUR plots (exact names/paths)
    # -----------------------------
    os.makedirs("output/exp2", exist_ok=True)

    # 1) Histogram of held-out MSE (normals)
    plt.figure(figsize=(6,4))
    plt.hist(err_val, bins=30)
    plt.title("Reconstruction Error on Held-Out Normals")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/exp2/heldout_error_hist.png")
    plt.close()

    # 2) Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal","Anomaly"]
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix")
    th = cm.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > th else "black")
    fig.tight_layout()
    fig.savefig("output/exp2/confusion_matrix.png")
    plt.close()

    # 3) Bar chart of classification metrics
    metrics = [acc, prec, rec, f1]
    names   = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(names, metrics, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0.95, 1.01)  # zoom into top range for clarity (as in paper)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for bar, m in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.002, f"{m:.4f}", ha="center", va="bottom")
    ax.set_title("Classification Metrics", pad=12)
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig("output/exp2/classification_metrics.png")
    plt.close()

    # 4) RMSE distributions
    rmse_val  = np.sqrt(err_val)
    rmse_anom = np.sqrt(err_anom)
    plt.figure(figsize=(6,4))
    plt.hist(rmse_val,  bins=30, alpha=0.6, label="Held-Out Normals")
    plt.hist(rmse_anom, bins=30, alpha=0.6, label="Anomalies")
    plt.title("RMSE Distributions")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/exp2/rmse_hist.png")
    plt.close()

    # save metrics csv
    import pandas as pd
    df = pd.DataFrame([{
        "Experiment": "Exp2_AE",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }])
    df.to_csv("output/exp2/statistics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=".",
                        help="Path to Coventry2018 root (contains Sensor1_BigArea_Single/Double)")
    args = parser.parse_args()
    main(args.dataset_root)