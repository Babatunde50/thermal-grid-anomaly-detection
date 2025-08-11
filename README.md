# Infrared Grid Anomaly Detection

Supervised CNN baseline for anomalous human activity detection on low-resolution (8×8) thermal grids.

## Data (not included)
Place the Coventry-2018 dataset CSVs under:
data/raw/Coventry2018/Sensor1_BigArea_Single/*.csv


Do **not** commit data.

## Setup

pip install -r requirements.txt


## Run Exp 1 (Supervised CNN baseline)
python -m src.exp1_cnn_supervised.train --config src/exp1_cnn_supervised/config.yaml

Results (per-run CSV, summary JSON, plots) will be written under `outputs/exp1_cnn_supervised/<timestamp>/`.