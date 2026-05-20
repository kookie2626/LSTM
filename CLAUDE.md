# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EMS (Energy Management System) ML pipeline for two tasks:
1. **Electricity forecasting** — predict `grid_P` (total power consumption, W) using VMD-LSTM
2. **Anomaly detection** — detect gateway failure periods using residual + Isolation Forest ensemble

Data source: PostgreSQL DB (`ems.reduced_measurement_1h` for aggregate data, `ems.cr_measurement_1h` for individual meters). Connection config is in `.env`.

## Environment Setup

```bash
source .venv_gemini/bin/activate
pip install -r requirements_runpod.txt
```

## Running Scripts

```bash
# Train forecasting model (grid_P, VMD-LSTM)
python scripts/train/train_vmd_lstm.py

# Train anomaly detection — residual + IF (current approach)
python scripts/train/train_anomaly_residual.py

# Batch train all individual meters (VMD-LSTM + residual IF per meter)
python scripts/train/train_all_meters.py
python scripts/train/train_all_meters.py --skip-existing   # skip already-trained meters
python scripts/train/train_all_meters.py --meter H1.Z10    # single meter only

# Run inference
python inference.py --start 2022-01-01 --end 2022-12-31
python inference.py --start 2023-06-01 --end 2023-06-30 --mode forecast
python inference.py --start 2022-05-01 --end 2022-08-01 --mode anomaly

# Visualize results
python scripts/plot/visualize.py --start 2022-01-01 --end 2022-12-31

# Upload trained models to remote MLflow server
python scripts/report/upload_to_mlflow.py
```

> `scripts/train/train_anomaly_ifae.py` (LSTM-AE + IF) is the legacy approach, superseded by `train_anomaly_residual.py`.

## Architecture

### Repository Layout

```
ML/
├── data_loader.py          # shared: DB load, feature engineering, splits
├── inference.py            # shared: model class definitions + inference functions
├── scripts/
│   ├── train/              # training scripts (4 files)
│   ├── report/             # report generation + MLflow upload (5 files)
│   └── plot/               # visualization scripts (5 files)
├── docs/                   # project_documentation.md, detailed_metrics.md
└── outputs/
    ├── models/             # saved checkpoints and scalers
    └── ...                 # CSV results and HTML reports
```

### Data Pipeline (`data_loader.py`)
Central module imported by all training/inference scripts via `from project.ML.data_loader import ...`. Defines:
- `load_raw()` — pivots long-format DB data into wide DataFrame
- `add_features()` — appends 6 cyclic time features (hour/dow/month sin+cos); fills missing values with 0
- `get_splits()` — returns train/val/test DataFrames with gateway failure rows removed from train only
- `FEATURE_COLS` (11 features), `TARGET_COL = "grid_P"`, split boundaries, `GATEWAY_FAILURES` list

**Data splits:** Train 2018–2021 / Val 2022 / Test 2023. Gateway failures (4 periods, 2020–2022) excluded from train only. The 2022-05-06–2022-07-14 period doubles as the anomaly pseudo-label for val evaluation.

### Model Architectures

Model class definitions live in `inference.py` and are imported by training scripts (`from project.ML.inference import ...`) to avoid duplication.

**LSTMForecaster** (forecasting):
- 2-layer LSTM → attention pooling → FC(hidden→128→1) in `train_vmd_lstm.py` / `inference.py`; FC(hidden→64→1) in `train_all_meters.py`
- Loss: HuberLoss, optimizer: Adam + ReduceLROnPlateau, gradient clipping at 1.0

**Residual + IF** (`scripts/train/train_anomaly_residual.py` — current approach):
- Uses VMD-LSTM forecast residuals as primary anomaly signal
- Isolation Forest on raw features as secondary signal
- Threshold k auto-tuned over val pseudo-labels (k ∈ [0.5, 4.0])
- Anomaly levels: HIGH (both signals), LOW (one signal), NORMAL

**LSTMAutoencoder** (`scripts/train/train_anomaly_ifae.py` — legacy):
- 2-layer LSTM encoder → attention → latent FC → decoder LSTM → reconstruction
- Anomaly score: MSE reconstruction error; threshold = mean + k×std of train errors (k fixed at 2.0)

### Per-Meter Training (`scripts/train/train_all_meters.py`)
Self-contained script (does not import `data_loader.py`). Trains VMD-LSTM + residual IF for each meter from `ems.cr_measurement_1h`. Skips meters with fewer than 2000 train rows. Saves artifacts to `outputs/models/meters/{meter_urn}/`.

### VMD Preprocessing
`scripts/train/train_vmd_lstm.py` decomposes `grid_P` into K=4 IMF components via VMD (alpha=2000), adds lag-1 of each IMF plus 168h/336h lags of `grid_P`, expanding features from 11 to 17. `train_all_meters.py` skips VMD for speed and uses 24h/48h/168h/336h lag features instead.

### MLflow Tracking
- **Remote server**: `http://121.134.46.24:5000` (configured in `.env` as `MLFLOW_TRACKING_URI`)
- **Local fallback**: `file:///workspace/mlruns`
- Training writes to **local** `mlruns/`; `upload_to_mlflow.py` explicitly pushes to the remote server
- Experiments: `SSA-IPSO-LSTM` (forecasting), `Residual-IF-Anomaly` (residual anomaly), `LSTM-AE-Anomaly` (legacy), `All-Meters` (batch)
- Remote upload uses experiment names `VMD-LSTM-Grid` and `Residual-IF-Anomaly`

### Output Artifacts
```
outputs/
  models/
    vmd_lstm_grid_electricity.pt   # forecasting model checkpoint
    vmd_lstm_scaler.pkl            # scaler_X, scaler_y
    residual_threshold.pkl         # threshold, res_mean, res_std, best_k
    residual_iforest.pkl           # IF model (residual approach)
    residual_if_scaler.pkl         # IF feature scaler
    anomaly_lstmae.pt              # legacy LSTM-AE checkpoint
    anomaly_iforest.pkl            # legacy IF model
    anomaly_scaler.pkl             # legacy scaler
    meters/{meter_urn}/            # per-meter artifacts (same structure)
  all_meters_results.csv
  all_meters_summary.csv
  all_meters_detailed_report.html
  team_report.html
```

## Key Design Decisions

- Missing values are always filled with 0 (team policy, not a default)
- Val/test splits retain all rows including failure periods; only training excludes them
- `train_anomaly_residual.py` supersedes `train_anomaly_ifae.py` — residual-based approach has better alignment with gateway failure characteristics
- `inference.py` is the single source of truth for model class definitions; training scripts import from it
