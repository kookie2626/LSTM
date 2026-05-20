"""로컬에 저장된 최종 모델을 원격 MLflow 서버에 업로드.

사용법:
    python upload_to_mlflow.py
    python upload_to_mlflow.py --tracking-uri http://121.134.46.24:5000
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from project.ML.data_loader import FEATURE_COLS, TARGET_COL, add_features, load_raw
from project.ML.inference import predict_forecast, predict_anomaly

load_dotenv()

REMOTE_URI  = "http://121.134.46.24:5000"
MODEL_DIR   = Path("outputs/models")

TRAIN_START = "2018-01-01"; TRAIN_END = "2021-12-31"
VAL_START   = "2022-01-01"; VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01"; TEST_END  = "2023-12-31"


# ── 메트릭 계산 헬퍼 ──────────────────────────────────────────────────────────

def forecast_metrics(fc: pd.DataFrame, label: str) -> dict:
    mask = fc["actual"] > 1.0
    mae  = float(fc["error"].abs().mean())
    rmse = float(np.sqrt((fc["error"] ** 2).mean()))
    mape = float(fc["abs_pct_error"][mask].mean()) if mask.any() else float("nan")
    return {
        f"{label}_mae_W":  round(mae,  1),
        f"{label}_rmse_W": round(rmse, 1),
        f"{label}_mape":   round(mape, 2),
    }


def anomaly_metrics(an: pd.DataFrame, label: str) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    # pseudo-label: ANOMALY_PERIOD 안 = 1
    ANOMALY_PERIODS = [("2022-05-06", "2022-07-14")]
    y_true = np.zeros(len(an), dtype=int)
    for s, e in ANOMALY_PERIODS:
        mask = (an.index >= s) & (an.index <= e)
        y_true[np.array(mask)] = 1

    y_pred = (an["anomaly_level"] != "NORMAL").astype(int).values
    y_score = an["vote"].values.astype(float)

    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = float("nan")

    return {
        f"{label}_f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        f"{label}_recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        f"{label}_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        f"{label}_auc_roc":   round(auc, 4),
        f"{label}_n_high":    int((an["anomaly_level"] == "HIGH").sum()),
        f"{label}_n_low":     int((an["anomaly_level"] == "LOW").sum()),
        f"{label}_n_total":   int(len(an)),
    }


# ── 업로드 ────────────────────────────────────────────────────────────────────

def upload_forecast(df: pd.DataFrame, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment("VMD-LSTM-Grid")

    ckpt = torch.load(MODEL_DIR / "vmd_lstm_grid_electricity.pt",
                      map_location="cpu", weights_only=False)
    cfg  = ckpt["model_config"]

    with open(MODEL_DIR / "vmd_lstm_scaler.pkl", "rb") as f:
        scalers = pickle.load(f)

    print("▶ 예측 실행 중 (val)...")
    fc_val  = predict_forecast(df, VAL_START,  VAL_END)
    print("▶ 예측 실행 중 (test)...")
    fc_test = predict_forecast(df, TEST_START, TEST_END)

    params = {
        "model":       "VMD-LSTM",
        "seq_len":     cfg["seq_len"],
        "hidden_dim":  cfg["hidden_dim"],
        "num_layers":  cfg["num_layers"],
        "dropout":     cfg["dropout"],
        "vmd_k":       cfg["vmd_k"],
        "n_features":  cfg["input_dim"],
        "feature_cols": ", ".join(cfg["feature_cols"]),
        "attention":   True,
        "use_amp":     False,
        "scaler":      "MinMaxScaler",
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period":   f"{VAL_START} ~ {VAL_END}",
        "test_period":  f"{TEST_START} ~ {TEST_END}",
    }
    metrics = {
        **forecast_metrics(fc_val,  "val"),
        **forecast_metrics(fc_test, "test"),
    }

    with mlflow.start_run(experiment_id=exp.experiment_id,
                          run_name="vmd-lstm-final") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_DIR / "vmd_lstm_grid_electricity.pt"))
        mlflow.log_artifact(str(MODEL_DIR / "vmd_lstm_scaler.pkl"))
        print(f"  ✓ VMD-LSTM 업로드 완료 | run_id={run.info.run_id[:8]}")
        print(f"    val  MAPE={metrics['val_mape']:.1f}%  MAE={metrics['val_mae_W']/1000:.1f}kW")
        print(f"    test MAPE={metrics['test_mape']:.1f}%  MAE={metrics['test_mae_W']/1000:.1f}kW")
    return run.info.run_id


def upload_anomaly(df: pd.DataFrame, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment("Residual-IF-Anomaly")

    with open(MODEL_DIR / "residual_threshold.pkl", "rb") as f:
        art = pickle.load(f)

    print("▶ 이상탐지 실행 중 (val)...")
    an_val = predict_anomaly(df, VAL_START, VAL_END)

    params = {
        "model":           "Residual+IsolationForest",
        "threshold_W":     round(art["threshold"], 1),
        "best_k":          art["best_k"],
        "res_mean_W":      round(art["res_mean"], 1),
        "res_std_W":       round(art["res_std"],  1),
        "if_contamination": 0.15,
        "k_search_range":  "0.5~4.0 step 0.1",
        "anomaly_label_src": "gateway_failures 2022-05-06~2022-07-14",
    }
    metrics = anomaly_metrics(an_val, "val")

    with mlflow.start_run(experiment_id=exp.experiment_id,
                          run_name="residual-if-final") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_DIR / "residual_threshold.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "residual_iforest.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "residual_if_scaler.pkl"))
        print(f"  ✓ Residual-IF 업로드 완료 | run_id={run.info.run_id[:8]}")
        print(f"    val F1={metrics['val_f1']:.3f}  Recall={metrics['val_recall']:.3f}  "
              f"AUC={metrics['val_auc_roc']:.3f}")
    return run.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=REMOTE_URI)
    args = parser.parse_args()

    print("▶ 데이터 로드 중...")
    df = load_raw()
    df = add_features(df)

    print(f"\n▶ MLflow 서버: {args.tracking_uri}")
    upload_forecast(df, args.tracking_uri)
    upload_anomaly(df, args.tracking_uri)
    print("\n✓ 전체 업로드 완료")


if __name__ == "__main__":
    main()
