"""예측 잔차 기반 이상탐지 (Forecast Residual + Isolation Forest 앙상블).

기존 LSTM-AE 방식 대체:
  - VMD-LSTM 예측 잔차(|actual - predicted|)를 주요 이상 스코어로 사용
  - Isolation Forest로 다변량 컨텍스트 보완
  - val pseudo-label로 임계치 k 자동 최적화 (threshold = mean + k * std)

AUC 0.547 → 개선 기대:
  AE 방식은 "재구성 어려운 패턴"을 이상으로 보지만,
  게이트웨이 장애는 "예측과 다른 값"으로 나타나므로 잔차가 더 직접적인 신호.

RunPod 실행:
    python train_anomaly_residual.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from project.ML.data_loader import FEATURE_COLS, TARGET_COL, add_features, get_splits, load_raw
from project.ML.inference import LSTMForecaster, _Attention  # 모델 정의 재사용

load_dotenv()

MODEL_DIR = Path("outputs/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IF_CONTAM   = 0.15
K_SEARCH    = np.arange(0.5, 4.1, 0.1)   # 임계치 탐색 범위
ANOMALY_PERIODS = [("2022-05-06", "2022-07-14")]  # val pseudo-label

# ── 유틸 ─────────────────────────────────────────────────────────────────────

def pseudo_labels(timestamps: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=int)
    ts_dates = pd.to_datetime(timestamps).date
    for start, end in ANOMALY_PERIODS:
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        labels[(ts_dates >= s) & (ts_dates <= e)] = 1
    return labels


def print_metrics(name: str, y_true, y_pred, scores):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"  [{name}] Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  "
          f"F1={f1:.3f}  AUC={auc:.3f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


# ── VMD-LSTM 잔차 계산 ─────────────────────────────────────────────────────

def compute_residuals(df: pd.DataFrame, split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """저장된 VMD-LSTM으로 split 구간 예측 잔차 반환.

    Returns:
        residuals: |actual - predicted| (W 단위)
        timestamps: 잔차에 대응하는 타임스탬프
    """
    ckpt = torch.load(MODEL_DIR / "vmd_lstm_grid_electricity.pt",
                      map_location=DEVICE, weights_only=False)
    cfg  = ckpt["model_config"]

    model = LSTMForecaster(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    with open(MODEL_DIR / "vmd_lstm_scaler.pkl", "rb") as f:
        scalers = pickle.load(f)
    scaler_X, scaler_y = scalers["scaler_X"], scalers["scaler_y"]

    feat_cols = cfg["feature_cols"]
    seq_len   = cfg["seq_len"]

    # VMD + lag 피처는 df 전체에 이미 적용돼 있다고 가정 (main에서 주입)
    missing = [c for c in feat_cols if c not in split.columns]
    if missing:
        raise ValueError(f"피처 누락: {missing}")

    X      = scaler_X.transform(split[feat_cols])
    y_true = scaler_y.transform(split[[TARGET_COL]]).ravel()

    seqs = np.array([X[i: i + seq_len] for i in range(len(X) - seq_len)], dtype=np.float32)
    ts   = split.index[seq_len:].to_numpy()

    with torch.no_grad():
        preds_s = model(torch.from_numpy(seqs).to(DEVICE)).cpu().numpy()

    y_pred_w = np.maximum(
        scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel(), 0
    )
    y_true_w = scaler_y.inverse_transform(y_true[seq_len:].reshape(-1, 1)).ravel()

    residuals = np.abs(y_true_w - y_pred_w)
    return residuals, ts


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[2] / "mlruns"))
    mlflow.set_experiment("Residual-IF-Anomaly")

    print(f"▶ 디바이스: {DEVICE}")
    print("▶ 데이터 로드 중...")
    df = load_raw()
    df = add_features(df)

    # VMD + lag 피처 생성 (train_vmd_lstm.py와 동일)
    from vmdpy import VMD
    print("▶ VMD 분해 중...")
    signal = df[TARGET_COL].values.astype(float)
    u, _, _ = VMD(signal, 2000, 0, 4, 0, 1, 1e-7)
    for i in range(4):
        col = f"imf_{i + 1}_lag1"
        df[col] = u[i]
        df[col] = df[col].shift(1).fillna(0)
    for lag_h in [168, 336, 504, 672]:
        df[f"grid_P_lag{lag_h}h"] = df[TARGET_COL].shift(lag_h).fillna(0)

    train, val, test = get_splits(df)
    print(f"  학습 : {train.index[0].date()} ~ {train.index[-1].date()}")
    print(f"  검증 : {val.index[0].date()} ~ {val.index[-1].date()}")
    print(f"  테스트: {test.index[0].date()} ~ {test.index[-1].date()}")

    # ── 잔차 계산 ────────────────────────────────────────────────────────────
    print("▶ 잔차 계산 중 (train)...")
    train_res, _      = compute_residuals(df, train)

    print("▶ 잔차 계산 중 (val)...")
    val_res,   val_ts = compute_residuals(df, val)

    print("▶ 잔차 계산 중 (test)...")
    test_res, test_ts = compute_residuals(df, test)

    res_mean = train_res.mean()
    res_std  = train_res.std()
    print(f"\n  train 잔차  mean={res_mean:.1f} W  std={res_std:.1f} W")

    # ── Isolation Forest 학습 ────────────────────────────────────────────────
    print("▶ Isolation Forest 학습 중...")
    from sklearn.preprocessing import MinMaxScaler
    scaler_if = MinMaxScaler()
    X_train_if = scaler_if.fit_transform(train[FEATURE_COLS])
    iso = IsolationForest(contamination=IF_CONTAM, random_state=42, n_jobs=-1)
    iso.fit(X_train_if)

    # ── val IF 스코어 ─────────────────────────────────────────────────────────
    ckpt     = torch.load(MODEL_DIR / "vmd_lstm_grid_electricity.pt",
                          map_location="cpu", weights_only=False)
    seq_len  = ckpt["model_config"]["seq_len"]
    n_val    = len(val_res)

    X_val_if = scaler_if.transform(val[FEATURE_COLS])
    if_val   = iso.predict(X_val_if[seq_len: seq_len + n_val])
    if_flag_val = (if_val == -1).astype(int)

    X_test_if = scaler_if.transform(test[FEATURE_COLS])
    if_test   = iso.predict(X_test_if[seq_len: seq_len + len(test_res)])
    if_flag_test = (if_test == -1).astype(int)

    # ── 임계치 최적화 (val pseudo-label 기준) ───────────────────────────────
    print("▶ 임계치 최적화 중 (val F1 기준)...")
    y_true_val = pseudo_labels(val_ts[:n_val])
    best_k, best_f1 = 2.0, 0.0
    for k in K_SEARCH:
        threshold = res_mean + k * res_std
        res_flag  = (val_res > threshold).astype(int)
        vote      = res_flag + if_flag_val
        y_pred    = (vote >= 1).astype(int)
        f1 = f1_score(y_true_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, round(float(k), 1)

    threshold = res_mean + best_k * res_std
    print(f"  최적 k={best_k}  val F1={best_f1:.3f}  threshold={threshold:.1f} W")

    # ── 최종 평가 ─────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="Residual+IF"):
        mlflow.log_params({
            "model":          "Residual+IF",
            "if_contamination": IF_CONTAM,
            "best_k":         best_k,
            "threshold_W":    round(float(threshold), 1),
            "res_mean_W":     round(float(res_mean), 1),
            "res_std_W":      round(float(res_std), 1),
        })

        for split_name, res_arr, if_flag, ts_arr in [
            ("val",  val_res,  if_flag_val,  val_ts),
            ("test", test_res, if_flag_test, test_ts),
        ]:
            n = len(res_arr)
            res_flag = (res_arr > threshold).astype(int)
            vote     = res_flag + if_flag
            y_pred   = (vote >= 1).astype(int)
            y_true   = pseudo_labels(ts_arr[:n])

            print(f"\n▶ {split_name} 이상탐지 결과")
            print(f"  전체 {n:,}개 | 이상 pseudo-label: {y_true.sum():,}개 "
                  f"({y_true.mean()*100:.1f}%)")
            print(f"  탐지: HIGH={(vote==2).sum():,}  LOW={(vote==1).sum():,}  "
                  f"NORMAL={(vote==0).sum():,}")

            m = print_metrics(split_name, y_true, y_pred, res_arr)
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()})

        # ── 저장 ──────────────────────────────────────────────────────────────
        artifact = {
            "threshold":  float(threshold),
            "res_mean":   float(res_mean),
            "res_std":    float(res_std),
            "best_k":     best_k,
        }
        res_path = MODEL_DIR / "residual_threshold.pkl"
        with open(res_path, "wb") as f:
            pickle.dump(artifact, f)

        if_path = MODEL_DIR / "residual_iforest.pkl"
        with open(if_path, "wb") as f:
            pickle.dump(iso, f)

        if_scaler_path = MODEL_DIR / "residual_if_scaler.pkl"
        with open(if_scaler_path, "wb") as f:
            pickle.dump(scaler_if, f)

        mlflow.log_artifact(str(res_path))
        mlflow.log_artifact(str(if_path))
        print(f"\n▶ 임계치 저장: {res_path}")
        print(f"▶ IF 모델 저장: {if_path}")


if __name__ == "__main__":
    main()
