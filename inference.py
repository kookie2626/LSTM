"""저장된 VMD-LSTM + IF+LSTM-AE 모델로 추론 실행.

사용법:
    python inference.py --start 2022-01-01 --end 2022-12-31
    python inference.py --start 2023-06-01 --end 2023-06-30 --mode forecast
    python inference.py --start 2022-05-01 --end 2022-08-01 --mode anomaly
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv

from project.ML.data_loader import FEATURE_COLS, TARGET_COL, add_features, load_raw

load_dotenv()

MODEL_DIR = Path("outputs/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 모델 정의 (학습 스크립트와 동일) ──────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.w(x), dim=1)
        return (x * weights).sum(dim=1)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, **_):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = _Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.attn(out)).squeeze(-1)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len, dropout=0.1, **_):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.enc_attn     = nn.Linear(hidden_dim, 1)
        self.enc_fc       = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc       = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.out_fc       = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder_lstm(x)
        w = torch.softmax(self.enc_attn(out), dim=1)
        return self.enc_fc((out * w).sum(dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.decoder_lstm(h)
        return self.out_fc(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_forecast_model():
    ckpt = torch.load(MODEL_DIR / "vmd_lstm_grid_electricity.pt",
                      map_location=DEVICE, weights_only=False)
    cfg = ckpt["model_config"]
    model = LSTMForecaster(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    with open(MODEL_DIR / "vmd_lstm_scaler.pkl", "rb") as f:
        scalers = pickle.load(f)

    return model, scalers, cfg


def load_anomaly_model():
    ckpt = torch.load(MODEL_DIR / "anomaly_lstmae.pt",
                      map_location=DEVICE, weights_only=False)
    cfg = ckpt["model_config"]
    model = LSTMAutoencoder(**cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    with open(MODEL_DIR / "anomaly_iforest.pkl", "rb") as f:
        iso = pickle.load(f)

    with open(MODEL_DIR / "anomaly_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, iso, scaler, float(ckpt["threshold"])


# ── 피처 준비 ─────────────────────────────────────────────────────────────────

def _build_forecast_features(df: pd.DataFrame, vmd_k: int, feature_cols: list[str]) -> pd.DataFrame:
    """VMD 분해 + weekly lag 피처 추가 (학습과 동일한 방식)."""
    from vmdpy import VMD
    signal = df[TARGET_COL].values.astype(float)
    u, _, _ = VMD(signal, 2000, 0, vmd_k, 0, 1, 1e-7)
    for i in range(vmd_k):
        col = f"imf_{i + 1}_lag1"
        df[col] = u[i]
        df[col] = df[col].shift(1).fillna(0)
    for lag_h in [168, 336]:
        col = f"grid_P_lag{lag_h}h"
        df[col] = df[TARGET_COL].shift(lag_h).fillna(0)
    # feature_cols에 없는 컬럼이 생겼을 경우 대비해 순서 맞춤
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"피처 누락: {missing}")
    return df


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([X[i: i + seq_len] for i in range(len(X) - seq_len)], dtype=np.float32)


# ── 추론 ─────────────────────────────────────────────────────────────────────

def predict_forecast(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """VMD-LSTM 전력 소비량 예측.

    Returns:
        ts 인덱스 DataFrame: actual(W), predicted(W), error(W), abs_pct_error(%)
    """
    model, scalers, cfg = load_forecast_model()
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]
    seq_len  = cfg["seq_len"]
    feat_cols = cfg["feature_cols"]
    vmd_k    = cfg["vmd_k"]

    df = _build_forecast_features(df.copy(), vmd_k, feat_cols)
    target = df[(df.index >= start) & (df.index <= end)]

    X = scaler_X.transform(target[feat_cols])
    y_true = scaler_y.transform(target[[TARGET_COL]]).ravel()
    seqs = _make_sequences(X, seq_len)
    ts   = target.index[seq_len:]

    with torch.no_grad():
        preds_s = model(torch.from_numpy(seqs).to(DEVICE)).cpu().numpy()

    y_pred_w = np.maximum(
        scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel(), 0
    )
    y_true_w = scaler_y.inverse_transform(y_true[seq_len:].reshape(-1, 1)).ravel()

    mask = y_true_w > 1.0
    ape = np.where(mask, np.abs((y_true_w - y_pred_w) / y_true_w) * 100, np.nan)

    return pd.DataFrame({
        "actual":        y_true_w,
        "predicted":     y_pred_w,
        "error":         y_true_w - y_pred_w,
        "abs_pct_error": ape,
    }, index=ts)


def predict_anomaly(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """예측 잔차 + Isolation Forest 앙상블 이상탐지.

    VMD-LSTM 예측 잔차(|actual - predicted|)를 주요 이상 스코어로 사용.
    val pseudo-label로 최적화된 임계치 적용.

    Returns:
        ts 인덱스 DataFrame: residual(W), res_flag, if_flag, vote, anomaly_level
    """
    # 잔차 임계치 + IF 모델 로드
    with open(MODEL_DIR / "residual_threshold.pkl", "rb") as f:
        art = pickle.load(f)
    with open(MODEL_DIR / "residual_iforest.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(MODEL_DIR / "residual_if_scaler.pkl", "rb") as f:
        scaler_if = pickle.load(f)

    threshold = art["threshold"]

    # VMD-LSTM 예측 → 잔차
    fc = predict_forecast(df, start, end)
    residuals = fc["error"].abs().values
    ts        = fc.index

    # IF 플래그 (원본 피처 기준)
    target  = df[(df.index >= start) & (df.index <= end)]
    ckpt    = torch.load(MODEL_DIR / "vmd_lstm_grid_electricity.pt",
                         map_location="cpu", weights_only=False)
    seq_len = ckpt["model_config"]["seq_len"]
    n       = len(residuals)

    X_if    = scaler_if.transform(target[FEATURE_COLS])
    if_pred = iso.predict(X_if[seq_len: seq_len + n])
    if_flag = (if_pred == -1).astype(int)

    res_flag = (residuals > threshold).astype(int)
    vote     = res_flag + if_flag
    level    = np.where(vote == 2, "HIGH", np.where(vote == 1, "LOW", "NORMAL"))

    return pd.DataFrame({
        "actual":        fc["actual"].values,
        "predicted":     fc["predicted"].values,
        "residual":      residuals,
        "res_flag":      res_flag,
        "if_flag":       if_flag,
        "vote":          vote,
        "anomaly_level": level,
    }, index=ts)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMS 추론 실행")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end",   required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--mode",  choices=["forecast", "anomaly", "both"],
                        default="both")
    parser.add_argument("--out",   default="outputs", help="결과 저장 디렉토리")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("▶ 데이터 로드 중...")
    df = load_raw()
    df = add_features(df)

    if args.mode in ("forecast", "both"):
        print(f"▶ 예측 실행 ({args.start} ~ {args.end})...")
        fc = predict_forecast(df, args.start, args.end)
        mae  = fc["error"].abs().mean()
        mape = fc["abs_pct_error"].mean()
        print(f"  MAE : {mae:,.0f} W")
        print(f"  MAPE: {mape:.2f}%")
        path = out_dir / f"forecast_{args.start}_{args.end}.csv"
        fc.to_csv(path)
        print(f"  저장: {path}")

    if args.mode in ("anomaly", "both"):
        print(f"▶ 이상탐지 실행 ({args.start} ~ {args.end})...")
        an = predict_anomaly(df, args.start, args.end)
        n_high = (an["anomaly_level"] == "HIGH").sum()
        n_low  = (an["anomaly_level"] == "LOW").sum()
        total  = len(an)
        print(f"  HIGH: {n_high:,} ({n_high/total*100:.1f}%)  "
              f"LOW: {n_low:,} ({n_low/total*100:.1f}%)  "
              f"NORMAL: {total-n_high-n_low:,}")
        path = out_dir / f"anomaly_{args.start}_{args.end}.csv"
        an.to_csv(path)
        print(f"  저장: {path}")


if __name__ == "__main__":
    main()
