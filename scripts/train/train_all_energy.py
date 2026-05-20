"""EMS 전체 에너지 카테고리 일괄 학습 (VMD-LSTM + Residual+IF).

타겟 7종:
  grid_P      - electricity/total/P  : 총 전력 소비 (W)
  pv_gen      - electricity/pv/P     : 태양광 발전량 (W, abs 처리)
  chp_gen     - electricity/chp/P    : 열병합 발전량 (W, abs 처리)
  cool_P      - cooling/total/P      : 냉방 총 소비 (W)
  cool_elec_P - cooling/cool_elec/P  : 냉동기 소비 (W)
  heat_P      - heating/total/P      : 난방 총 소비 (W)
  chp_heat_P  - heating/chp_heat/P   : 열병합 열 출력 (W)

각 타겟마다:
  - 다른 에너지 6종 + 기상(Ta/Igm) + 시간 sin/cos 6개 = 14개 기본 피처
  - VMD(K=4) IMF lag-1 4개 + 목표값 lag(24h/168h/336h) 3개 → 총 21개 피처
  - LSTM Forecaster (2-layer, hidden=192, attention) → 1h ahead 예측
  - 예측 잔차 + Isolation Forest 앙상블 → 이상탐지

RunPod 실행:
    PYTHONPATH=/workspace python scripts/train/train_all_energy.py
    PYTHONPATH=/workspace python scripts/train/train_all_energy.py --target grid_P
    PYTHONPATH=/workspace python scripts/train/train_all_energy.py --skip-existing

로컬 실행:
    PYTHONPATH=/home/keun/workspace python scripts/train/train_all_energy.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD

load_dotenv()

# ── 설정 ─────────────────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01";  TRAIN_END = "2021-12-31"
VAL_START   = "2022-01-01";  VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01";  TEST_END  = "2023-12-31"

GATEWAY_FAILURES = [
    ("2020-02-13", "2020-03-06"),
    ("2020-08-20", "2020-09-17"),
    ("2021-11-15", "2021-12-10"),
    ("2022-05-06", "2022-07-14"),
]
ANOMALY_PERIODS = [("2022-05-06", "2022-07-14")]

SEQ_LEN        = 24
BATCH_SIZE     = 256
EPOCHS         = 100
LR             = 1e-3
HIDDEN_DIM     = 192
NUM_LAYERS     = 2
DROPOUT        = 0.2
EARLY_STOP_PAT = 15
VMD_K          = 4
VMD_ALPHA      = 2000
IF_CONTAM      = 0.15
K_SEARCH       = np.arange(0.5, 4.1, 0.1)

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

OUT_BASE = Path("outputs/models/energy")
OUT_BASE.mkdir(parents=True, exist_ok=True)

CONNECT_KWARGS = {
    "host":     os.environ["DB_HOST"],
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "dbname":   os.environ["DB_NAME"],
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

TIME_COLS    = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
WEATHER_COLS = ["Ta", "Igm"]

# (DB category, DB subcategory, abs_sign)
TARGET_DEFS: dict[str, tuple[str, str, bool]] = {
    "grid_P":      ("electricity", "total",     False),
    "pv_gen":      ("electricity", "pv",        True),   # 발전 → abs
    "chp_gen":     ("electricity", "chp",       True),   # 발전 → abs
    "cool_P":      ("cooling",     "total",     False),
    "cool_elec_P": ("cooling",     "cool_elec", False),
    "heat_P":      ("heating",     "total",     False),
    "chp_heat_P":  ("heating",     "chp_heat",  False),
}
ALL_ENERGY_COLS = list(TARGET_DEFS.keys())

# DB 컬럼 → 내부 이름 매핑
COL_MAP = {
    ("electricity", "total",     "P"):   "grid_P",
    ("electricity", "pv",        "P"):   "pv_gen",
    ("electricity", "chp",       "P"):   "chp_gen",
    ("cooling",     "total",     "P"):   "cool_P",
    ("cooling",     "cool_elec", "P"):   "cool_elec_P",
    ("heating",     "total",     "P"):   "heat_P",
    ("heating",     "chp_heat",  "P"):   "chp_heat_P",
    ("weather",     "weather",   "Ta"):  "Ta",
    ("weather",     "weather",   "Igm"): "Igm",
}


# ── 모델 ─────────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.w(x), dim=1)
        return (x * weights).sum(dim=1)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, **_):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = _Attention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.attn(out)).squeeze(-1)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_all_energy() -> pd.DataFrame:
    """모든 에너지 측정값 + 기상 데이터를 wide DataFrame으로 반환.

    - pv_gen / chp_gen: DB에 발전량이 음수로 저장 → abs 처리
    - 결측: 0으로 채움 (pv는 2019-06 이전 데이터 없음 → 0)
    """
    sql = """
        SELECT ts, category, subcategory, measurement, value
        FROM ems.reduced_measurement_1h
        WHERE
            (measurement = 'P' AND (
                (category = 'electricity' AND subcategory IN ('total','pv','chp'))
                OR (category = 'cooling'  AND subcategory IN ('total','cool_elec'))
                OR (category = 'heating'  AND subcategory IN ('total','chp_heat'))
            ))
            OR (category = 'weather' AND subcategory = 'weather'
                AND measurement IN ('Ta','Igm'))
        ORDER BY ts
    """
    with psycopg.connect(**CONNECT_KWARGS) as conn:
        df_long = pd.read_sql(sql, conn)

    df_long["col"] = df_long.apply(
        lambda r: COL_MAP.get((r["category"], r["subcategory"], r["measurement"])),
        axis=1,
    )
    df_long = df_long.dropna(subset=["col"])

    df = df_long.pivot_table(index="ts", columns="col", values="value", aggfunc="first")
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    df["pv_gen"]  = df["pv_gen"].abs()
    df["chp_gen"] = df["chp_gen"].abs()

    ts = df.index
    df["hour_sin"]  = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * ts.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * ts.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

    for col in ALL_ENERGY_COLS + WEATHER_COLS + TIME_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def get_splits(df: pd.DataFrame):
    train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    mask  = pd.Series(False, index=train.index)
    for s, e in GATEWAY_FAILURES:
        mask |= (train.index >= s) & (train.index <= e)
    train = train[~mask]
    val  = df[(df.index >= VAL_START)  & (df.index <= VAL_END)].copy()
    test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
    return train, val, test


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 100.0) -> float:
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def pseudo_labels(timestamps: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=int)
    dates  = pd.to_datetime(timestamps).date
    for start, end in ANOMALY_PERIODS:
        s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        labels[(dates >= s) & (dates <= e)] = 1
    return labels


def run_inference(model, split_df, feature_cols, target_col, scaler_X, scaler_y):
    """DataFrame 입력 → y_true, y_pred (W 단위), residuals, timestamps."""
    X  = scaler_X.transform(split_df[feature_cols])
    yt = scaler_y.transform(split_df[[target_col]]).ravel()
    seqs   = np.array([X[i: i + SEQ_LEN] for i in range(len(X) - SEQ_LEN)], dtype=np.float32)
    ts_arr = split_df.index[SEQ_LEN:].to_numpy()

    model.eval()
    with torch.no_grad():
        preds_s = model(torch.from_numpy(seqs).to(DEVICE)).cpu().numpy()

    y_pred_w = np.maximum(scaler_y.inverse_transform(preds_s.reshape(-1, 1)).ravel(), 0)
    y_true_w = scaler_y.inverse_transform(yt[SEQ_LEN:].reshape(-1, 1)).ravel()
    residuals = np.abs(y_true_w - y_pred_w)

    mae  = mean_absolute_error(y_true_w, y_pred_w)
    rmse = float(np.sqrt(mean_squared_error(y_true_w, y_pred_w)))
    mp   = calc_mape(y_true_w, y_pred_w)

    return y_true_w, y_pred_w, residuals, ts_arr, mae, rmse, mp


# ── 타겟별 학습 ───────────────────────────────────────────────────────────────

def train_one_target(df_src: pd.DataFrame, target: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  타겟: {target}")
    print(f"{'='*60}")

    df = df_src.copy()
    out_dir = OUT_BASE / target
    out_dir.mkdir(parents=True, exist_ok=True)

    other_energy  = [c for c in ALL_ENERGY_COLS if c != target and c in df.columns]
    base_feat_cols = other_energy + WEATHER_COLS + TIME_COLS

    # ── VMD 분해 ──────────────────────────────────────────────────────────────
    print(f"▶ VMD 분해 중 (K={VMD_K}, alpha={VMD_ALPHA})...")
    signal = df[target].values.astype(float)
    u, _, _ = VMD(signal, VMD_ALPHA, 0, VMD_K, 0, 1, 1e-7)
    imf_cols = []
    for i in range(VMD_K):
        col = f"imf{i+1}_lag1"
        df[col] = np.roll(u[i], 1)   # lag-1 (shift by 1)
        df[col].iloc[0] = 0
        imf_cols.append(col)

    # ── lag 피처 ──────────────────────────────────────────────────────────────
    lag_cols = []
    for lag_h in [24, 168, 336]:
        col = f"lag{lag_h}h"
        df[col] = df[target].shift(lag_h).fillna(0)
        lag_cols.append(col)

    feature_cols = base_feat_cols + imf_cols + lag_cols
    print(f"▶ 피처 수: {len(feature_cols)}"
          f"  (에너지 {len(other_energy)} + 기상 {len(WEATHER_COLS)}"
          f" + 시간 {len(TIME_COLS)} + VMD {len(imf_cols)} + lag {len(lag_cols)})")

    # ── 분할 ──────────────────────────────────────────────────────────────────
    train, val, test = get_splits(df)
    print(f"  학습 : {train.index[0].date()} ~ {train.index[-1].date()} ({len(train):,}행)")
    print(f"  검증 : {val.index[0].date()}   ~ {val.index[-1].date()}   ({len(val):,}행)")
    print(f"  테스트: {test.index[0].date()} ~ {test.index[-1].date()}  ({len(test):,}행)")

    # ── 스케일링 ──────────────────────────────────────────────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[feature_cols])
    y_train = scaler_y.fit_transform(train[[target]]).ravel()
    X_val   = scaler_X.transform(val[feature_cols])
    y_val   = scaler_y.transform(val[[target]]).ravel()

    X_tr_s, y_tr_s = make_sequences(X_train, y_train, SEQ_LEN)
    X_va_s, y_va_s = make_sequences(X_val,   y_val,   SEQ_LEN)

    def make_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            pin_memory=(DEVICE == "cuda"), num_workers=4, persistent_workers=True,
        )

    train_loader = make_loader(X_tr_s, y_tr_s, shuffle=True)
    val_loader   = make_loader(X_va_s, y_va_s, shuffle=False)

    # ── LSTM 학습 ─────────────────────────────────────────────────────────────
    model     = LSTMForecaster(len(feature_cols), HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    print(f"▶ LSTM 학습 중 (epochs={EPOCHS}, early_stop={EARLY_STOP_PAT}, device={DEVICE})...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_losses.append(criterion(model(Xb.to(DEVICE)), yb.to(DEVICE)).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={t_loss:.6f} | val={v_loss:.6f}"
                  f" | no_improve={no_improve}")

        if no_improve >= EARLY_STOP_PAT:
            print(f"  Early stopping at epoch {epoch}")
            mlflow.log_metric("early_stop_epoch", epoch)
            break

    model.load_state_dict(best_state)

    # ── 예측 + 잔차 계산 (train/val/test) ────────────────────────────────────
    print("▶ 예측 평가 중...")
    _, _, train_res, _, _, _, _ = run_inference(
        model, train, feature_cols, target, scaler_X, scaler_y)
    _, _, val_res, val_ts, val_mae, val_rmse, val_mape = run_inference(
        model, val,   feature_cols, target, scaler_X, scaler_y)
    _, _, test_res, test_ts, test_mae, test_rmse, test_mape = run_inference(
        model, test,  feature_cols, target, scaler_X, scaler_y)

    print(f"\n  val  MAE={val_mae:,.1f}W  RMSE={val_rmse:,.1f}W  MAPE={val_mape:.2f}%")
    print(f"  test MAE={test_mae:,.1f}W  RMSE={test_rmse:,.1f}W  MAPE={test_mape:.2f}%")

    res_mean = train_res.mean()
    res_std  = train_res.std()
    print(f"  train 잔차 mean={res_mean:,.1f}W  std={res_std:,.1f}W")

    # ── Isolation Forest 학습 ─────────────────────────────────────────────────
    print("▶ Isolation Forest 학습 중...")
    if_feature_cols = other_energy + WEATHER_COLS + TIME_COLS
    scaler_if       = MinMaxScaler()
    X_if_train      = scaler_if.fit_transform(train[if_feature_cols])
    iso = IsolationForest(contamination=IF_CONTAM, random_state=42, n_jobs=-1)
    iso.fit(X_if_train)

    n_val, n_test = len(val_res), len(test_res)
    X_if_val  = scaler_if.transform(val[if_feature_cols])
    X_if_test = scaler_if.transform(test[if_feature_cols])
    if_val  = (iso.predict(X_if_val[SEQ_LEN:  SEQ_LEN + n_val])  == -1).astype(int)
    if_test = (iso.predict(X_if_test[SEQ_LEN: SEQ_LEN + n_test]) == -1).astype(int)

    # ── 임계치 최적화 (val pseudo-label 기준) ─────────────────────────────────
    print("▶ 임계치 최적화 중 (val F1 기준)...")
    y_true_val = pseudo_labels(val_ts[:n_val])
    best_k, best_f1 = 2.0, 0.0
    for k in K_SEARCH:
        thr      = res_mean + k * res_std
        res_flag = (val_res > thr).astype(int)
        y_pred   = (res_flag + if_val >= 1).astype(int)
        f1 = f1_score(y_true_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, round(float(k), 1)

    threshold = res_mean + best_k * res_std
    print(f"  최적 k={best_k}  val F1={best_f1:.3f}  threshold={threshold:,.1f}W")

    # ── 이상탐지 평가 ─────────────────────────────────────────────────────────
    def eval_anomaly(res_arr, if_flag, ts_arr, n):
        y_true   = pseudo_labels(ts_arr[:n])
        res_flag = (res_arr > threshold).astype(int)
        y_pred   = (res_flag + if_flag >= 1).astype(int)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, res_arr) if len(np.unique(y_true)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        return f1, prec, rec, auc

    val_f1,  val_prec,  val_rec,  val_auc  = eval_anomaly(val_res,  if_val,  val_ts,  n_val)
    test_f1, test_prec, test_rec, test_auc = eval_anomaly(test_res, if_test, test_ts, n_test)

    print(f"  val  F1={val_f1:.3f}  Prec={val_prec:.3f}  Rec={val_rec:.3f}  AUC={val_auc:.3f}")
    print(f"  test F1={test_f1:.3f}  Prec={test_prec:.3f}  Rec={test_rec:.3f}  AUC={test_auc:.3f}")

    # ── 아티팩트 저장 ─────────────────────────────────────────────────────────
    model_path = out_dir / "lstm_model.pt"
    torch.save({
        "model_state":  best_state,
        "model_config": {
            "input_dim":    len(feature_cols),
            "hidden_dim":   HIDDEN_DIM,
            "num_layers":   NUM_LAYERS,
            "dropout":      DROPOUT,
            "seq_len":      SEQ_LEN,
            "vmd_k":        VMD_K,
            "feature_cols": feature_cols,
            "target":       target,
        },
    }, model_path)

    scaler_path    = out_dir / "scaler.pkl"
    threshold_path = out_dir / "residual_threshold.pkl"
    iforest_path   = out_dir / "iforest.pkl"
    if_scaler_path = out_dir / "if_scaler.pkl"

    with open(scaler_path,    "wb") as f: pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    with open(threshold_path, "wb") as f: pickle.dump({
        "threshold": float(threshold), "res_mean": float(res_mean),
        "res_std":   float(res_std),   "best_k":   best_k,
    }, f)
    with open(iforest_path,   "wb") as f: pickle.dump(iso,       f)
    with open(if_scaler_path, "wb") as f: pickle.dump(scaler_if, f)

    # ── MLflow 기록 ───────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=target):
        mlflow.log_params({
            "target":           target,
            "n_features":       len(feature_cols),
            "hidden_dim":       HIDDEN_DIM,
            "seq_len":          SEQ_LEN,
            "vmd_k":            VMD_K,
            "best_k":           best_k,
            "threshold_W":      round(float(threshold), 1),
            "if_contamination": IF_CONTAM,
        })
        mlflow.log_metrics({
            "val_mae":   round(val_mae,   2),
            "val_rmse":  round(val_rmse,  2),
            "val_mape":  round(val_mape,  4) if not np.isnan(val_mape)  else -1,
            "val_f1":    round(val_f1,    4),
            "val_auc":   round(val_auc,   4) if not np.isnan(val_auc)   else -1,
            "test_mae":  round(test_mae,  2),
            "test_rmse": round(test_rmse, 2),
            "test_mape": round(test_mape, 4) if not np.isnan(test_mape) else -1,
            "test_f1":   round(test_f1,   4),
            "test_auc":  round(test_auc,  4) if not np.isnan(test_auc)  else -1,
        })
        for path in [model_path, scaler_path, threshold_path, iforest_path]:
            mlflow.log_artifact(str(path))

    print(f"▶ 저장 완료: {out_dir}/")

    return {
        "target":      target,
        "status":      "OK",
        "val_mae":     round(val_mae,   1),
        "val_rmse":    round(val_rmse,  1),
        "val_mape":    round(val_mape,  2) if not np.isnan(val_mape)  else float("nan"),
        "val_f1":      round(val_f1,    4),
        "val_auc":     round(val_auc,   4) if not np.isnan(val_auc)   else float("nan"),
        "test_mae":    round(test_mae,  1),
        "test_rmse":   round(test_rmse, 1),
        "test_mape":   round(test_mape, 2) if not np.isnan(test_mape) else float("nan"),
        "test_f1":     round(test_f1,   4),
        "test_auc":    round(test_auc,  4) if not np.isnan(test_auc)  else float("nan"),
        "best_k":      best_k,
        "threshold_W": round(float(threshold), 1),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMS 전체 에너지 카테고리 일괄 학습")
    parser.add_argument("--target",        default=None,
                        help=f"단일 타겟만 학습. 선택: {', '.join(TARGET_DEFS)}")
    parser.add_argument("--skip-existing", action="store_true",
                        help="lstm_model.pt 이미 존재하면 건너뜀")
    args = parser.parse_args()

    mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[2] / "mlruns"))
    mlflow.set_experiment("All-Energy")

    print(f"▶ 디바이스: {DEVICE}")
    print("▶ 데이터 로드 중 (DB → wide DataFrame)...")
    df = load_all_energy()
    print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df):,}행)")
    print(f"  에너지 컬럼: {[c for c in ALL_ENERGY_COLS if c in df.columns]}")

    targets = [args.target] if args.target else list(TARGET_DEFS.keys())

    results: list[dict] = []
    for target in targets:
        if target not in TARGET_DEFS:
            print(f"⚠  알 수 없는 타겟: {target}  (선택: {', '.join(TARGET_DEFS)})")
            continue

        model_path = OUT_BASE / target / "lstm_model.pt"
        if args.skip_existing and model_path.exists():
            print(f"⏭  {target} — 이미 학습됨, 건너뜀")
            continue

        try:
            result = train_one_target(df, target)
        except Exception as e:
            print(f"\n✗ {target} 학습 실패: {e}")
            traceback.print_exc()
            result = {"target": target, "status": f"ERROR: {e}"}

        results.append(result)
        pd.DataFrame(results).to_csv("outputs/all_energy_results.csv", index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/all_energy_results.csv", index=False)

    print("\n" + "=" * 60)
    print("✓ 전체 완료")
    ok = results_df[results_df.get("status", pd.Series()) == "OK"] if "status" in results_df.columns else results_df
    if len(ok) > 0 and "val_mae" in ok.columns:
        cols = ["target", "val_mae", "val_mape", "test_mae", "test_mape", "val_f1"]
        print(ok[[c for c in cols if c in ok.columns]].to_string(index=False))
    print(f"\n▶ 결과 저장: outputs/all_energy_results.csv")
    print(f"▶ 모델 경로: {OUT_BASE}/{{target}}/")


if __name__ == "__main__":
    main()
