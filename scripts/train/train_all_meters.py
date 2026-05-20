"""81개 계량기 전체 VMD-LSTM + Residual+IF 자동 학습.

- 데이터 소스: ems.cr_measurement_1h (개별 계량기)
- 날씨 피처: ems.reduced_measurement_1h (Ta, Igm)
- 학습 조건: train 구간 2000행 이상인 계량기만 학습
- 저장 경로: outputs/models/meters/{meter_urn}/
- MLflow 실험: All-Meters

실행:
    python train_all_meters.py
    python train_all_meters.py --skip-existing   # 이미 학습된 것 건너뜀
    python train_all_meters.py --meter H1.Z10    # 특정 계량기만
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────────────────
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
IF_CONTAM      = 0.15
K_SEARCH       = np.arange(0.5, 4.1, 0.1)
MIN_TRAIN_ROWS = 2000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

OUT_BASE = Path("outputs/models/meters")
OUT_BASE.mkdir(parents=True, exist_ok=True)

CONNECT_KWARGS = {
    "host":     os.environ["DB_HOST"],
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "dbname":   os.environ["DB_NAME"],
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}


# ── 모델 정의 ─────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return (x * torch.softmax(self.w(x), dim=1)).sum(dim=1)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = _Attention(hidden_dim)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.attn(out)).squeeze(-1)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_weather() -> pd.DataFrame:
    """날씨 데이터 (Ta, Igm) 로드."""
    sql = """
        SELECT ts, subcategory || '_' || measurement AS col, value
        FROM ems.reduced_measurement_1h
        WHERE category = 'weather' AND subcategory = 'weather'
              AND measurement IN ('Ta', 'Igm')
        ORDER BY ts
    """
    with psycopg.connect(**CONNECT_KWARGS) as conn:
        df = pd.read_sql(sql, conn)
    df = df.pivot_table(index="ts", columns="col", values="value", aggfunc="first")
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = ["Igm", "Ta"]
    return df.sort_index()


def load_meter(meter_urn: str) -> pd.Series:
    """개별 계량기 P 데이터 로드 (단건 조회 — 배치 처리 시 load_all_meters 사용 권장)."""
    sql = f"""
        SELECT ts, value
        FROM ems.cr_measurement_1h
        WHERE meter_urn = '{meter_urn}' AND measurement = 'P'
        ORDER BY ts
    """
    with psycopg.connect(**CONNECT_KWARGS) as conn:
        df = pd.read_sql(sql, conn)
    df.index = pd.to_datetime(df["ts"], utc=True)
    return df["value"].rename("target_P").sort_index()


def load_all_meters(meter_list: list[str]) -> pd.DataFrame:
    """80개 계량기 P 데이터를 쿼리 1번으로 일괄 로드.

    Returns: wide DataFrame — index=ts(UTC), columns=meter_urn
    """
    placeholders = ",".join(f"'{m}'" for m in meter_list)
    sql = f"""
        SELECT ts, meter_urn, value
        FROM ems.cr_measurement_1h
        WHERE meter_urn IN ({placeholders}) AND measurement = 'P'
        ORDER BY ts
    """
    with psycopg.connect(**CONNECT_KWARGS) as conn:
        df = pd.read_sql(sql, conn)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    wide = df.pivot_table(index="ts", columns="meter_urn", values="value", aggfunc="first")
    wide.sort_index(inplace=True)
    return wide


def get_meter_list() -> list[str]:
    sql = """
        SELECT m.meter_urn
        FROM ems.full_meter m
        WHERE EXISTS (
            SELECT 1 FROM ems.cr_measurement_1h c
            WHERE c.meter_urn = m.meter_urn AND c.measurement = 'P'
        )
        ORDER BY m.meter_group, m.meter_urn
    """
    with psycopg.connect(**CONNECT_KWARGS) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [r[0] for r in cur.fetchall()]


# ── 피처 엔지니어링 ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """시간 피처 + 주간 lag (VMD 생략 — 계량기 80개 배치 처리 속도 우선)."""
    ts = df.index
    df["hour_sin"]  = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * ts.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * ts.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

    for lag_h in [24, 48, 168, 336]:
        df[f"target_P_lag{lag_h}h"] = df["target_P"].shift(lag_h).fillna(0)

    df = df.fillna(0)

    feat_cols = [
        "target_P", "Ta", "Igm",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "target_P_lag24h", "target_P_lag48h", "target_P_lag168h", "target_P_lag336h",
    ]
    return df, feat_cols


def make_splits(df: pd.DataFrame):
    mask = pd.Series(False, index=df.index)
    for s, e in GATEWAY_FAILURES:
        mask |= (df.index >= s) & (df.index <= e)
    train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END) & ~mask]
    val   = df[(df.index >= VAL_START)   & (df.index <= VAL_END)]
    test  = df[(df.index >= TEST_START)  & (df.index <= TEST_END)]
    return train, val, test


def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def pseudo_labels(timestamps):
    labels = np.zeros(len(timestamps), dtype=int)
    ts_dates = pd.to_datetime(timestamps).date
    for s, e in ANOMALY_PERIODS:
        labels[(ts_dates >= pd.Timestamp(s).date()) &
               (ts_dates <= pd.Timestamp(e).date())] = 1
    return labels


# ── 학습 ──────────────────────────────────────────────────────────────────────

def train_one_meter(meter_urn: str, weather: pd.DataFrame,
                    all_meters: pd.DataFrame | None = None) -> dict:
    """단일 계량기 학습. 결과 dict 반환.

    all_meters: load_all_meters()로 미리 로드한 wide DataFrame.
                None이면 개별 DB 쿼리로 폴백.
    """
    result = {"meter_urn": meter_urn, "status": "OK"}

    # ── 데이터 준비 ──
    if all_meters is not None and meter_urn in all_meters.columns:
        meter_s = all_meters[meter_urn].rename("target_P").dropna()
    else:
        meter_s = load_meter(meter_urn)
    df = weather.copy()
    df["target_P"] = meter_s.reindex(df.index).fillna(0)

    df, feat_cols = build_features(df)
    train, val, test = make_splits(df)

    if len(train) < MIN_TRAIN_ROWS:
        result["status"] = f"SKIP (train={len(train)}행)"
        return result

    # ── 스케일링 ──
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = scaler_X.fit_transform(train[feat_cols])
    y_train = scaler_y.fit_transform(train[["target_P"]]).ravel()
    X_val   = scaler_X.transform(val[feat_cols])
    y_val   = scaler_y.transform(val[["target_P"]]).ravel()
    X_test  = (scaler_X.transform(test[feat_cols]) if len(test) > SEQ_LEN
               else np.zeros((0, len(feat_cols))))
    y_test  = (scaler_y.transform(test[["target_P"]]).ravel() if len(test) > SEQ_LEN
               else np.zeros(0))

    Xtr_s, ytr_s = make_sequences(X_train, y_train, SEQ_LEN)
    Xva_s, yva_s = make_sequences(X_val,   y_val,   SEQ_LEN)

    def make_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            pin_memory=(DEVICE == "cuda"), num_workers=2, persistent_workers=True,
        )

    tr_loader = make_loader(Xtr_s, ytr_s, True)
    va_loader = make_loader(Xva_s, yva_s, False)

    # ── VMD-LSTM 학습 ──
    model = LSTMForecaster(len(feat_cols), HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-5)
    crit  = nn.HuberLoss()

    best_val_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for Xb, yb in va_loader:
                v_losses.append(crit(model(Xb.to(DEVICE)), yb.to(DEVICE)).item())
        v_loss = float(np.mean(v_losses))
        sched.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP_PAT:
            break

    model.load_state_dict(best_state)

    # ── val 평가 ──
    def predict(X_seq):
        seqs = np.array([X_seq[i: i + SEQ_LEN] for i in range(len(X_seq) - SEQ_LEN)],
                        dtype=np.float32)
        preds = []
        with torch.no_grad():
            for i in range(0, len(seqs), 512):
                preds.append(model(torch.from_numpy(seqs[i:i+512]).to(DEVICE)).cpu().numpy())
        return np.concatenate(preds)

    def inv(arr, sc): return sc.inverse_transform(arr.reshape(-1, 1)).ravel()

    yhat_val = np.maximum(inv(predict(X_val), scaler_y), 0)
    ytru_val = inv(y_val[SEQ_LEN:], scaler_y)
    val_mae  = float(np.abs(ytru_val - yhat_val).mean())
    val_rmse = float(np.sqrt(np.mean((ytru_val - yhat_val) ** 2)))
    val_sum  = float(np.sum(ytru_val))
    val_wape = float(np.sum(np.abs(ytru_val - yhat_val)) / val_sum * 100) if val_sum > 0 else float("nan")
    _mask_val = ytru_val > 100
    val_mape = float(np.mean(np.abs((ytru_val[_mask_val] - yhat_val[_mask_val]) / ytru_val[_mask_val])) * 100) if _mask_val.sum() > 0 else float("nan")

    test_mae = test_rmse = test_wape = test_mape = float("nan")
    if len(X_test) > SEQ_LEN:
        yhat_te   = np.maximum(inv(predict(X_test), scaler_y), 0)
        ytru_te   = inv(y_test[SEQ_LEN:], scaler_y)
        test_mae  = float(np.abs(ytru_te - yhat_te).mean())
        test_rmse = float(np.sqrt(np.mean((ytru_te - yhat_te) ** 2)))
        test_sum  = float(np.sum(ytru_te))
        test_wape = float(np.sum(np.abs(ytru_te - yhat_te)) / test_sum * 100) if test_sum > 0 else float("nan")
        _mask_te  = ytru_te > 100
        test_mape = float(np.mean(np.abs((ytru_te[_mask_te] - yhat_te[_mask_te]) / ytru_te[_mask_te])) * 100) if _mask_te.sum() > 0 else float("nan")

    result.update({"val_mae": val_mae, "val_rmse": val_rmse,
                   "val_mape": val_mape, "val_wape": val_wape,
                   "test_mae": test_mae, "test_rmse": test_rmse,
                   "test_mape": test_mape, "test_wape": test_wape})

    # ── 잔차 기반 이상탐지 ──
    def residuals(X_seq, y_true_scaled):
        yhat = np.maximum(inv(predict(X_seq), scaler_y), 0)
        ytru = inv(y_true_scaled[SEQ_LEN:], scaler_y)
        return np.abs(ytru - yhat)

    train_res = residuals(X_train, y_train)
    val_res   = residuals(X_val,   y_val)
    res_mean, res_std = train_res.mean(), train_res.std()

    scaler_if = MinMaxScaler()
    X_if_tr   = scaler_if.fit_transform(train[["target_P", "Ta", "Igm"]])
    iso       = IsolationForest(contamination=IF_CONTAM, random_state=42, n_jobs=-1)
    iso.fit(X_if_tr)

    n_val    = len(val_res)
    X_if_val = scaler_if.transform(val[["target_P", "Ta", "Igm"]])
    if_flag_val = (iso.predict(X_if_val[SEQ_LEN: SEQ_LEN + n_val]) == -1).astype(int)

    y_true_val_lbl = pseudo_labels(val.index[SEQ_LEN: SEQ_LEN + n_val].to_numpy())
    best_k, best_f1 = 2.0, 0.0
    for k in K_SEARCH:
        thr   = res_mean + k * res_std
        vote  = (val_res > thr).astype(int) + if_flag_val
        f1    = f1_score(y_true_val_lbl, (vote >= 1).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_k = f1, round(float(k), 1)

    threshold = res_mean + best_k * res_std
    res_flag  = (val_res > threshold).astype(int)
    vote      = res_flag + if_flag_val
    y_pred    = (vote >= 1).astype(int)
    val_prec  = precision_score(y_true_val_lbl, y_pred, zero_division=0)
    val_rec   = recall_score(y_true_val_lbl, y_pred, zero_division=0)
    val_auc   = (roc_auc_score(y_true_val_lbl, val_res)
                 if len(np.unique(y_true_val_lbl)) > 1 else float("nan"))

    result.update({"val_f1": best_f1, "val_precision": val_prec,
                   "val_recall": val_rec, "val_auc": val_auc,
                   "best_k": best_k, "threshold_W": float(threshold)})

    # ── 저장 ──
    out_dir = OUT_BASE / meter_urn.replace(".", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state":  best_state,
        "model_config": {
            "input_dim": len(feat_cols), "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "seq_len": SEQ_LEN, "vmd_k": 0, "feature_cols": feat_cols,
        },
    }, out_dir / "vmd_lstm.pt")

    with open(out_dir / "vmd_lstm_scaler.pkl", "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    with open(out_dir / "residual_threshold.pkl", "wb") as f:
        pickle.dump({"threshold": float(threshold), "res_mean": float(res_mean),
                     "res_std": float(res_std), "best_k": best_k}, f)
    with open(out_dir / "residual_iforest.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(out_dir / "residual_if_scaler.pkl", "wb") as f:
        pickle.dump(scaler_if, f)

    return result


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--meter", default=None, help="특정 계량기만 학습")
    args = parser.parse_args()

    mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[2] / "mlruns"))
    mlflow.set_experiment("All-Meters")

    print(f"▶ 디바이스: {DEVICE}")
    print("▶ 날씨 데이터 로드 중...")
    weather = load_weather()

    meters = [args.meter] if args.meter else get_meter_list()
    print(f"▶ 대상 계량기: {len(meters)}개")

    print("▶ 계량기 데이터 일괄 로드 중 (DB 쿼리 1회)...")
    all_meters_data = load_all_meters(meters)
    print(f"  로드 완료: {all_meters_data.shape[1]}개 계량기\n")

    summary = []
    for i, meter_urn in enumerate(meters, 1):
        out_dir = OUT_BASE / meter_urn.replace(".", "_")
        if args.skip_existing and (out_dir / "vmd_lstm.pt").exists():
            print(f"[{i:2d}/{len(meters)}] {meter_urn:<22} SKIP (기존 모델 존재)")
            continue

        t0 = time.time()
        print(f"[{i:2d}/{len(meters)}] {meter_urn:<22} 학습 중...", end=" ", flush=True)

        try:
            res = train_one_meter(meter_urn, weather, all_meters_data)

            mlflow.end_run()  # stale run 정리
            with mlflow.start_run(run_name=meter_urn):
                mlflow.log_param("meter_urn", meter_urn)
                if res["status"] == "OK":
                    mlflow.log_metrics({k: v for k, v in res.items()
                                        if isinstance(v, float) and not np.isnan(v)})

            elapsed = time.time() - t0
            if res["status"] == "OK":
                print(f"완료 ({elapsed:.0f}s) | "
                      f"val MAE={res['val_mae']/1000:.1f}kW "
                      f"WAPE={res['val_wape']:.1f}% "
                      f"F1={res['val_f1']:.3f}")
            else:
                print(res["status"])

        except Exception as e:
            res = {"meter_urn": meter_urn, "status": f"ERROR: {e}"}
            print(f"ERROR: {e}")

        summary.append(res)

    # ── 결과 요약 ──
    print("\n" + "=" * 70)
    ok   = [r for r in summary if r.get("status") == "OK"]
    skip = [r for r in summary if r.get("status", "").startswith("SKIP")]
    err  = [r for r in summary if r.get("status", "").startswith("ERROR")]

    print(f"완료: {len(ok)}개  스킵: {len(skip)}개  오류: {len(err)}개")
    if ok:
        def _avg(key): return [r[key] for r in ok if not np.isnan(r.get(key, float("nan")))]
        maes  = _avg("val_mae");  rmses = _avg("val_rmse")
        mapes = _avg("val_mape"); wapes = _avg("val_wape"); f1s = _avg("val_f1")
        print(f"val MAE  평균: {np.mean(maes)/1000:.1f} kW")
        print(f"val RMSE 평균: {np.mean(rmses)/1000:.1f} kW")
        print(f"val MAPE 평균: {np.mean(mapes):.1f}%")
        print(f"val WAPE 평균: {np.mean(wapes):.1f}%")
        print(f"val F1   평균: {np.mean(f1s):.3f}")

    # CSV 저장
    pd.DataFrame(summary).to_csv("outputs/all_meters_results.csv", index=False)
    print("\n결과 저장: outputs/all_meters_results.csv")


if __name__ == "__main__":
    main()
