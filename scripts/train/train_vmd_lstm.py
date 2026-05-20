"""VMD-LSTM Grid 전기 소비량 예측 모델 (논문 3 근거).

논문: Yu Zhang et al. (2025) "Seasonally Adaptive VMD-SSA-LSTM:
      A Hybrid Deep Learning Framework for High-Accuracy District Heating Load Forecasting"

구현 전략:
  - VMD(K=4)로 grid_P 시계열을 IMF 4개로 분해
  - 각 IMF의 lag_1h를 추가 피처로 사용 (11개 → 15개)
  - MinMaxScaler, train 2018~2020 / val 2021 / test 2022~2023 (팀 통일 기준)
  - SSA 하이퍼파라미터 최적화는 고정 파라미터로 단순화

RunPod 실행:
    pip install -r requirements_runpod.txt
    python train_vmd_lstm.py

로컬 실행:
    uv run --with vmdpy --with torch --with pandas --with mlflow \
           --with "psycopg[binary]" --with python-dotenv --with scikit-learn \
           python scripts/ml/train_vmd_lstm.py
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from project.ML.data_loader import FEATURE_COLS, TARGET_COL, add_features, get_splits, load_raw

load_dotenv()

# ── VMD 파라미터 ─────────────────────────────────────────────────────────────
VMD_K     = 4
VMD_ALPHA = 2000
VMD_TAU   = 0
VMD_DC    = 0
VMD_INIT  = 1
VMD_TOL   = 1e-7

# ── LSTM 하이퍼파라미터 ──────────────────────────────────────────────────────
SEQ_LEN        = 24      # 1일 lookback (주별 패턴은 lag 피처로 직접 주입)
BATCH_SIZE     = 256
EPOCHS         = 100
LR             = 1e-3
HIDDEN_DIM     = 192
NUM_LAYERS     = 2
DROPOUT        = 0.2
EARLY_STOP_PAT = 15
DEVICE     = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
USE_AMP = False   # bfloat16 정밀도 손실이 [0,1] 스케일 loss에 영향
DTYPE   = torch.float16

OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 모델 ─────────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.w(x), dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)             # (B, H)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(model: nn.Module, loader, scaler_y: MinMaxScaler, y_seq: np.ndarray):
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=USE_AMP):
                preds.append(model(Xb.to(DEVICE)).float().cpu().numpy())
    y_pred_s = np.concatenate(preds)
    y_true = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).ravel()
    y_pred = np.maximum(scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel(), 0)
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mape(y_true, y_pred),
    )


# ── VMD 분해 ─────────────────────────────────────────────────────────────────

def apply_vmd(df, base_feature_cols: list[str]) -> tuple:
    """grid_P를 VMD로 분해하고 IMF lag_1h 피처를 추가."""
    from vmdpy import VMD

    print(f"▶ VMD 분해 중 (K={VMD_K}, alpha={VMD_ALPHA})...")
    signal = df[TARGET_COL].values.astype(float)
    u, _, _ = VMD(signal, VMD_ALPHA, VMD_TAU, VMD_K, VMD_DC, VMD_INIT, VMD_TOL)

    imf_cols = []
    for i in range(VMD_K):
        col = f"imf_{i + 1}_lag1"
        df[col] = u[i]
        df[col] = df[col].shift(1).fillna(0)
        imf_cols.append(col)
        print(f"  IMF {i + 1}: mean={u[i].mean():.1f}  std={u[i].std():.1f}")

    return df, base_feature_cols + imf_cols


def add_lag_features(df, feature_cols: list[str]) -> tuple:
    """주별 패턴 포착을 위한 long-lag 피처 추가 (1주·2주 전 grid_P)."""
    lag_cols = []
    for lag_h in [168, 336]:   # 1주 전, 2주 전
        col = f"grid_P_lag{lag_h}h"
        df[col] = df[TARGET_COL].shift(lag_h).fillna(0)
        lag_cols.append(col)
        print(f"  lag {lag_h}h: mean={df[col].mean():.1f}")
    return df, feature_cols + lag_cols


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[2] / "mlruns"))
    mlflow.set_experiment("SSA-IPSO-LSTM")

    print(f"▶ 디바이스: {DEVICE}")
    print("▶ DB에서 데이터 로드 중...")
    df = load_raw()
    df = add_features(df)

    df, feature_cols = apply_vmd(df, list(FEATURE_COLS))
    df, feature_cols = add_lag_features(df, feature_cols)
    print(f"▶ 피처 수: {len(feature_cols)}  (기본 {len(FEATURE_COLS)} + IMF lag {VMD_K} + weekly lag 2)")

    train, val, test = get_splits(df)
    print(f"  학습  : {train.index[0].date()} ~ {train.index[-1].date()} ({len(train):,}행)")
    print(f"  검증  : {val.index[0].date()} ~ {val.index[-1].date()} ({len(val):,}행)")
    print(f"  테스트: {test.index[0].date()} ~ {test.index[-1].date()} ({len(test):,}행)")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[feature_cols])
    y_train = scaler_y.fit_transform(train[[TARGET_COL]]).ravel()
    X_val   = scaler_X.transform(val[feature_cols])
    y_val   = scaler_y.transform(val[[TARGET_COL]]).ravel()
    X_test  = scaler_X.transform(test[feature_cols])
    y_test  = scaler_y.transform(test[[TARGET_COL]]).ravel()

    X_train_s, y_train_s = make_sequences(X_train, y_train, SEQ_LEN)
    X_val_s,   y_val_s   = make_sequences(X_val,   y_val,   SEQ_LEN)
    X_test_s,  y_test_s  = make_sequences(X_test,  y_test,  SEQ_LEN)

    def make_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            pin_memory=(DEVICE == "cuda"), num_workers=4, persistent_workers=True,
        )

    train_loader = make_loader(X_train_s, y_train_s, shuffle=True)
    val_loader   = make_loader(X_val_s,   y_val_s,   shuffle=False)
    test_loader  = make_loader(X_test_s,  y_test_s,  shuffle=False)

    model = LSTMForecaster(len(feature_cols), HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5)
    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_state    = None
    amp_scaler    = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)

    with mlflow.start_run(run_name="VMD-LSTM"):
        mlflow.log_params({
            "vmd_k": VMD_K, "vmd_alpha": VMD_ALPHA,
            "seq_len": SEQ_LEN, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "n_features": len(feature_cols), "scaler": "MinMaxScaler",
            "use_amp": USE_AMP, "attention": True,
        })

        print(f"▶ VMD-LSTM 학습 중 (epochs={EPOCHS}, early_stop={EARLY_STOP_PAT})...")
        no_improve = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            t_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=USE_AMP):
                    loss = criterion(model(Xb), yb)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                t_losses.append(loss.item())

            model.eval()
            v_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=USE_AMP):
                        v_losses.append(criterion(model(Xb), yb).item())

            t_loss = float(np.mean(t_losses))
            v_loss = float(np.mean(v_losses))
            scheduler.step(v_loss)
            mlflow.log_metrics({"train_loss": t_loss, "val_loss": v_loss}, step=epoch)

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

        for split_name, loader, y_seq in [
            ("val",  val_loader,  y_val_s),
            ("test", test_loader, y_test_s),
        ]:
            mae_v, rmse_v, mape_v = evaluate(model, loader, scaler_y, y_seq)
            print(f"\n▶ {split_name} 결과")
            print(f"  MAE  : {mae_v:.2f} W  |  RMSE : {rmse_v:.2f} W  |  MAPE : {mape_v:.2f}%")
            mlflow.log_metrics({
                f"{split_name}_mae":  round(mae_v, 4),
                f"{split_name}_rmse": round(rmse_v, 4),
                f"{split_name}_mape": round(mape_v, 4),
            })

        # 저장
        model_path = OUT_DIR / "vmd_lstm_grid_electricity.pt"
        torch.save({
            "model_state": best_state,
            "model_config": {
                "input_dim": len(feature_cols), "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS, "dropout": DROPOUT,
                "seq_len": SEQ_LEN, "vmd_k": VMD_K, "feature_cols": feature_cols,
            },
        }, model_path)

        scaler_path = OUT_DIR / "vmd_lstm_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))
        print(f"\n▶ 모델 저장: {model_path}")
        print(f"▶ 스케일러 저장: {scaler_path}")


if __name__ == "__main__":
    main()
