"""IF + LSTM-AE 하이브리드 이상탐지 (논문 7 근거).

논문: Wei Wei (2025) "Research on building energy consumption anomaly detection
     based on data mining technology" [IPMLP]

방법:
  1. LSTM Autoencoder: 재구성 오차(MSE) 계산
  2. Isolation Forest: 밀도 기반 이상 판별
  3. 앙상블: 두 모델 동시 탐지 → HIGH, 한 개만 → LOW, 없음 → NORMAL

임계치 (팀 통일):
  AE: mean(train_MSE) + 3 * std(train_MSE)  (MSD 방식)
  IF: contamination=0.05

평가 레이블 (pseudo-label):
  게이트웨이 장애 구간을 이상 구간으로 간주
  - val  (2021): 2021-11-15 ~ 2021-12-10
  - test (2022~2023): 2022-05-06 ~ 2022-07-14

MLflow 실험: LSTM-AE-Anomaly

RunPod 실행:
    pip install -r requirements_runpod.txt
    python train_anomaly_ifae.py

로컬 실행:
    uv run --with torch --with pandas --with "psycopg[binary]" \\
           --with python-dotenv --with scikit-learn --with mlflow \\
           python scripts/ml/train_anomaly_ifae.py
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

from project.ML.data_loader import FEATURE_COLS, add_features, get_splits, load_raw

load_dotenv()

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
SEQ_LEN        = 24
BATCH_SIZE     = 1024
EPOCHS         = 100
LR             = 1e-3
HIDDEN_DIM     = 128
LATENT_DIM     = 32
DROPOUT        = 0.2
IF_CONTAM      = 0.15   # val 기준 실제 이상 비율 ~19% 에 맞게 조정
EARLY_STOP_PAT = 15
AE_THRESHOLD_K = 2.0    # mean + k*std (3σ → 2σ 로 민감도 향상)
DEVICE     = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"

torch.backends.cudnn.benchmark = True
USE_AMP = DEVICE == "cuda"
DTYPE   = torch.bfloat16 if (USE_AMP and torch.cuda.is_bf16_supported()) else torch.float16

# 게이트웨이 장애 구간 (pseudo-label)
# train(2018~2021) 내 장애 3개는 data_loader에서 학습 제외 처리
# val(2022) 내 장애 1개만 평가 레이블로 사용
# test(2023)에는 알려진 장애 없음 → 레이블 전부 0 (탐지 없으면 정상 기대)
ANOMALY_PERIODS = [
    ("2022-05-06", "2022-07-14"),   # Workshop gateway #2 — val 구간 내
]

OUT_DIR = Path("outputs/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── LSTM Autoencoder ──────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len

        # Encoder: 2-layer LSTM + attention pooling → latent
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.enc_attn     = nn.Linear(hidden_dim, 1)
        self.enc_fc       = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent → 2-layer LSTM → 시계열 재구성
        self.dec_fc       = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.out_fc       = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder_lstm(x)                           # (B, T, H)
        w = torch.softmax(self.enc_attn(out), dim=1)            # (B, T, 1)
        return self.enc_fc((out * w).sum(dim=1))                # (B, latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).unsqueeze(1).expand(-1, self.seq_len, -1)
        out, _ = self.decoder_lstm(h)
        return self.out_fc(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def make_sequences(X: np.ndarray, seq_len: int):
    return np.array(
        [X[i: i + seq_len] for i in range(len(X) - seq_len)],
        dtype=np.float32,
    )


def reconstruction_mse(model: LSTMAutoencoder, loader) -> np.ndarray:
    """배치별 재구성 MSE 반환 (sample당 1개 스칼라)."""
    model.eval()
    errors = []
    with torch.no_grad():
        for (Xb,) in loader:
            Xb = Xb.to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=USE_AMP):
                recon = model(Xb)
            mse = ((Xb - recon.float()) ** 2).mean(dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def pseudo_labels(timestamps: np.ndarray) -> np.ndarray:
    """게이트웨이 장애 구간을 1, 나머지를 0으로 레이블링.

    timestamps가 pandas.Timestamp, datetime.date 또는 numpy.datetime64 혼합일 수 있으므로
    pandas로 통일해 비교합니다.
    """
    labels = np.zeros(len(timestamps), dtype=int)
    # 날짜 단위로 비교하면 tz-aware / tz-naive 불일치 문제를 피할 수 있습니다.
    ts_dates = pd.to_datetime(timestamps).date
    for start, end in ANOMALY_PERIODS:
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        mask = (ts_dates >= start_date) & (ts_dates <= end_date)
        labels[mask] = 1
    return labels


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"  [{name}] Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  "
          f"F1={f1:.3f}  AUC={auc:.3f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mlflow.set_tracking_uri(str(Path(__file__).resolve().parents[2] / "mlruns"))
    mlflow.set_experiment("LSTM-AE-Anomaly")

    print(f"▶ 디바이스: {DEVICE}")
    print("▶ 데이터 로드 중...")
    df = load_raw()
    df = add_features(df)
    train, val, test = get_splits(df)

    print(f"  학습  : {train.index[0].date()} ~ {train.index[-1].date()} ({len(train):,}행)")
    print(f"  검증  : {val.index[0].date()} ~ {val.index[-1].date()} ({len(val):,}행)")
    print(f"  테스트: {test.index[0].date()} ~ {test.index[-1].date()} ({len(test):,}행)")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_val   = scaler.transform(val[FEATURE_COLS])
    X_test  = scaler.transform(test[FEATURE_COLS])

    X_train_s = make_sequences(X_train, SEQ_LEN)
    X_val_s   = make_sequences(X_val,   SEQ_LEN)
    X_test_s  = make_sequences(X_test,  SEQ_LEN)

    # 시퀀스의 마지막 타임스탬프를 대표 시각으로 사용
    val_ts  = val.index[SEQ_LEN:].to_numpy()
    test_ts = test.index[SEQ_LEN:].to_numpy()

    def make_loader(X, shuffle=False):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            pin_memory=(DEVICE == "cuda"), num_workers=4, persistent_workers=True,
        )

    train_loader = make_loader(X_train_s, shuffle=True)
    val_loader   = make_loader(X_val_s)
    test_loader  = make_loader(X_test_s)

    # ── LSTM-AE 학습 ─────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        input_dim=len(FEATURE_COLS), hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM, seq_len=SEQ_LEN, dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    amp_scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)

    with mlflow.start_run(run_name="IF+LSTM-AE"):
        mlflow.log_params({
            "model":          "IF+LSTM-AE",
            "seq_len":        SEQ_LEN,
            "hidden_dim":     HIDDEN_DIM,
            "latent_dim":     LATENT_DIM,
            "epochs":         EPOCHS,
            "lr":             LR,
            "if_contamination": IF_CONTAM,
            "ae_threshold_k": AE_THRESHOLD_K,
            "threshold":      f"MSD (mean+{AE_THRESHOLD_K}std)",
            "n_features":     len(FEATURE_COLS),
            "scaler":         "MinMaxScaler",
            "use_amp":        USE_AMP,
        })

        print(f"▶ LSTM-AE 학습 중 (epochs={EPOCHS}, early_stop={EARLY_STOP_PAT})...")
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            t_losses = []
            for (Xb,) in train_loader:
                Xb = Xb.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=USE_AMP):
                    loss = criterion(model(Xb), Xb)
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                t_losses.append(loss.item())

            model.eval()
            v_losses = []
            with torch.no_grad():
                for (Xb,) in val_loader:
                    Xb = Xb.to(DEVICE)
                    with torch.amp.autocast(DEVICE, dtype=DTYPE, enabled=USE_AMP):
                        v_losses.append(criterion(model(Xb), Xb).item())

            t_loss = float(np.mean(t_losses))
            v_loss = float(np.mean(v_losses))
            mlflow.log_metrics({"train_recon_loss": t_loss, "val_recon_loss": v_loss}, step=epoch)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | train={t_loss:.6f} | val={v_loss:.6f}"
                      f" | no_improve={no_improve}")

            if no_improve >= EARLY_STOP_PAT:
                print(f"  Early stopping at epoch {epoch}")
                mlflow.log_metric("early_stop_epoch", epoch)
                break

        model.load_state_dict(best_state)

        # ── MSD 임계치 산출 (train 재구성 오차 기준) ─────────────────────────
        train_errors = reconstruction_mse(model, train_loader)
        threshold    = train_errors.mean() + AE_THRESHOLD_K * train_errors.std()
        print(f"\n▶ AE 임계치 (MSD k={AE_THRESHOLD_K}): {threshold:.6f}")
        print(f"  train MSE  mean={train_errors.mean():.6f}  std={train_errors.std():.6f}")
        mlflow.log_metric("ae_threshold_msd", float(threshold))

        # ── Isolation Forest 학습 ─────────────────────────────────────────────
        print("▶ Isolation Forest 학습 중...")
        iso = IsolationForest(contamination=IF_CONTAM, random_state=42, n_jobs=-1)
        iso.fit(X_train)     # 시퀀스 아닌 원본 피처 사용

        # ── 이상탐지 및 평가 ─────────────────────────────────────────────────
        for split_name, loader, X_flat, ts_arr in [
            ("val",  val_loader,  X_val,  val_ts),
            ("test", test_loader, X_test, test_ts),
        ]:
            # AE 재구성 오차
            ae_errors = reconstruction_mse(model, loader)
            ae_flag   = (ae_errors > threshold).astype(int)

            # IF 예측 (-1=이상, 1=정상) → 시퀀스 마지막 타임스텝 기준
            n_seq = len(X_flat) - SEQ_LEN
            if_pred = iso.predict(X_flat[SEQ_LEN:SEQ_LEN + n_seq])
            if_flag = (if_pred == -1).astype(int)

            # 앙상블 (0=NORMAL, 1=LOW, 2=HIGH)
            vote = ae_flag + if_flag
            y_pred_binary = (vote >= 1).astype(int)

            y_true = pseudo_labels(ts_arr[:n_seq])

            print(f"\n▶ {split_name} 이상탐지 결과")
            print(f"  전체 {n_seq:,}개 | 이상 pseudo-label: {y_true.sum():,}개 "
                  f"({y_true.mean() * 100:.1f}%)")
            print(f"  탐지: HIGH={( vote==2).sum():,}  LOW={(vote==1).sum():,}  "
                  f"NORMAL={(vote==0).sum():,}")

            m = print_metrics(split_name, y_true, y_pred_binary, vote)
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()})

        # ── 저장 ──────────────────────────────────────────────────────────────
        ae_path = OUT_DIR / "anomaly_lstmae.pt"
        torch.save({
            "model_state":  best_state,
            "model_config": {
                "input_dim":  len(FEATURE_COLS),
                "hidden_dim": HIDDEN_DIM,
                "latent_dim": LATENT_DIM,
                "seq_len":    SEQ_LEN,
            },
            "threshold": float(threshold),
        }, ae_path)

        if_path = OUT_DIR / "anomaly_iforest.pkl"
        with open(if_path, "wb") as f:
            pickle.dump(iso, f)

        scaler_path = OUT_DIR / "anomaly_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        mlflow.log_artifact(str(ae_path))
        mlflow.log_artifact(str(if_path))
        mlflow.log_artifact(str(scaler_path))

        print(f"\n▶ LSTM-AE 저장: {ae_path}")
        print(f"▶ Isolation Forest 저장: {if_path}")


if __name__ == "__main__":
    main()
