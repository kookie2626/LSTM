"""예측 및 이상탐지 결과 시각화.

사용법:
    python visualize.py --start 2022-01-01 --end 2022-12-31
    python visualize.py --start 2023-01-01 --end 2023-12-31 --mode forecast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import koreanize_matplotlib  # noqa: F401  — 한글 폰트 자동 등록
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from project.ML.data_loader import add_features, load_raw
from project.ML.inference import predict_anomaly, predict_forecast

load_dotenv()

OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "actual":    "#2c7bb6",
    "predicted": "#d7191c",
    "error":     "#fdae61",
    "high":      "#d73027",
    "low":       "#fee090",
    "normal":    "#e0f3f8",
}


# ── 예측 시각화 ───────────────────────────────────────────────────────────────

def plot_forecast(fc: pd.DataFrame, start: str, end: str,
                  sample_days: int = 14) -> Path:
    """실제값 vs 예측값 플롯 (전체 기간 + 샘플 2주 확대)."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 3, 2]})
    fig.suptitle(f"VMD-LSTM 전력 소비량 예측  ({start} ~ {end})",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── 전체 기간 ──
    ax = axes[0]
    ax.plot(fc.index, fc["actual"] / 1000,    color=PALETTE["actual"],
            lw=0.8, label="실제값", alpha=0.9)
    ax.plot(fc.index, fc["predicted"] / 1000, color=PALETTE["predicted"],
            lw=0.8, label="예측값", alpha=0.8)
    ax.set_ylabel("전력 (kW)")
    ax.set_title("전체 기간")
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # ── 샘플 확대 (첫 sample_days일) ──
    sample_end = fc.index[0] + pd.Timedelta(days=sample_days)
    zoom = fc[fc.index <= sample_end]
    ax2 = axes[1]
    ax2.plot(zoom.index, zoom["actual"] / 1000,    color=PALETTE["actual"],
             lw=1.2, label="실제값", marker="o", ms=2)
    ax2.plot(zoom.index, zoom["predicted"] / 1000, color=PALETTE["predicted"],
             lw=1.2, label="예측값", marker="s", ms=2)
    ax2.set_ylabel("전력 (kW)")
    ax2.set_title(f"확대: 처음 {sample_days}일")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # ── 오차 ──
    ax3 = axes[2]
    ax3.fill_between(fc.index, fc["error"] / 1000, 0,
                     where=(fc["error"] >= 0), color=PALETTE["actual"],
                     alpha=0.5, label="과소예측")
    ax3.fill_between(fc.index, fc["error"] / 1000, 0,
                     where=(fc["error"] < 0), color=PALETTE["predicted"],
                     alpha=0.5, label="과대예측")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_ylabel("오차 (kW)")
    ax3.set_title("예측 오차")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax3.grid(axis="y", alpha=0.3)

    # 지표 텍스트
    mae  = fc["error"].abs().mean()
    mape = fc["abs_pct_error"].mean()
    rmse = np.sqrt((fc["error"] ** 2).mean())
    fig.text(0.01, 0.01,
             f"MAE: {mae/1000:.1f} kW   RMSE: {rmse/1000:.1f} kW   MAPE: {mape:.1f}%",
             fontsize=10, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = OUT_DIR / f"forecast_{start}_{end}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 이상탐지 시각화 ───────────────────────────────────────────────────────────

def plot_anomaly(df_raw: pd.DataFrame, an: pd.DataFrame,
                 start: str, end: str) -> Path:
    """grid_P 시계열에 이상탐지 결과 오버레이."""
    target = df_raw[(df_raw.index >= an.index[0]) & (df_raw.index <= an.index[-1])]

    fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                             gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.suptitle(f"이상탐지 결과  ({start} ~ {end})",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── 전력 시계열 + 이상 구간 배경 ──
    ax = axes[0]
    ax.plot(target.index, target["grid_P"] / 1000,
            color=PALETTE["actual"], lw=0.7, label="grid_P", zorder=3)

    high_mask = an["anomaly_level"] == "HIGH"
    low_mask  = an["anomaly_level"] == "LOW"
    for ts in an.index[high_mask]:
        ax.axvspan(ts, ts + pd.Timedelta(hours=1),
                   color=PALETTE["high"], alpha=0.4, lw=0)
    for ts in an.index[low_mask]:
        ax.axvspan(ts, ts + pd.Timedelta(hours=1),
                   color=PALETTE["low"], alpha=0.5, lw=0)

    patches = [
        mpatches.Patch(color=PALETTE["high"], alpha=0.6, label="HIGH 이상"),
        mpatches.Patch(color=PALETTE["low"],  alpha=0.6, label="LOW 이상"),
        mpatches.Patch(color=PALETTE["actual"], label="grid_P"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.set_ylabel("전력 (kW)")
    ax.set_title("전력 소비량 + 이상 구간")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # ── 예측 잔차 ──
    ax2 = axes[1]
    ax2.plot(an.index, an["residual"] / 1000, color="#4575b4", lw=0.7, label="예측 잔차")
    threshold_val = an["residual"][an["res_flag"] == 1].min() if an["res_flag"].any() else None
    if threshold_val:
        ax2.axhline(threshold_val / 1000, color=PALETTE["high"], lw=1.2,
                    ls="--", label=f"임계치 ≈ {threshold_val/1000:.1f} kW")
    ax2.set_ylabel("잔차 (kW)")
    ax2.set_title("VMD-LSTM 예측 잔차 (|실제 - 예측|)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # ── 앙상블 vote 히트맵 ──
    ax3 = axes[2]
    colors = an["vote"].map({0: PALETTE["normal"], 1: PALETTE["low"], 2: PALETTE["high"]})
    ax3.bar(an.index, 1, width=pd.Timedelta(hours=1),
            color=colors.values, align="edge")
    ax3.set_yticks([])
    ax3.set_title("앙상블 판정 (NORMAL / LOW / HIGH)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    n_high = high_mask.sum()
    n_low  = low_mask.sum()
    total  = len(an)
    fig.text(0.01, 0.01,
             f"HIGH: {n_high:,} ({n_high/total*100:.1f}%)   "
             f"LOW: {n_low:,} ({n_low/total*100:.1f}%)   "
             f"NORMAL: {total-n_high-n_low:,} ({(total-n_high-n_low)/total*100:.1f}%)",
             fontsize=10, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = OUT_DIR / f"anomaly_{start}_{end}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EMS 결과 시각화")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end",   required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--mode",  choices=["forecast", "anomaly", "both"],
                        default="both")
    args = parser.parse_args()

    print("▶ 데이터 로드 중...")
    df = load_raw()
    df = add_features(df)

    if args.mode in ("forecast", "both"):
        print("▶ 예측 실행 중...")
        fc = predict_forecast(df, args.start, args.end)
        path = plot_forecast(fc, args.start, args.end)
        print(f"  저장: {path}")

    if args.mode in ("anomaly", "both"):
        print("▶ 이상탐지 실행 중...")
        an = predict_anomaly(df, args.start, args.end)
        path = plot_anomaly(df, an, args.start, args.end)
        print(f"  저장: {path}")


if __name__ == "__main__":
    main()
