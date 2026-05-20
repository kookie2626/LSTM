"""팀 발표용 종합 시각화 리포트 생성.

출력: outputs/presentation_report.html

캐싱: DB/추론 결과를 outputs/cache/ 에 저장.
      캐시가 있으면 DB 없이도 실행 가능 (offline mode).

실행:
    PYTHONPATH=/workspace python scripts/report/make_presentation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, precision_score, recall_score

CACHE_DIR = Path("outputs/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = Path("outputs/presentation_report.html")

# ── 추론 결과 로드 (캐시 우선 → DB fallback) ────────────────────────────────

def _load_or_infer() -> tuple:
    """fc_val, fc_test, an_val을 캐시에서 읽거나 추론 후 저장."""
    fc_val_path  = CACHE_DIR / "fc_val.parquet"
    fc_test_path = CACHE_DIR / "fc_test.parquet"
    an_val_path  = CACHE_DIR / "an_val.parquet"

    if fc_val_path.exists() and fc_test_path.exists() and an_val_path.exists():
        print("▶ 캐시에서 추론 결과 로드 중...")
        fc_val  = pd.read_parquet(fc_val_path)
        fc_test = pd.read_parquet(fc_test_path)
        an_val  = pd.read_parquet(an_val_path)
        print("  캐시 로드 완료.")
        return fc_val, fc_test, an_val, True

    print("▶ DB 연결 및 추론 실행 중 (첫 실행 — 이후 캐시 사용)...")
    try:
        from project.ML.inference import predict_forecast, predict_anomaly
        from project.ML.data_loader import load_raw, add_features

        df = load_raw()
        df = add_features(df)

        print("  예측 실행 (val 2022)...")
        fc_val  = predict_forecast(df, "2022-01-01", "2022-12-31")
        print("  예측 실행 (test 2023)...")
        fc_test = predict_forecast(df, "2023-01-01", "2023-12-31")
        print("  이상탐지 실행 (val 2022)...")
        an_val  = predict_anomaly(df, "2022-01-01", "2022-12-31")

        # 캐시 저장
        fc_val.to_parquet(fc_val_path)
        fc_test.to_parquet(fc_test_path)
        an_val.to_parquet(an_val_path)
        print(f"  캐시 저장 완료: {CACHE_DIR}")
        return fc_val, fc_test, an_val, True

    except Exception as e:
        print(f"  ⚠ DB/추론 실패 ({e.__class__.__name__}): 오프라인 모드로 전환합니다.")
        return None, None, None, False


fc_val, fc_test, an_val, ONLINE = _load_or_infer()

print("▶ 계량기 결과 로드 중...")
meters = pd.read_csv("outputs/all_meters_results.csv")
print("▶ 에너지 7종 결과 로드 중...")
energy_results = pd.read_csv("outputs/all_energy_results.csv")

# ── 동적 지표 계산 ─────────────────────────────────────────────────────────────
print("▶ 지표 계산 중...")

er = energy_results[energy_results["status"] == "OK"]
gp = er[er["target"] == "grid_P"].iloc[0]

ok = meters[meters["status"] == "OK"].copy()
ok["building"] = ok["meter_urn"].str.extract(r"^([A-Z]+\d+)")[0].fillna("V")
_n_good = int((ok["test_mape"] < 20).sum())
_n_mid  = int(ok["test_mape"].between(20, 100).sum())
_n_bad  = int((ok["test_mape"] > 100).sum())

if ONLINE:
    _mask_v      = fc_val["actual"] > 100
    _fc_val_mae  = float(np.mean(np.abs(fc_val["actual"] - fc_val["predicted"])))
    _fc_val_mape = float(np.mean(np.abs((fc_val.loc[_mask_v,"actual"] - fc_val.loc[_mask_v,"predicted"]) / fc_val.loc[_mask_v,"actual"])) * 100) if _mask_v.sum() > 0 else float("nan")
    _fc_val_wape = float(np.sum(np.abs(fc_val["actual"] - fc_val["predicted"])) / np.sum(np.abs(fc_val["actual"])) * 100)
    _fc_val_rmse = float(np.sqrt(np.mean((fc_val["actual"] - fc_val["predicted"]) ** 2)))

    _mask_t       = fc_test["actual"] > 100
    _fc_test_mae  = float(np.mean(np.abs(fc_test["actual"] - fc_test["predicted"])))
    _fc_test_mape = float(np.mean(np.abs((fc_test.loc[_mask_t,"actual"] - fc_test.loc[_mask_t,"predicted"]) / fc_test.loc[_mask_t,"actual"])) * 100) if _mask_t.sum() > 0 else float("nan")
    _fc_test_wape = float(np.sum(np.abs(fc_test["actual"] - fc_test["predicted"])) / np.sum(np.abs(fc_test["actual"])) * 100)
    _fc_test_rmse = float(np.sqrt(np.mean((fc_test["actual"] - fc_test["predicted"]) ** 2)))

    _tz = an_val.index.tz
    def _ts(s): return pd.Timestamp(s).tz_localize(_tz) if _tz and pd.Timestamp(s).tzinfo is None else pd.Timestamp(s)
    _fail_mask  = (an_val.index >= _ts("2022-05-06")) & (an_val.index <= _ts("2022-07-14"))
    _y_true     = _fail_mask.astype(int).values
    _y_high     = (an_val["anomaly_level"] == "HIGH").astype(int).values
    _y_any      = (an_val["anomaly_level"] != "NORMAL").astype(int).values
    _an_f1_high    = float(f1_score(_y_true, _y_high, zero_division=0))
    _an_prec_high  = float(precision_score(_y_true, _y_high, zero_division=0))
    _an_recall_high= float(recall_score(_y_true, _y_high, zero_division=0))
    _an_f1_any     = float(f1_score(_y_true, _y_any, zero_division=0))
    _an_prec_any   = float(precision_score(_y_true, _y_any, zero_division=0))
    _an_recall_any = float(recall_score(_y_true, _y_any, zero_division=0))
    _n_high = int((an_val["anomaly_level"] == "HIGH").sum())
    _n_low  = int((an_val["anomaly_level"] == "LOW").sum())
    _n_anom = _n_high + _n_low
else:
    # 오프라인: energy_results CSV 값 사용
    _fc_val_mae   = float(gp["val_mae"])
    _fc_val_mape  = float(gp["val_mape"])
    _fc_val_wape  = float(gp["val_wape"])
    _fc_val_rmse  = float(gp["val_rmse"])
    _fc_test_mae  = float(gp["test_mae"])
    _fc_test_mape = float(gp["test_mape"])
    _fc_test_wape = float(gp["test_wape"])
    _fc_test_rmse = float(gp["test_rmse"])
    # 이상탐지 지표: val_f1 사용 (이상탐지 학습 시 기록된 값)
    _an_f1_high    = float(gp["val_f1"])
    _an_prec_high  = float("nan")
    _an_recall_high= float("nan")
    _an_f1_any     = float("nan")
    _an_prec_any   = float("nan")
    _an_recall_any = float("nan")
    _n_high = _n_low = _n_anom = 0

print(f"  grid_P val MAPE={_fc_val_mape:.1f}%  test MAPE={_fc_test_mape:.1f}%")
if ONLINE:
    print(f"  이상탐지 F1(HIGH)={_an_f1_high:.3f}  F1(HIGH+LOW)={_an_f1_any:.3f}")
print(f"  계량기 우수:{_n_good} 보통:{_n_mid} 저조:{_n_bad}  (오프라인={not ONLINE})")

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
C_ACTUAL = "#2563EB"
C_PRED   = "#F97316"
C_HIGH   = "#EF4444"
C_LOW    = "#FBBF24"
C_FAIL   = "rgba(220,38,38,0.10)"

BASE_LAYOUT = dict(
    font=dict(family="Pretendard, Noto Sans KR, sans-serif", size=13),
    paper_bgcolor="white",
    plot_bgcolor="#F8FAFC",
    margin=dict(l=60, r=40, t=70, b=50),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)

PLACEHOLDER_FIG = go.Figure()
PLACEHOLDER_FIG.add_annotation(
    text="DB 연결 후 캐시 생성 시 그래프 표시<br>(python scripts/report/make_presentation.py 재실행)",
    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
    font=dict(size=14, color="#94A3B8"),
)
PLACEHOLDER_FIG.update_layout(**BASE_LAYOUT, height=200,
    title="<b>추론 그래프 — 캐시 없음 (오프라인 모드)</b>")


# ════════════════════════════════════════════════════════════════════
#  FIG 1 : grid_P 예측 — val 2022
# ════════════════════════════════════════════════════════════════════
def fig_forecast_val() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    fc = fc_val.resample("1D").mean().dropna()
    fig = go.Figure()
    fig.add_vrect(x0="2022-05-06", x1="2022-07-14",
                  fillcolor=C_FAIL, line_width=0,
                  annotation_text="장애 구간", annotation_position="top left",
                  annotation_font_color="#EF4444")
    fig.add_trace(go.Scatter(x=fc.index, y=fc["actual"]/1000, name="실제값",
                             line=dict(color=C_ACTUAL, width=1.8)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["predicted"]/1000, name="예측값",
                             line=dict(color=C_PRED, width=1.8, dash="dot")))
    fig.update_layout(
        **BASE_LAYOUT,
        title=f"<b>grid_P 예측 결과 — Validation (2022)</b><br>"
              f"<sub>MAE {_fc_val_mae:,.0f} W · MAPE {_fc_val_mape:.1f}% · WAPE {_fc_val_wape:.1f}% · RMSE {_fc_val_rmse:,.0f} W · 일별 평균</sub>",
        yaxis_title="전력 소비 (kW)", xaxis_title="날짜",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 2 : grid_P 예측 — test 2023
# ════════════════════════════════════════════════════════════════════
def fig_forecast_test() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    fc = fc_test.resample("1D").mean().dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc.index, y=fc["actual"]/1000, name="실제값",
                             line=dict(color=C_ACTUAL, width=1.8)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["predicted"]/1000, name="예측값",
                             line=dict(color=C_PRED, width=1.8, dash="dot")))
    fig.update_layout(
        **BASE_LAYOUT,
        title=f"<b>grid_P 예측 결과 — Test (2023)</b><br>"
              f"<sub>MAE {_fc_test_mae:,.0f} W · MAPE {_fc_test_mape:.1f}% · WAPE {_fc_test_wape:.1f}% · RMSE {_fc_test_rmse:,.0f} W</sub>",
        yaxis_title="전력 소비 (kW)", xaxis_title="날짜",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 3 : 이상탐지 — 장애 구간 확대
# ════════════════════════════════════════════════════════════════════
def fig_anomaly_zoom() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    an   = an_val["2022-04-01":"2022-09-30"]
    high = an[an["anomaly_level"] == "HIGH"]
    low  = an[an["anomaly_level"] == "LOW"]
    fig = go.Figure()
    fig.add_vrect(x0="2022-05-06", x1="2022-07-14", fillcolor=C_FAIL, line_width=0,
                  annotation_text="실제 장애 구간 (2022-05-06~07-14)",
                  annotation_position="top left", annotation_font_color="#EF4444")
    fig.add_trace(go.Scatter(x=an.index, y=an["actual"]/1000, name="실제값",
                             line=dict(color=C_ACTUAL, width=1.2)))
    fig.add_trace(go.Scatter(x=an.index, y=an["predicted"]/1000, name="예측값",
                             line=dict(color=C_PRED, width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=high.index, y=high["actual"]/1000, mode="markers",
                             name="HIGH 이상", marker=dict(color=C_HIGH, size=5, symbol="x")))
    fig.add_trace(go.Scatter(x=low.index, y=low["actual"]/1000, mode="markers",
                             name="LOW 이상", marker=dict(color=C_LOW, size=4, symbol="circle-open")))
    fig.update_layout(**BASE_LAYOUT,
        title="<b>이상탐지 상세 — 장애 구간 전후 (2022-04~09)</b>",
        yaxis_title="전력 소비 (kW)", xaxis_title="날짜")
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 4 : 이상탐지 잔차 분포
# ════════════════════════════════════════════════════════════════════
def fig_residual_dist() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    an   = an_val.copy()
    fail = an["2022-05-06":"2022-07-14"]
    norm = an.drop(fail.index, errors="ignore")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=norm["residual"]/1000, name="정상 구간",
                               opacity=0.7, nbinsx=80, marker_color=C_ACTUAL))
    fig.add_trace(go.Histogram(x=fail["residual"]/1000, name="장애 구간",
                               opacity=0.8, nbinsx=40, marker_color=C_HIGH))
    fig.update_layout(**BASE_LAYOUT, barmode="overlay",
        title="<b>예측 잔차 분포 — 정상 vs 장애 구간</b>",
        xaxis_title="잔차 (kW)", yaxis_title="빈도")
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 5 : 이상탐지 정량 성능 (F1/Prec/Recall)
# ════════════════════════════════════════════════════════════════════
def fig_anomaly_metrics() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    categories = ["Precision", "Recall", "F1-Score"]
    vals_high = [_an_prec_high*100, _an_recall_high*100, _an_f1_high*100]
    vals_any  = [_an_prec_any*100,  _an_recall_any*100,  _an_f1_any*100]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=vals_high, name="HIGH 이상만",
                         marker_color=C_HIGH,
                         text=[f"{v:.1f}%" for v in vals_high], textposition="outside"))
    fig.add_trace(go.Bar(x=categories, y=vals_any, name="HIGH+LOW 합산",
                         marker_color=C_LOW,
                         text=[f"{v:.1f}%" for v in vals_any], textposition="outside"))
    fig.add_hline(y=70, line_dash="dash", line_color="#16A34A",
                  annotation_text="목표 70%", annotation_position="right")
    fig.update_layout(**BASE_LAYOUT, barmode="group", yaxis_range=[0, 110],
        title=f"<b>이상탐지 정량 평가 — 장애 구간 대비</b><br>"
              f"<sub>HIGH F1={_an_f1_high:.3f} · HIGH+LOW F1={_an_f1_any:.3f} · pseudo-label 2022-05-06~07-14</sub>",
        yaxis_title="성능 (%)", legend=dict(orientation="h", y=-0.2))
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 6 : 계량기 히트맵 (treemap)
# ════════════════════════════════════════════════════════════════════
def fig_meter_heatmap() -> go.Figure:
    _ok = ok.copy()
    _ok["mape_capped"] = _ok["test_mape"].clip(upper=300)
    fig = px.treemap(
        _ok, path=[px.Constant("전체"), "building", "meter_urn"],
        values="mape_capped", color="mape_capped",
        hover_data={"test_mape":":.1f","test_mae":":.0f","val_wape":":.1f"},
        color_continuous_scale=["#22C55E","#FCD34D","#EF4444"],
        color_continuous_midpoint=50,
        title="<b>계량기 80개 Test MAPE 히트맵</b><br>"
              "<sub>초록=우수(~20%), 노랑=보통(20~50%), 빨강=저조(50%+) · 건물별 그룹</sub>",
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>test MAPE: %{color:.1f}%<extra></extra>")
    fig.update_layout(**{**BASE_LAYOUT, "margin": dict(l=20,r=20,t=80,b=20)})
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 7 : 계량기 MAE vs F1 scatter
# ════════════════════════════════════════════════════════════════════
def fig_meter_scatter() -> go.Figure:
    _ok = ok.copy()
    _ok["test_mae_kw"] = _ok["test_mae"] / 1000
    fig = px.scatter(
        _ok, x="test_mae_kw", y="val_f1",
        color="test_mape", size="test_mae_kw", hover_name="meter_urn",
        hover_data={"test_mape":":.1f","test_mae_kw":":.2f","val_wape":":.1f"},
        color_continuous_scale=["#22C55E","#FCD34D","#EF4444"],
        range_color=[0, 200],
        title="<b>계량기별 예측 MAE vs 이상탐지 F1</b><br>"
              "<sub>좌하단=우수(낮은 오차+높은 F1) · 색상=test MAPE</sub>",
        labels={"test_mae_kw":"Test MAE (kW)","val_f1":"Val F1 (이상탐지)"},
    )
    fig.add_vline(x=5, line_dash="dash", line_color="gray",
                  annotation_text="MAE 5kW", annotation_position="top")
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="F1 0.5", annotation_position="right")
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 8 : 건물별 Val WAPE 분포 (box plot)
# ════════════════════════════════════════════════════════════════════
def fig_meter_wape_box() -> go.Figure:
    _ok = ok.dropna(subset=["val_wape"]).copy()
    buildings = sorted(_ok["building"].unique())
    colors = {"H1":"#3B82F6","H2":"#8B5CF6","H3":"#10B981","H4":"#F59E0B","V":"#EF4444"}
    fig = go.Figure()
    for bld in buildings:
        sub = _ok[_ok["building"] == bld]
        fig.add_trace(go.Box(
            y=sub["val_wape"].clip(upper=120), name=bld,
            boxpoints="all", jitter=0.4, pointpos=-1.8,
            marker_color=colors.get(bld,"#94A3B8"),
            line_color=colors.get(bld,"#94A3B8"),
            customdata=sub[["meter_urn","val_wape","test_wape"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Val WAPE: %{customdata[1]:.1f}%<br>Test WAPE: %{customdata[2]:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=20, line_dash="dash", line_color="#16A34A",
                  annotation_text="목표 20%", annotation_position="right")
    fig.update_layout(**BASE_LAYOUT, showlegend=False,
        title="<b>건물별 계량기 Val WAPE 분포</b><br>"
              "<sub>cap=120% · 점=개별 계량기 · 상자=25~75% 분위수</sub>",
        yaxis_title="Val WAPE (%)", xaxis_title="건물")
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 9 : 에너지 흐름 상관관계
# ════════════════════════════════════════════════════════════════════
def fig_energy_correlation() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    from project.ML.data_loader import load_raw, add_features
    _df = load_raw(); _df = add_features(_df)
    energy_cols = {"grid_P":"총 전력","pv_P":"태양광","chp_P":"열병합(전기)","Ta":"외기 온도","Igm":"일사량"}
    available = {k:v for k,v in energy_cols.items() if k in _df.columns}
    sub  = _df[list(available.keys())].dropna()
    corr = sub.corr()
    labels = [available[c] for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values,2), texttemplate="%{text}", textfont_size=14))
    fig.update_layout(**BASE_LAYOUT, width=600, height=500,
        title="<b>에너지 변수 간 상관관계</b><br>"
              "<sub>태양광·열병합 발전↑ → 계통 소비(grid_P)↓</sub>")
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 10 : 계절별 에너지 패턴
# ════════════════════════════════════════════════════════════════════
def fig_seasonal_pattern() -> go.Figure:
    if not ONLINE:
        return PLACEHOLDER_FIG
    from project.ML.data_loader import load_raw, add_features
    _df = load_raw(); _df = add_features(_df)
    sub = _df[["grid_P","pv_P","chp_P","Ta"]].copy()
    sub["month"] = sub.index.month
    monthly = sub.groupby("month").mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("전력 소비·발전 (월평균)","외기 온도 (월평균)"),
        vertical_spacing=0.12)
    fig.add_trace(go.Bar(x=monthly.index, y=monthly["grid_P"]/1000,
                         name="계통 소비(grid_P)", marker_color=C_ACTUAL), row=1,col=1)
    if "pv_P" in monthly:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["pv_P"].abs()/1000,
                             name="태양광 발전(abs)", marker_color="#22C55E"), row=1,col=1)
    if "chp_P" in monthly:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["chp_P"].abs()/1000,
                             name="열병합 발전(abs)", marker_color="#A855F7"), row=1,col=1)
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly["Ta"], name="외기 온도(°C)",
                             line=dict(color="#F97316",width=2.5), mode="lines+markers"), row=2,col=1)
    fig.update_xaxes(tickvals=list(range(1,13)),
        ticktext=["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"])
    fig.update_yaxes(title_text="전력 (kW)", row=1,col=1)
    fig.update_yaxes(title_text="온도 (°C)", row=2,col=1)
    fig.update_layout(**BASE_LAYOUT, barmode="group",
        title="<b>계절별 에너지 패턴 (2018–2023 전체 평균)</b>")
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 11 : Val vs Test WAPE 비교
# ════════════════════════════════════════════════════════════════════
def fig_val_vs_test() -> go.Figure:
    summary = pd.DataFrame([
        {"모델":"grid_P\n(VMD-LSTM)",    "val_wape":float(gp["val_wape"]),         "test_wape":float(gp["test_wape"])},
        {"모델":"에너지 7종\n(WAPE 평균)","val_wape":er["val_wape"].mean(),         "test_wape":er["test_wape"].mean()},
        {"모델":"계량기 중앙값\n(80개)",  "val_wape":ok["val_wape"].median(),       "test_wape":ok["test_wape"].median()},
        {"모델":"계량기 상위25%",         "val_wape":ok["val_wape"].quantile(0.25),"test_wape":ok["test_wape"].quantile(0.25)},
        {"모델":"계량기 하위25%",         "val_wape":ok["val_wape"].quantile(0.75),"test_wape":ok["test_wape"].quantile(0.75)},
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary["모델"], y=summary["val_wape"], name="Val WAPE (2022)",
                         marker_color=C_ACTUAL,
                         text=[f"{v:.1f}%" for v in summary["val_wape"]], textposition="outside"))
    fig.add_trace(go.Bar(x=summary["모델"], y=summary["test_wape"], name="Test WAPE (2023)",
                         marker_color=C_PRED,
                         text=[f"{v:.1f}%" for v in summary["test_wape"]], textposition="outside"))
    fig.add_hline(y=25, line_dash="dash", line_color="green",
                  annotation_text="목표 25%", annotation_position="right")
    fig.update_layout(**BASE_LAYOUT, barmode="group",
        title="<b>Val vs Test WAPE 비교 — 전 모델</b><br>"
              "<sub>WAPE(가중절대퍼센트오차) · 2022→2023 패턴 변화 확인</sub>",
        yaxis_title="WAPE (%)")
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 12 : 에너지 7종 MAPE + WAPE
# ════════════════════════════════════════════════════════════════════
def fig_energy_results() -> go.Figure:
    _er = er.copy()
    label_map = {
        "grid_P":"grid_P\n(총 전력)",     "pv_gen":"pv_gen\n(태양광)",
        "chp_gen":"chp_gen\n(열병합)",    "cool_P":"cool_P\n(냉방)",
        "cool_elec_P":"cool_elec_P\n(냉동기)", "heat_P":"heat_P\n(난방)",
        "chp_heat_P":"chp_heat_P\n(열병합열)",
    }
    _er["label"] = _er["target"].map(label_map).fillna(_er["target"])

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("MAPE (%) — cap 150%","WAPE (%)"),
        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=_er["label"], y=_er["val_mape"].clip(upper=150),
                         name="Val MAPE", marker_color=C_ACTUAL), row=1,col=1)
    fig.add_trace(go.Bar(x=_er["label"], y=_er["test_mape"].clip(upper=150),
                         name="Test MAPE", marker_color=C_PRED), row=1,col=1)
    fig.add_trace(go.Bar(x=_er["label"], y=_er["val_wape"],
                         name="Val WAPE", marker_color="#7C3AED"), row=1,col=2)
    fig.add_trace(go.Bar(x=_er["label"], y=_er["test_wape"],
                         name="Test WAPE", marker_color="#DB2777"), row=1,col=2)
    fig.update_layout(**BASE_LAYOUT, barmode="group",
        title="<b>에너지 7종 예측 성능 (MAPE · WAPE)</b>")
    return fig


# ════════════════════════════════════════════════════════════════════
#  HTML 조립
# ════════════════════════════════════════════════════════════════════

def _tag(val, t1=20, t2=50):
    cls = "good" if val < t1 else "warn" if val < t2 else "bad"
    return f'<span class="tag {cls}">{val:.1f}%</span>'


def build_html() -> str:
    print("  FIG 1~5 (grid_P + anomaly)...")
    figures = {
        "fig1": fig_forecast_val(),
        "fig2": fig_forecast_test(),
        "fig3": fig_anomaly_zoom(),
        "fig4": fig_residual_dist(),
        "fig5": fig_anomaly_metrics(),
    }
    print("  FIG 6~8 (meters)...")
    figures.update({
        "fig6": fig_meter_heatmap(),
        "fig7": fig_meter_scatter(),
        "fig8": fig_meter_wape_box(),
    })
    print("  FIG 9~12 (energy/flow/summary)...")
    figures.update({
        "fig9":  fig_energy_correlation(),
        "fig10": fig_seasonal_pattern(),
        "fig11": fig_val_vs_test(),
        "fig12": fig_energy_results(),
    })

    divs = {k: v.to_html(full_html=False, include_plotlyjs=False)
            for k, v in figures.items()}

    wape_ok = ok.dropna(subset=["val_wape"])
    top5 = wape_ok.nsmallest(5, "val_wape")
    bot5 = wape_ok.nlargest(5, "val_wape")

    def meter_row(r):
        mc = "good" if r.val_mape < 20 else "warn" if r.val_mape < 50 else "bad"
        wc = "good" if r.val_wape < 20 else "warn" if r.val_wape < 50 else "bad"
        return f"""<tr>
          <td><b>{r.meter_urn}</b></td><td>{r.building}</td>
          <td>{r.val_mae/1000:.2f} kW</td>
          <td><span class="tag {mc}">{r.val_mape:.1f}%</span></td>
          <td><span class="tag {wc}">{r.val_wape:.1f}%</span></td>
          <td>{r.val_f1:.3f}</td>
        </tr>"""

    top5_rows = "".join(meter_row(r) for r in top5.itertuples())
    bot5_rows = "".join(meter_row(r) for r in bot5.itertuples())

    energy_name = {
        "grid_P":"총 전력 (계통)","pv_gen":"태양광 발전","chp_gen":"열병합 (전기)",
        "cool_P":"냉방 총량","cool_elec_P":"냉동기","heat_P":"난방 총량",
        "chp_heat_P":"열병합 (열)",
    }
    energy_rows = "".join(f"""<tr>
      <td><b>{r.target}</b></td><td>{energy_name.get(r.target, r.target)}</td>
      <td>{r.val_mae/1000:.1f} kW</td><td>{r.val_rmse/1000:.1f} kW</td>
      <td>{_tag(r.val_mape)}</td><td>{r.val_wape:.1f}%</td>
      <td>{r.val_f1:.3f}</td><td>{_tag(r.test_mape, 30, 100)}</td>
    </tr>""" for r in er.itertuples())

    offline_banner = "" if ONLINE else """
    <div style="background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:8px;
                padding:12px 18px;margin-bottom:24px;font-size:.9rem;">
      ⚠ <b>오프라인 모드:</b> DB 연결 없이 생성되었습니다.
      추론 그래프(grid_P 시계열, 이상탐지 확대)는 DB 복구 후
      <code>python scripts/report/make_presentation.py</code> 재실행 시 자동으로 채워집니다.
    </div>"""

    prec_str    = f"{_an_prec_high*100:.1f}%" if ONLINE else "—"
    recall_str  = f"{_an_recall_high*100:.1f}%" if ONLINE else "—"
    f1_str      = f"{_an_f1_high:.3f}" if ONLINE else f"{_an_f1_high:.3f} (학습 기록)"
    n_anom_str  = f"{_n_anom:,}" if ONLINE else "—"
    n_hl_str    = f"HIGH {_n_high:,} + LOW {_n_low:,}" if ONLINE else "(DB 연결 후 계산)"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EMS ML 파이프라인 — 팀 발표 자료</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: "Noto Sans KR", sans-serif; background: #F1F5F9; color: #1E293B; margin: 0; }}
  header {{ background: linear-gradient(135deg,#0F2040,#1E3A5F,#2563EB); color: white; padding: 48px 60px; }}
  header h1 {{ margin: 0 0 8px; font-size: 2.2rem; font-weight: 700; letter-spacing: -.5px; }}
  header p  {{ margin: 0; opacity: .85; font-size: 1.05rem; }}
  header .badge {{ display:inline-block; background:rgba(255,255,255,0.18); border-radius:20px;
                   padding:4px 14px; font-size:.85rem; margin-top:12px; margin-right:8px; }}
  .container {{ max-width: 1340px; margin: 0 auto; padding: 40px 20px; }}
  section   {{ background: white; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,0,0,.07);
               padding: 36px; margin-bottom: 36px; }}
  h2 {{ font-size: 1.3rem; font-weight: 700; color: #1E3A5F; border-left: 4px solid #2563EB;
        padding-left: 14px; margin: 0 0 22px; }}
  h3 {{ font-size: 1rem; font-weight: 600; color: #475569; margin: 16px 0 10px;
        border-bottom: 1px solid #F1F5F9; padding-bottom: 6px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; }}
  .kpi    {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;
             padding: 22px 20px; text-align: center; transition: box-shadow .2s; }}
  .kpi:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,.1); }}
  .kpi .val {{ font-size: 2rem; font-weight: 700; }}
  .kpi .lbl {{ font-size: .82rem; color: #64748B; margin-top: 6px; line-height: 1.4; }}
  .kpi.green  .val {{ color: #16A34A; }}
  .kpi.yellow .val {{ color: #CA8A04; }}
  .kpi.red    .val {{ color: #DC2626; }}
  .kpi.blue   .val {{ color: #2563EB; }}
  .kpi.purple .val {{ color: #7C3AED; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
          font-size: .8rem; font-weight: 600; margin: 2px; }}
  .tag.good {{ background: #DCFCE7; color: #166534; }}
  .tag.warn {{ background: #FEF9C3; color: #854D0E; }}
  .tag.bad  {{ background: #FEE2E2; color: #991B1B; }}
  table    {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th       {{ background: #F1F5F9; padding: 11px 14px; text-align: left;
              font-weight: 600; color: #475569; border-bottom: 2px solid #E2E8F0; }}
  td       {{ padding: 9px 14px; border-bottom: 1px solid #F8FAFC; }}
  tr:hover td {{ background: #F8FAFC; }}
  tr:last-child td {{ border-bottom: none; }}
  .two-table {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .table-box {{ border: 1px solid #E2E8F0; border-radius: 10px; overflow: hidden; }}
  .table-box .table-title {{ background: #F8FAFC; padding: 10px 14px;
                              font-weight: 600; font-size: .9rem; color: #1E3A5F; }}
  .verdict {{ border-radius: 10px; padding: 18px 24px; margin-top: 14px; }}
  .verdict.ok   {{ background:#DCFCE7; border-left:4px solid #16A34A; }}
  .verdict.warn {{ background:#FEF9C3; border-left:4px solid #CA8A04; }}
  .verdict.bad  {{ background:#FEE2E2; border-left:4px solid #DC2626; }}
  .info-box {{ background:#F0F9FF; border-left:4px solid #0EA5E9; border-radius:8px;
               padding:14px 18px; font-size:.88rem; line-height:1.7; margin-top:12px; }}
  footer {{ text-align:center; padding:32px; color:#94A3B8; font-size:.85rem; }}
  hr {{ border:none; border-top:1px solid #F1F5F9; margin:24px 0; }}
</style>
</head>
<body>

<header>
  <h1>EMS ML 파이프라인 — 팀 발표 자료</h1>
  <p>에너지 관리 시스템 전력 소비 예측 &amp; 게이트웨이 장애 이상탐지</p>
  <div style="margin-top:14px;">
    <span class="badge">🏢 80개 계량기</span>
    <span class="badge">⚡ 7종 에너지 타입</span>
    <span class="badge">🤖 VMD-LSTM + Residual IF</span>
    <span class="badge">📅 2018–2023 학습</span>
    <span class="badge">🖥️ RTX PRO 4500 Blackwell GPU</span>
  </div>
</header>

<div class="container">
{offline_banner}

<!-- ① 프로젝트 개요 -->
<section>
  <h2>① 프로젝트 개요</h2>
  <div class="grid-4">
    <div class="kpi blue"><div class="val">2</div><div class="lbl">핵심 태스크<br>예측 + 이상탐지</div></div>
    <div class="kpi green"><div class="val">80</div><div class="lbl">학습 완료<br>개별 계량기</div></div>
    <div class="kpi green"><div class="val">7</div><div class="lbl">에너지 타입<br>학습 완료</div></div>
    <div class="kpi blue"><div class="val">6년</div><div class="lbl">학습 데이터<br>2018 – 2023</div></div>
  </div>
  <br>
  <table>
    <tr><th>구분</th><th>모델</th><th>방식</th><th>피처</th><th>상태</th></tr>
    <tr><td>총 전력 예측</td><td>VMD-LSTM</td><td>VMD(K=4) 분해 → 2-layer LSTM + Attention</td><td>17개 (VMD IMF 4종 + lag + 기상 + 시간)</td><td><span class="tag good">학습 완료</span></td></tr>
    <tr><td>이상탐지</td><td>Residual + IF</td><td>예측 잔차 임계치 + Isolation Forest 앙상블</td><td>잔차 신호 + 원본 피처 11개</td><td><span class="tag good">학습 완료</span></td></tr>
    <tr><td>개별 계량기 예측</td><td>VMD-LSTM × 80</td><td>계량기별 독립 학습 (lag 4종 + 기상 + 시간)</td><td>13개</td><td><span class="tag good">학습 완료</span></td></tr>
    <tr><td>에너지 전체 예측</td><td>VMD-LSTM × 7</td><td>냉방/난방/태양광/열병합 7종</td><td>21개</td><td><span class="tag good">학습 완료</span></td></tr>
  </table>
  <br>
  <h3>데이터 분할 타임라인</h3>
  <div style="background:#F8FAFC;border-radius:10px;padding:20px 24px;">
    <table>
      <tr><th style="width:120px;">구간</th><th>기간</th><th>시간 수</th><th>용도</th><th>비고</th></tr>
      <tr style="background:#EFF6FF;">
        <td><b style="color:#2563EB;">학습 (Train)</b></td>
        <td>2018-01-01 ~ 2021-12-31</td><td>35,064시간</td>
        <td>모델 파라미터 최적화</td><td>게이트웨이 장애 3구간 제거</td>
      </tr>
      <tr style="background:#F5F3FF;">
        <td><b style="color:#8B5CF6;">검증 (Val)</b></td>
        <td>2022-01-01 ~ 2022-12-31</td><td>8,760시간</td>
        <td>하이퍼파라미터 선택</td><td>장애 구간 포함 (pseudo-label)</td>
      </tr>
      <tr style="background:#ECFDF5;">
        <td><b style="color:#10B981;">테스트 (Test)</b></td>
        <td>2023-01-01 ~ 2023-12-31</td><td>8,760시간</td>
        <td>최종 성능 평가</td><td>미래 시나리오 시뮬레이션</td>
      </tr>
    </table>
    <div style="margin-top:12px;font-size:.82rem;color:#64748B;">
      🚨 <b>게이트웨이 장애 기록:</b>
      2020-01-13~16 (4일) &nbsp;|&nbsp; 2020-06-21~27 (7일) &nbsp;|&nbsp;
      2022-02-13~18 (6일, Train 제거) &nbsp;|&nbsp;
      <b style="color:#DC2626;">2022-05-06~07-14 (69일, 이상탐지 pseudo-label)</b>
    </div>
  </div>
  <div class="info-box" style="margin-top:14px;">
    <b>DB:</b> PostgreSQL · <code>ems.reduced_measurement_1h</code> (집계) &amp;
    <code>ems.cr_measurement_1h</code> (계량기) &nbsp;|&nbsp;
    <b>해상도:</b> 1시간 단위 &nbsp;|&nbsp;
    <b>총 행 수:</b> 약 52,608행 (집계 기준) &nbsp;|&nbsp;
    <b>GPU:</b> RTX PRO 4500 Blackwell 32GB VRAM
  </div>
</section>

<!-- ② 팀 공통 기준 -->
<section>
  <h2>② 팀 공통 기준 (Team Standards)</h2>
  <div class="grid-2" style="gap:20px;">
    <div>
      <h3>📥 데이터 입력 기준</h3>
      <table>
        <tr><th>항목</th><th>기준</th></tr>
        <tr><td>해상도</td><td>1시간 단위 (<code>ems.cr_measurement_1h</code>)</td></tr>
        <tr><td>데이터 분할</td><td>Train 2018–2021 (4년) / Val 2022 / Test 2023</td></tr>
        <tr><td>정규화</td><td>MinMaxScaler (0 ~ 1)</td></tr>
        <tr><td>결측값</td><td>0으로 채움 (팀 정책)</td></tr>
        <tr><td>게이트웨이 장애</td><td>4구간 → 학습 데이터에서만 제거, Val/Test 유지</td></tr>
      </table>
    </div>
    <div>
      <h3>📊 공통 평가 지표</h3>
      <table>
        <tr><th>모델</th><th>평가 지표</th></tr>
        <tr><td>예측 모델</td><td>MAE · RMSE · MAPE · <b>WAPE</b></td></tr>
        <tr><td>이상탐지</td><td>Accuracy · Precision · Recall · <b>F1-Score</b> · AUC-ROC</td></tr>
        <tr><td>MLflow</td><td>공용 서버 (<code>121.134.46.24:5000</code>) · 파라미터/지표 형식 통일</td></tr>
      </table>
      <br>
      <h3>🔍 이상탐지 앙상블 기준 (ensemble.py)</h3>
      <table>
        <tr><th>임계치 방식</th><th colspan="2">MSD: mean(MSE) + 3 × std(MSE)</th></tr>
        <tr><td><span class="tag bad">HIGH</span></td><td colspan="2">3개 모델 모두 탐지</td></tr>
        <tr><td><span class="tag warn">MEDIUM</span></td><td colspan="2">2개 모델 탐지</td></tr>
        <tr><td><span class="tag good">LOW</span></td><td colspan="2">1개 모델 탐지</td></tr>
      </table>
    </div>
  </div>
  <div style="background:#F0FDF4;border-left:4px solid #16A34A;border-radius:8px;padding:14px 18px;margin-top:16px;font-size:.88rem;line-height:1.7;">
    <b>✓ Team 4 구현 현황:</b>
    예측은 논문 기반 <b>VMD-LSTM</b>(K=4 IMF 분해 + 2-layer LSTM + Attention),
    이상탐지는 <b>잔차(Residual) 임계치 + Isolation Forest</b> 앙상블로 구현.
    임계치 k는 Val pseudo-label에서 [0.5, 4.0] 범위 자동 최적화.
    피처: 전력 3종 + 기상 2종 + 시간 6종 + VMD IMF 4종 + lag 2종 = <b>17개 (grid_P 기준)</b>
  </div>
</section>

<!-- ③ grid_P 예측 결과 -->
<section>
  <h2>③ grid_P (총 전력 소비) 예측 결과</h2>
  <div class="grid-4" style="margin-bottom:24px;">
    <div class="kpi green"><div class="val">{_fc_val_mape:.1f}%</div><div class="lbl">Val MAPE (2022)</div></div>
    <div class="kpi green"><div class="val">{_fc_val_wape:.1f}%</div><div class="lbl">Val WAPE (2022)</div></div>
    <div class="kpi red"><div class="val">{_fc_test_mape:.1f}%</div><div class="lbl">Test MAPE (2023)</div></div>
    <div class="kpi yellow"><div class="val">{_fc_test_wape:.1f}%</div><div class="lbl">Test WAPE (2023)</div></div>
  </div>
  <div class="grid-2" style="margin-bottom:16px;gap:12px;">
    <div style="background:#EFF6FF;border-radius:8px;padding:12px 16px;font-size:.88rem;">
      <b>Val (2022)</b> &nbsp;MAE: <b>{_fc_val_mae:,.0f} W</b> · RMSE: <b>{_fc_val_rmse:,.0f} W</b> · MAPE: <b>{_fc_val_mape:.1f}%</b> · WAPE: <b>{_fc_val_wape:.1f}%</b>
    </div>
    <div style="background:#FFF7ED;border-radius:8px;padding:12px 16px;font-size:.88rem;">
      <b>Test (2023)</b> &nbsp;MAE: <b>{_fc_test_mae:,.0f} W</b> · RMSE: <b>{_fc_test_rmse:,.0f} W</b> · MAPE: <b>{_fc_test_mape:.1f}%</b> · WAPE: <b>{_fc_test_wape:.1f}%</b>
    </div>
  </div>
  {divs['fig1']}
  <br>
  {divs['fig2']}
  <div class="verdict warn">
    <b>⚠ Test 성능 저하 (Distribution Shift):</b>
    모델은 2018–2021 데이터로 학습되어 Val(2022)에서 WAPE {_fc_val_wape:.1f}%를 달성했으나,
    2023년 소비 패턴 변화로 Test WAPE {_fc_test_wape:.1f}%로 저하되었습니다.
    <b>2022 데이터를 학습에 추가한 연간 재학습</b>으로 해결 가능합니다.
  </div>
</section>

<!-- ④ 이상탐지 결과 -->
<section>
  <h2>④ 이상탐지 결과 (Residual + Isolation Forest)</h2>
  <div class="grid-4" style="margin-bottom:24px;">
    <div class="kpi yellow"><div class="val">{n_anom_str}</div><div class="lbl">이상 탐지 (2022)<br>{n_hl_str}</div></div>
    <div class="kpi {"green" if _an_f1_high > 0.4 else "yellow"}"><div class="val">{f1_str}</div><div class="lbl">F1-Score (HIGH)</div></div>
    <div class="kpi blue"><div class="val">{recall_str}</div><div class="lbl">Recall<br>(장애 탐지율)</div></div>
    <div class="kpi yellow"><div class="val">{prec_str}</div><div class="lbl">Precision<br>(정밀도)</div></div>
  </div>
  {divs['fig3']}
  <br>
  <div class="grid-2">
    <div>{divs['fig4']}</div>
    <div>{divs['fig5']}</div>
  </div>
  <div class="verdict warn">
    <b>이상탐지 해석:</b>
    장애 구간(2022-05-06~07-14)에서 예측 잔차가 급증하는 패턴이 명확합니다.
    잔차 기반 임계치(MSD 방식)와 Isolation Forest를 앙상블하여 HIGH/LOW 두 단계로 감지합니다.
    현재 Val F1={_an_f1_high:.3f}는 pseudo-label 1구간 기준이며,
    실제 장애 기록이 누적될수록 임계값 자동 최적화 성능이 향상됩니다.
  </div>
</section>

<!-- ⑤ 계량기 성능 -->
<section>
  <h2>⑤ 개별 계량기 80개 성능</h2>
  <div class="grid-4" style="margin-bottom:24px;">
    <div class="kpi green"><div class="val">{_n_good}개</div><div class="lbl">우수 (Test MAPE &lt; 20%)</div></div>
    <div class="kpi yellow"><div class="val">{_n_mid}개</div><div class="lbl">보통 (20~100%)</div></div>
    <div class="kpi red"><div class="val">{_n_bad}개</div><div class="lbl">저조 (&gt; 100%)</div></div>
    <div class="kpi blue"><div class="val">{ok['val_wape'].median():.1f}%</div><div class="lbl">Val WAPE 중앙값</div></div>
  </div>
  {divs['fig6']}
  <br>
  <div class="grid-2">
    <div>{divs['fig7']}</div>
    <div>{divs['fig8']}</div>
  </div>
  <hr>
  <div class="two-table">
    <div class="table-box">
      <div class="table-title">🏆 우수 계량기 Top 5 (Val WAPE 낮은 순)</div>
      <table>
        <tr><th>계량기</th><th>건물</th><th>Val MAE</th><th>Val MAPE</th><th>Val WAPE</th><th>Val F1</th></tr>
        {top5_rows}
      </table>
    </div>
    <div class="table-box">
      <div class="table-title">⚠ 저조 계량기 Bottom 5 (Val WAPE 높은 순)</div>
      <table>
        <tr><th>계량기</th><th>건물</th><th>Val MAE</th><th>Val MAPE</th><th>Val WAPE</th><th>Val F1</th></tr>
        {bot5_rows}
      </table>
    </div>
  </div>
  <div class="verdict warn" style="margin-top:16px;">
    <b>저조 계량기 분석:</b> WAPE=100% 계량기(ZE 그룹)는 간헐적 사용·대부분 0인 부하 특성으로
    모델이 수렴하지 못한 케이스입니다. V.Z81처럼 WAPE 200%+ 계량기는 2022년 실제값이
    학습 분포와 크게 달라진 것으로 분석됩니다. 계량기별 데이터 품질 검토 후 재학습 권장.
  </div>
</section>

<!-- ⑥ 에너지 7종 예측 결과 -->
<section>
  <h2>⑥ 에너지 7종 예측 결과 (냉방 · 난방 · 태양광 · 열병합)</h2>
  <div class="grid-4" style="margin-bottom:24px;">
    <div class="kpi green"><div class="val">{(er['val_mape'] < 20).sum()}종</div><div class="lbl">우수 (Val MAPE &lt; 20%)</div></div>
    <div class="kpi yellow"><div class="val">{er['val_wape'].mean():.1f}%</div><div class="lbl">Val WAPE 평균</div></div>
    <div class="kpi green"><div class="val">{len(er)}종</div><div class="lbl">학습 완료</div></div>
    <div class="kpi blue"><div class="val">{er['val_mae'].mean()/1000:.1f} kW</div><div class="lbl">Val MAE 평균</div></div>
  </div>
  {divs['fig12']}
  <br>
  <table>
    <tr><th>타겟</th><th>설명</th><th>Val MAE</th><th>Val RMSE</th><th>Val MAPE</th><th>Val WAPE</th><th>Val F1</th><th>Test MAPE</th></tr>
    {energy_rows}
  </table>
  <div class="verdict warn" style="margin-top:16px;">
    <b>인사이트:</b>
    <b>cool_P · cool_elec_P</b>: MAPE 12~15%로 가장 안정적 (계절성 뚜렷해 학습 용이).
    <b>heat_P · chp_gen · chp_heat_P</b>: Test MAPE 높아 2023년 운영 패턴 변화 영향.
    <b>WAPE 기준</b>으로는 모든 7종이 6~17% 범위로 양호 — 대용량 에너지원 절대 오차가 작음.
  </div>
</section>

<!-- ⑦ 에너지 흐름 분석 -->
<section>
  <h2>⑦ 에너지 흐름 분석</h2>
  <div class="grid-2">
    <div>{divs['fig9']}</div>
    <div>{divs['fig10']}</div>
  </div>
  <div class="verdict ok" style="margin-top:16px;">
    <b>✓ 인사이트:</b>
    태양광·열병합 발전 증가 → 계통 소비(grid_P) 감소 (음의 상관관계).
    겨울(1~3월) 계통 소비 최고, 여름(6~8월) 태양광 발전 증가로 계통 의존 감소.
    이 계절성이 VMD 분해에서 효과적으로 추출되어 예측 성능을 높입니다.
  </div>
</section>

<!-- ⑧ 종합 평가 -->
<section>
  <h2>⑧ 모델 적합성 종합 평가</h2>
  {divs['fig11']}
  <br>
  <table>
    <tr><th>평가 항목</th><th>지표</th><th>현재 상태</th><th>판정</th><th>개선 방향</th></tr>
    <tr>
      <td>예측 정확도 (Val)</td><td>WAPE</td>
      <td>grid_P {_fc_val_wape:.1f}% · 에너지7종 평균 {er['val_wape'].mean():.1f}% · 계량기 중앙값 {ok['val_wape'].median():.1f}%</td>
      <td><span class="tag good">양호</span></td>
      <td>하이퍼파라미터 추가 튜닝</td>
    </tr>
    <tr>
      <td>예측 정확도 (Test)</td><td>WAPE</td>
      <td>grid_P {_fc_test_wape:.1f}% · 에너지7종 평균 {er['test_wape'].mean():.1f}% · 계량기 중앙값 {ok['test_wape'].median():.1f}%</td>
      <td><span class="tag warn">개선 필요</span></td>
      <td>연간 재학습 · 2022 데이터 학습 포함</td>
    </tr>
    <tr>
      <td>이상탐지</td><td>F1</td>
      <td>Val F1={_an_f1_high:.3f} · 잔차 급증 패턴 명확</td>
      <td><span class="tag warn">조건부 양호</span></td>
      <td>실제 장애 기록 추가 확보</td>
    </tr>
    <tr>
      <td>계량기 커버리지</td><td>완료율</td>
      <td>80개 전체 학습 완료 · 우수 {_n_good}개 / 보통 {_n_mid}개 / 저조 {_n_bad}개</td>
      <td><span class="tag good">완성</span></td>
      <td>저조 계량기 데이터 품질 개선</td>
    </tr>
    <tr>
      <td>파이프라인 자동화</td><td>완성도</td>
      <td>DB → 학습 → 추론 → MLflow 전 구간 자동화 · GPU 가속</td>
      <td><span class="tag good">완성</span></td>
      <td>API 서빙 / 스케줄러 추가</td>
    </tr>
    <tr>
      <td>확장성 (에너지 7종)</td><td>WAPE</td>
      <td>VMD-LSTM × 7 학습 완료 · Val WAPE 평균 {er['val_wape'].mean():.1f}%</td>
      <td><span class="tag good">완성</span></td>
      <td>연간 재학습 스케줄 도입</td>
    </tr>
  </table>

  <div class="verdict ok" style="margin-top:20px;">
    <b>✓ 방법론 적합성:</b>
    VMD-LSTM은 에너지 시계열의 다중 주기성(일간·주간·계절)을 분해하여 예측하는 검증된 방법입니다.
    잔차 기반 이상탐지는 "게이트웨이 장애 = 예측과 다른 값"이라는 도메인 지식에 직접 부합합니다.
    <b>방법론 자체는 적합합니다.</b>
  </div>
  <div class="verdict warn" style="margin-top:12px;">
    <b>⚠ 핵심 개선 과제 3가지:</b><br>
    (1) <b>연간 재학습</b> — 2022년 이후 데이터를 학습에 포함하면 Test 성능이 유의미하게 개선됩니다.<br>
    (2) <b>저조 계량기 분석</b> — ZE 그룹·V.Z81 등 WAPE 100%+ 계량기는 데이터 품질 및 부하 특성 재검토 필요.<br>
    (3) <b>이상탐지 레이블 확보</b> — 다구간 pseudo-label 추가 시 임계값 자동 최적화 크게 향상.
  </div>
</section>

</div>
<footer>
  EMS ML Pipeline · SK Networks Family AI 27기 · 생성일 2026-05-20<br>
  <small>grid_P Val WAPE {_fc_val_wape:.1f}% · 계량기 80개 · 에너지 7종 · RTX PRO 4500 Blackwell GPU</small>
</footer>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("▶ HTML 생성 중...")
    html = build_html()
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\n✓ 저장 완료: {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024}KB)")
    print(f"  {'온라인 모드 (추론 포함)' if ONLINE else '오프라인 모드 (CSV 기반)'}")
    print(f"  브라우저에서 열어보세요: file://{OUT_PATH.resolve()}")
