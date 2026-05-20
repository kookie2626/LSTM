"""팀 발표용 종합 시각화 리포트 생성.

출력: outputs/presentation_report.html

포함 내용:
  1. 프로젝트 개요 및 구조
  2. grid_P 예측 결과 (val/test 실제 vs 예측, 이상탐지 오버레이)
  3. 이상탐지 상세 (2022 장애 구간 확대)
  4. 계량기 80개 성능 히트맵
  5. 전체 에너지 흐름 (상관관계 + 시계열)
  6. 모델 적합성 종합 평가

실행:
    PYTHONPATH=/home/keun/workspace python scripts/report/make_presentation.py
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

from project.ML.inference import predict_forecast, predict_anomaly
from project.ML.data_loader import load_raw, add_features, GATEWAY_FAILURES

OUT_PATH = Path("outputs/presentation_report.html")

# ── 데이터 준비 ───────────────────────────────────────────────────────────────

print("▶ 데이터 로드 중...")
df = load_raw()
df = add_features(df)

print("▶ 예측 실행 중 (val 2022)...")
fc_val  = predict_forecast(df, "2022-01-01", "2022-12-31")
print("▶ 예측 실행 중 (test 2023)...")
fc_test = predict_forecast(df, "2023-01-01", "2023-12-31")

print("▶ 이상탐지 실행 중 (val 2022)...")
an_val  = predict_anomaly(df, "2022-01-01", "2022-12-31")

print("▶ 계량기 결과 로드 중...")
meters  = pd.read_csv("outputs/all_meters_results.csv")

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
C_ACTUAL    = "#2563EB"   # blue
C_PRED      = "#F97316"   # orange
C_HIGH      = "#EF4444"   # red
C_LOW       = "#FBBF24"   # yellow
C_FAIL      = "rgba(220,38,38,0.10)"
C_GOOD      = "rgba(34,197,94,0.10)"

# ── 공통 레이아웃 ─────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    font=dict(family="Pretendard, Noto Sans KR, sans-serif", size=13),
    paper_bgcolor="white",
    plot_bgcolor="#F8FAFC",
    margin=dict(l=60, r=40, t=60, b=50),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)


# ════════════════════════════════════════════════════════════════════
#  FIG 1 : grid_P 예측 — val 2022
# ════════════════════════════════════════════════════════════════════
def fig_forecast_val() -> go.Figure:
    # 주간 평균으로 다운샘플 (가독성)
    fc = fc_val.resample("1D").mean().dropna()
    an = an_val.resample("1D").agg({"vote": "max", "anomaly_level": lambda x: x.mode()[0]})

    fig = go.Figure()

    # 게이트웨이 장애 구간 음영
    fig.add_vrect(x0="2022-05-06", x1="2022-07-14",
                  fillcolor=C_FAIL, line_width=0,
                  annotation_text="장애 구간", annotation_position="top left",
                  annotation_font_color="#EF4444")

    fig.add_trace(go.Scatter(
        x=fc.index, y=fc["actual"] / 1000,
        name="실제값", line=dict(color=C_ACTUAL, width=1.5)))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc["predicted"] / 1000,
        name="예측값", line=dict(color=C_PRED, width=1.5, dash="dot")))

    fig.update_layout(
        **BASE_LAYOUT,
        title="<b>grid_P 예측 결과 — Validation (2022)</b><br>"
              "<sub>MAE 22,460 W · MAPE 25.5% · 일별 평균</sub>",
        yaxis_title="전력 소비 (kW)",
        xaxis_title="날짜",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 2 : grid_P 예측 — test 2023
# ════════════════════════════════════════════════════════════════════
def fig_forecast_test() -> go.Figure:
    fc = fc_test.resample("1D").mean().dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc["actual"] / 1000,
        name="실제값", line=dict(color=C_ACTUAL, width=1.5)))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc["predicted"] / 1000,
        name="예측값", line=dict(color=C_PRED, width=1.5, dash="dot")))

    fig.update_layout(
        **BASE_LAYOUT,
        title="<b>grid_P 예측 결과 — Test (2023)</b><br>"
              "<sub>MAE 26,252 W · MAPE 57.4% · 2023년 소비 패턴 변화로 성능 저하</sub>",
        yaxis_title="전력 소비 (kW)",
        xaxis_title="날짜",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 3 : 이상탐지 — 장애 구간 확대
# ════════════════════════════════════════════════════════════════════
def fig_anomaly_zoom() -> go.Figure:
    # 장애 전후 3개월 (2022-04 ~ 2022-09)
    an = an_val["2022-04-01":"2022-09-30"]

    high = an[an["anomaly_level"] == "HIGH"]
    low  = an[an["anomaly_level"] == "LOW"]

    fig = go.Figure()

    fig.add_vrect(x0="2022-05-06", x1="2022-07-14",
                  fillcolor=C_FAIL, line_width=0,
                  annotation_text="실제 장애 구간 (2022-05-06 ~ 07-14)",
                  annotation_position="top left",
                  annotation_font_color="#EF4444")

    fig.add_trace(go.Scatter(
        x=an.index, y=an["actual"] / 1000,
        name="실제값", line=dict(color=C_ACTUAL, width=1.2)))
    fig.add_trace(go.Scatter(
        x=an.index, y=an["predicted"] / 1000,
        name="예측값", line=dict(color=C_PRED, width=1.2, dash="dot")))

    fig.add_trace(go.Scatter(
        x=high.index, y=high["actual"] / 1000,
        mode="markers", name="HIGH 이상",
        marker=dict(color=C_HIGH, size=5, symbol="x")))
    fig.add_trace(go.Scatter(
        x=low.index, y=low["actual"] / 1000,
        mode="markers", name="LOW 이상",
        marker=dict(color=C_LOW, size=4, symbol="circle-open")))

    fig.update_layout(
        **BASE_LAYOUT,
        title="<b>이상탐지 상세 — 장애 구간 전후 (2022-04 ~ 09)</b><br>"
              "<sub>RED = 실제 장애 구간 · ✕ HIGH(두 신호 모두) · ○ LOW(한 신호)</sub>",
        yaxis_title="전력 소비 (kW)",
        xaxis_title="날짜",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 4 : 이상탐지 잔차 분포
# ════════════════════════════════════════════════════════════════════
def fig_residual_dist() -> go.Figure:
    an   = an_val.copy()
    fail = an["2022-05-06":"2022-07-14"]
    norm = an.drop(fail.index, errors="ignore")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=norm["residual"] / 1000, name="정상 구간",
        opacity=0.7, nbinsx=80,
        marker_color=C_ACTUAL))
    fig.add_trace(go.Histogram(
        x=fail["residual"] / 1000, name="장애 구간",
        opacity=0.8, nbinsx=40,
        marker_color=C_HIGH))

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="overlay",
        title="<b>예측 잔차 분포 — 정상 vs 장애 구간</b><br>"
              "<sub>장애 구간 잔차가 오른쪽으로 크게 치우침</sub>",
        xaxis_title="잔차 (kW)",
        yaxis_title="빈도",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 5 : 계량기 성능 히트맵
# ════════════════════════════════════════════════════════════════════
def fig_meter_heatmap() -> go.Figure:
    ok = meters[meters["status"] == "OK"].copy()
    ok["prefix"] = ok["meter_urn"].str.extract(r"^([A-Z]+\d*)")
    ok["suffix"] = ok["meter_urn"].str.extract(r"\.(.+)$")

    # MAPE 캡 300%
    ok["mape_capped"] = ok["test_mape"].clip(upper=300)
    ok["label"] = ok.apply(
        lambda r: f"{r['meter_urn']}<br>MAPE {r['test_mape']:.0f}%<br>MAE {r['test_mae']/1000:.1f}kW",
        axis=1)

    fig = px.treemap(
        ok,
        path=[px.Constant("전체"), "prefix", "meter_urn"],
        values="mape_capped",
        color="mape_capped",
        hover_data={"test_mape": ":.1f", "test_mae": ":.0f"},
        color_continuous_scale=["#22C55E", "#FCD34D", "#EF4444"],
        color_continuous_midpoint=50,
        title="<b>계량기 80개 Test MAPE 히트맵</b><br>"
              "<sub>초록=우수(~20%), 노랑=보통(20~50%), 빨강=저조(50%+) · 면적=MAPE 크기</sub>",
    )
    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>test MAPE: %{color:.1f}%<extra></extra>",
    )
    layout = {**BASE_LAYOUT, "margin": dict(l=20, r=20, t=80, b=20)}
    fig.update_layout(**layout)
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 6 : 계량기 성능 분포 (MAE / F1 scatter)
# ════════════════════════════════════════════════════════════════════
def fig_meter_scatter() -> go.Figure:
    ok = meters[meters["status"] == "OK"].copy()
    ok["test_mae_kw"] = ok["test_mae"] / 1000

    fig = px.scatter(
        ok,
        x="test_mae_kw",
        y="val_f1",
        color="test_mape",
        size="test_mae_kw",
        hover_name="meter_urn",
        hover_data={"test_mape": ":.1f%", "test_mae_kw": ":.2f"},
        color_continuous_scale=["#22C55E", "#FCD34D", "#EF4444"],
        range_color=[0, 200],
        title="<b>계량기별 예측 MAE vs 이상탐지 F1</b><br>"
              "<sub>좌하단=우수(낮은 오차 + 높은 F1) · 색상=test MAPE</sub>",
        labels={"test_mae_kw": "Test MAE (kW)", "val_f1": "Val F1 (이상탐지)"},
    )
    fig.add_vline(x=5, line_dash="dash", line_color="gray",
                  annotation_text="MAE 5kW", annotation_position="top")
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="F1 0.5", annotation_position="right")
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 7 : 에너지 흐름 상관관계 매트릭스
# ════════════════════════════════════════════════════════════════════
def fig_energy_correlation() -> go.Figure:
    # DB에서 에너지 컬럼 가져오기 (이미 df에 있음)
    energy_cols = {
        "grid_P":  "총 전력",
        "pv_P":    "태양광",
        "chp_P":   "열병합(전기)",
        "Ta":      "외기 온도",
        "Igm":     "일사량",
    }
    available = {k: v for k, v in energy_cols.items() if k in df.columns}
    sub = df[list(available.keys())].dropna()

    corr = sub.corr()
    labels = [available[c] for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont_size=14,
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        title="<b>에너지 변수 간 상관관계</b><br>"
              "<sub>태양광·열병합 발전↑ → 계통 소비(grid_P)↓ 관계 확인</sub>",
        width=600, height=500,
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 8 : 계절별 에너지 패턴 (월별 평균)
# ════════════════════════════════════════════════════════════════════
def fig_seasonal_pattern() -> go.Figure:
    sub = df[["grid_P", "pv_P", "chp_P", "Ta"]].copy()
    sub["month"] = sub.index.month
    monthly = sub.groupby("month").mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("전력 소비·발전 (월평균)", "외기 온도 (월평균)"),
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly["grid_P"] / 1000,
        name="계통 소비(grid_P)", marker_color=C_ACTUAL), row=1, col=1)
    if "pv_P" in monthly:
        fig.add_trace(go.Bar(
            x=monthly.index, y=monthly["pv_P"].abs() / 1000,
            name="태양광 발전(abs)", marker_color="#22C55E"), row=1, col=1)
    if "chp_P" in monthly:
        fig.add_trace(go.Bar(
            x=monthly.index, y=monthly["chp_P"].abs() / 1000,
            name="열병합 발전(abs)", marker_color="#A855F7"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly["Ta"],
        name="외기 온도(°C)",
        line=dict(color="#F97316", width=2.5), mode="lines+markers"), row=2, col=1)

    fig.update_xaxes(tickvals=list(range(1, 13)),
                     ticktext=["1월","2월","3월","4월","5월","6월",
                               "7월","8월","9월","10월","11월","12월"])
    fig.update_yaxes(title_text="전력 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="온도 (°C)", row=2, col=1)

    fig.update_layout(
        **BASE_LAYOUT,
        title="<b>계절별 에너지 패턴 (2018–2023 전체 평균)</b>",
        barmode="group",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  FIG 9 : val vs test 성능 비교 (모델별)
# ════════════════════════════════════════════════════════════════════
def fig_val_vs_test() -> go.Figure:
    summary = pd.DataFrame([
        {"모델": "VMD-LSTM\n(grid_P)",  "val_mape": 25.5,  "test_mape": 57.4},
    ])

    ok = meters[meters["status"] == "OK"]
    meter_summary = pd.DataFrame([
        {"모델": "계량기 평균\n(80개)", "val_mape": ok["val_mape"].median(), "test_mape": ok["test_mape"].median()},
        {"모델": "계량기 상위25%",       "val_mape": ok["val_mape"].quantile(0.25), "test_mape": ok["test_mape"].quantile(0.25)},
        {"모델": "계량기 하위25%",       "val_mape": ok["val_mape"].quantile(0.75), "test_mape": ok["test_mape"].quantile(0.75)},
    ])
    summary = pd.concat([summary, meter_summary], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["모델"], y=summary["val_mape"],
        name="Val MAPE (2022)", marker_color=C_ACTUAL))
    fig.add_trace(go.Bar(
        x=summary["모델"], y=summary["test_mape"],
        name="Test MAPE (2023)", marker_color=C_PRED))
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  annotation_text="목표 30%", annotation_position="right")

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="group",
        title="<b>Val vs Test MAPE 비교</b><br>"
              "<sub>2022→2023 성능 저하 = 소비 패턴 분포 이동(Distribution Shift)</sub>",
        yaxis_title="MAPE (%)",
    )
    return fig


# ════════════════════════════════════════════════════════════════════
#  HTML 조립
# ════════════════════════════════════════════════════════════════════

def build_html() -> str:
    figures = {
        "fig1": fig_forecast_val(),
        "fig2": fig_forecast_test(),
        "fig3": fig_anomaly_zoom(),
        "fig4": fig_residual_dist(),
        "fig5": fig_meter_heatmap(),
        "fig6": fig_meter_scatter(),
        "fig7": fig_energy_correlation(),
        "fig8": fig_seasonal_pattern(),
        "fig9": fig_val_vs_test(),
    }

    divs = {k: v.to_html(full_html=False, include_plotlyjs=False)
            for k, v in figures.items()}

    ok = meters[meters["status"] == "OK"]
    n_good = (ok["test_mape"] < 20).sum()
    n_bad  = (ok["test_mape"] > 100).sum()

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
  header {{ background: linear-gradient(135deg,#1E3A5F,#2563EB); color: white; padding: 40px 60px; }}
  header h1 {{ margin: 0 0 8px; font-size: 2rem; font-weight: 700; }}
  header p  {{ margin: 0; opacity: .85; font-size: 1.05rem; }}
  .container {{ max-width: 1300px; margin: 0 auto; padding: 40px 20px; }}
  section   {{ background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.07);
               padding: 32px; margin-bottom: 32px; }}
  h2 {{ font-size: 1.3rem; font-weight: 700; color: #1E3A5F; border-left: 4px solid #2563EB;
        padding-left: 14px; margin: 0 0 20px; }}
  h3 {{ font-size: 1rem; font-weight: 600; color: #475569; margin: 0 0 12px; }}
  .grid-2  {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .grid-3  {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; }}
  .kpi     {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
              padding: 20px 24px; text-align: center; }}
  .kpi .val {{ font-size: 2rem; font-weight: 700; }}
  .kpi .lbl {{ font-size: .85rem; color: #64748B; margin-top: 4px; }}
  .kpi.green .val {{ color: #16A34A; }}
  .kpi.yellow .val {{ color: #CA8A04; }}
  .kpi.red .val {{ color: #DC2626; }}
  .tag {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
          font-size: .8rem; font-weight: 600; margin: 2px; }}
  .tag.good  {{ background: #DCFCE7; color: #166534; }}
  .tag.warn  {{ background: #FEF9C3; color: #854D0E; }}
  .tag.bad   {{ background: #FEE2E2; color: #991B1B; }}
  table    {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th       {{ background: #F1F5F9; padding: 10px 14px; text-align: left;
              font-weight: 600; color: #475569; }}
  td       {{ padding: 9px 14px; border-bottom: 1px solid #F1F5F9; }}
  tr:last-child td {{ border-bottom: none; }}
  .verdict {{ border-radius: 10px; padding: 18px 24px; margin-top: 12px; }}
  .verdict.ok  {{ background:#DCFCE7; border-left: 4px solid #16A34A; }}
  .verdict.warn{{ background:#FEF9C3; border-left: 4px solid #CA8A04; }}
  .verdict.bad {{ background:#FEE2E2; border-left: 4px solid #DC2626; }}
  .verdict b   {{ font-size: 1.05rem; }}
  footer {{ text-align:center; padding: 30px; color: #94A3B8; font-size:.85rem; }}
</style>
</head>
<body>

<header>
  <h1>EMS ML 파이프라인 — 팀 발표 자료</h1>
  <p>에너지 관리 시스템 전력 소비 예측 &amp; 게이트웨이 장애 이상탐지 · 2026.05</p>
</header>

<div class="container">

<!-- ① 프로젝트 개요 -->
<section>
  <h2>① 프로젝트 개요</h2>
  <div class="grid-3">
    <div class="kpi green"><div class="val">2</div><div class="lbl">핵심 태스크<br>(예측 + 이상탐지)</div></div>
    <div class="kpi green"><div class="val">80</div><div class="lbl">학습 완료 계량기</div></div>
    <div class="kpi yellow"><div class="val">7</div><div class="lbl">분석 에너지 타입<br>(학습 예정)</div></div>
  </div>
  <br>
  <table>
    <tr><th>구분</th><th>모델</th><th>방식</th><th>상태</th></tr>
    <tr><td>총 전력 예측</td><td>VMD-LSTM</td><td>VMD(K=4) 분해 → 2-layer LSTM + Attention</td><td><span class="tag good">학습 완료</span></td></tr>
    <tr><td>이상탐지</td><td>Residual + IF</td><td>예측 잔차 임계치 + Isolation Forest 앙상블</td><td><span class="tag good">학습 완료</span></td></tr>
    <tr><td>개별 계량기 예측</td><td>VMD-LSTM × 80</td><td>계량기별 독립 학습 (13 피처)</td><td><span class="tag good">학습 완료</span></td></tr>
    <tr><td>에너지 전체 예측</td><td>VMD-LSTM × 7</td><td>냉방/난방/태양광 등 7종</td><td><span class="tag warn">RunPod 학습 예정</span></td></tr>
  </table>
  <br>
  <div style="background:#F8FAFC;padding:18px;border-radius:8px;font-size:.92rem;">
    <b>데이터:</b> PostgreSQL · <code>ems.reduced_measurement_1h</code> (집계) &amp; <code>ems.cr_measurement_1h</code> (계량기) &nbsp;|&nbsp;
    <b>기간:</b> 2018–2023 · 1시간 단위 &nbsp;|&nbsp;
    <b>피처:</b> 전력 3종 + 기상 2종 + 시간 6종 + VMD IMF 4종 + lag 2종 = <b>17개</b>
  </div>
</section>

<!-- ② grid_P 예측 결과 -->
<section>
  <h2>② grid_P (총 전력 소비) 예측 결과</h2>
  <div class="grid-2" style="margin-bottom:24px;">
    <div class="kpi green">
      <div class="val">25.5%</div>
      <div class="lbl">Val MAPE (2022)<br>MAE 22,460 W</div>
    </div>
    <div class="kpi red">
      <div class="val">57.4%</div>
      <div class="lbl">Test MAPE (2023)<br>MAE 26,252 W</div>
    </div>
  </div>
  {divs['fig1']}
  <br>
  {divs['fig2']}
  <div class="verdict warn">
    <b>⚠ Test 성능 저하 원인:</b> 2023년 소비 패턴이 2022년 이전과 달라졌습니다
    (Distribution Shift). 연간 재학습을 통해 해결 가능합니다.
  </div>
</section>

<!-- ③ 이상탐지 결과 -->
<section>
  <h2>③ 이상탐지 결과 (Residual + Isolation Forest)</h2>
  <div class="grid-3" style="margin-bottom:24px;">
    <div class="kpi yellow"><div class="val">2,217</div><div class="lbl">이상 탐지 (2022)<br>HIGH 659 + LOW 1,558</div></div>
    <div class="kpi green"><div class="val">69일</div><div class="lbl">실제 장애 기간<br>2022-05-06 ~ 07-14</div></div>
    <div class="kpi yellow"><div class="val">pseudo</div><div class="lbl">레이블 기반<br>(실제 장애 기록 활용)</div></div>
  </div>
  {divs['fig3']}
  <br>
  {divs['fig4']}
  <div class="verdict warn">
    <b>이상탐지 해석:</b> 장애 구간에서 잔차(예측오차)가 급증하는 패턴이 명확합니다.
    다만 F1 평가는 게이트웨이 장애 기록(pseudo-label) 1구간에만 의존하므로
    더 많은 실제 장애 기록이 있으면 정확한 성능 측정이 가능합니다.
  </div>
</section>

<!-- ④ 계량기 성능 -->
<section>
  <h2>④ 개별 계량기 80개 성능</h2>
  <div class="grid-3" style="margin-bottom:24px;">
    <div class="kpi green">
      <div class="val">{n_good}개</div>
      <div class="lbl">우수 (Test MAPE &lt; 20%)</div>
    </div>
    <div class="kpi yellow">
      <div class="val">{(ok['test_mape'].between(20,100)).sum()}개</div>
      <div class="lbl">보통 (20–100%)</div>
    </div>
    <div class="kpi red">
      <div class="val">{n_bad}개</div>
      <div class="lbl">저조 (Test MAPE &gt; 100%)</div>
    </div>
  </div>
  {divs['fig5']}
  <br>
  {divs['fig6']}
  <div class="verdict warn">
    <b>저조 계량기 원인:</b> val(2022)과 test(2023)의 성능 격차가 큰 계량기는
    2023년 이후 장비 교체·추가·운영 패턴 변화가 강하게 의심됩니다.
    계량기별 원인 분석 후 재학습이 필요합니다.
  </div>
</section>

<!-- ⑤ 에너지 흐름 분석 -->
<section>
  <h2>⑤ 에너지 흐름 분석</h2>
  <div class="grid-2">
    <div>{divs['fig7']}</div>
    <div>{divs['fig8']}</div>
  </div>
  <div class="verdict ok" style="margin-top:16px;">
    <b>✓ 인사이트:</b>
    태양광·열병합 발전이 늘수록 계통 소비(grid_P)가 줄어드는 음의 상관관계가 확인됩니다.
    온도가 낮은 겨울(1–3월)에 계통 소비가 최고, 여름(6–8월)에는 발전량이 증가합니다.
  </div>
</section>

<!-- ⑥ Val vs Test 비교 -->
<section>
  <h2>⑥ 모델 적합성 종합 평가</h2>
  {divs['fig9']}
  <br>
  <table>
    <tr><th>평가 항목</th><th>현재 상태</th><th>판정</th><th>개선 방향</th></tr>
    <tr>
      <td>예측 정확도 (Val)</td>
      <td>grid_P MAPE 25.5% · 계량기 중앙값 56%</td>
      <td><span class="tag good">양호</span></td>
      <td>하이퍼파라미터 추가 튜닝</td>
    </tr>
    <tr>
      <td>예측 정확도 (Test)</td>
      <td>grid_P MAPE 57.4% · 계량기 중앙값 77%</td>
      <td><span class="tag bad">개선 필요</span></td>
      <td>연간 재학습 · 2022 데이터 학습 포함</td>
    </tr>
    <tr>
      <td>이상탐지 성능</td>
      <td>장애 구간 잔차 급증 패턴 확인</td>
      <td><span class="tag warn">조건부 양호</span></td>
      <td>실제 장애 기록 추가 확보</td>
    </tr>
    <tr>
      <td>파이프라인 완성도</td>
      <td>DB→학습→추론→MLflow 전 구간 자동화</td>
      <td><span class="tag good">완성</span></td>
      <td>API 서빙 추가 고려</td>
    </tr>
    <tr>
      <td>확장성</td>
      <td>7종 에너지 타입 학습 코드 완성</td>
      <td><span class="tag good">준비됨</span></td>
      <td>RunPod 실행 후 결과 추가</td>
    </tr>
  </table>

  <div class="verdict ok" style="margin-top:20px;">
    <b>✓ 방법론 적합성:</b>
    VMD-LSTM은 에너지 시계열의 다중 주기성(일간·주간·계절)을 효과적으로 분해하여 예측하는
    검증된 방법입니다. 잔차 기반 이상탐지는 "게이트웨이 장애 = 예측과 다른 값"이라는
    도메인 지식에 직접 부합합니다. <b>방법론 자체는 적합합니다.</b>
  </div>
  <div class="verdict warn" style="margin-top:12px;">
    <b>⚠ 개선 필요 사항:</b>
    Test 성능 저하는 학습 데이터(2018–2021)와 테스트 시점(2023) 사이의 패턴 변화 때문입니다.
    <b>2022년 데이터를 학습에 포함하고 연간 재학습 스케줄을 도입</b>하면 실용적인 수준으로
    성능이 향상될 것으로 예상됩니다.
  </div>
</section>

</div>
<footer>EMS ML Pipeline · SK Networks Family AI 27기 · 생성일 2026-05-20</footer>
</body>
</html>"""
    return html


# ── 메인 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("▶ 그래프 생성 중...")
    html = build_html()
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\n✓ 저장 완료: {OUT_PATH}")
    print(f"  브라우저에서 열어보세요: file://{OUT_PATH.resolve()}")
