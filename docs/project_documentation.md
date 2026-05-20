# Project Documentation: Electricity Forecasting and Anomaly Detection

## 1. 프로젝트 개요

이 프로젝트의 목적은 두 가지입니다.

1. 전력 소비(`grid_P`)를 정확하게 예측하는 시계열 예측 모델 개발
2. 에너지 시스템에서 비정상적 이상 상태를 자동으로 탐지하는 이상 탐지 모델 개발

이를 위해 `PostgreSQL` DB에서 1시간 단위 전력 및 기상 데이터를 가져와, 딥러닝 기반 파이프라인으로 학습합니다.

## 2. 데이터 소스 및 전처리

### 2.1 데이터 소스

`data_loader.py`는 다음 데이터를 로드합니다.

- `grid_P`: 전체 전력 소비 (electricity total P)
- `pv_P`: 태양광 발전 전력
- `chp_P`: 발전기(코젠) 전력
- `Ta`: 기온
- `Igm`: 일사량

대상 테이블: `ems.reduced_measurement_1h` (종합), `ems.cr_measurement_1h` (개별 계량기)

### 2.2 피처

공통 피처는 다음과 같습니다.

- `grid_P`, `pv_P`, `chp_P`, `Ta`, `Igm`
- 시간 기반 사이클릭 특성
  - `hour_sin`, `hour_cos`
  - `dow_sin`, `dow_cos`
  - `month_sin`, `month_cos`

총 11개 기본 피처를 사용합니다. 개별 계량기 학습(`train_all_meters.py`)에서는 24h/48h/168h/336h lag 피처 4개를 추가해 13개로 확장합니다.

### 2.3 데이터 분할

시간 구간은 다음과 같습니다.

- 학습: 2018-01-01 ~ 2021-12-31
- 검증: 2022-01-01 ~ 2022-12-31
- 테스트: 2023-01-01 ~ 2023-12-31

### 2.4 결측값 처리

결측값은 모두 `0`으로 채워집니다. 이 기준은 팀 내부 정책에 맞춘 처리 방식입니다.

### 2.5 이상 데이터 제거

학습 데이터에서 게이트웨이 장애 구간은 제거합니다. 제거 구간은 다음과 같습니다.

- 2020-02-13 ~ 2020-03-06
- 2020-08-20 ~ 2020-09-17
- 2021-11-15 ~ 2021-12-10
- 2022-05-06 ~ 2022-07-14

검증과 테스트 구간은 그대로 두고, 학습 데이터에서만 장애 구간을 제외합니다.

---

## 3. 모델 1: VMD-LSTM 예측 모델 (`scripts/train/train_vmd_lstm.py`)

### 3.1 목표

`grid_P` 전력 소비를 시계열 예측합니다. VMD(Variational Mode Decomposition)를 사용해 신호를 분해하고, LSTM 기반 모델로 미래 값을 예측합니다.

### 3.2 주요 방법

1. 데이터 로드 및 기본 피처 생성
2. `VMD`로 `grid_P` 시계열을 4개의 IMF 성분으로 분해
3. 각 IMF에 대해 1시간 지연(lag) 피처를 추가하여 입력 피처 수를 `11 → 15`로 확장
4. 추가로 1주(168h), 2주(336h) 지연 피처를 생성하여 최종 `17`개로 확장
5. `MinMaxScaler`로 입력과 타겟을 정규화
6. `SEQ_LEN=24`로 시퀀스 생성
7. LSTM + attention 모델 학습
8. 검증 손실로 early stopping
9. 최종 모델 및 스케일러 저장

### 3.3 모델 구조

`LSTMForecaster`는 다음을 포함합니다. (정의는 `inference.py`에서 공유)

- 2-layer LSTM
- Attention pooling
- Fully connected 출력층 (hidden → 128 → 1) — `train_all_meters.py`는 hidden → 64 → 1 사용
- `HuberLoss` 손실 함수

### 3.4 하이퍼파라미터

- VMD: `K=4`, `alpha=2000`
- 시퀀스 길이: `24`
- 배치 크기: `256`
- 학습률: `1e-3`
- 히든 차원: `192`
- 레이어 수: `2`
- 드롭아웃: `0.2`
- epochs: `100`
- early stopping patience: `15`

### 3.5 주요 결과

MLflow에 기록된 대표 결과는 다음과 같습니다.

- 검증 MAE: 약 `22,460` W
- 검증 RMSE: 약 `43,827` W
- 검증 MAPE: 약 `25.50%`
- 테스트 MAE: 약 `26,252` W
- 테스트 RMSE: 약 `45,761` W
- 테스트 MAPE: 약 `57.42%`

---

## 4. 모델 2: 잔차 기반 이상 탐지 (`scripts/train/train_anomaly_residual.py`) ✅ 현행

### 4.1 목표

VMD-LSTM 예측 잔차를 주요 신호로 활용하여 게이트웨이 장애 구간을 탐지합니다.

### 4.2 주요 방법

1. VMD-LSTM 예측 결과에서 잔차 `|actual - predicted|` 계산
2. 학습 잔차의 `mean + k * std`로 이상 임계치 설정
3. Isolation Forest로 다변량 컨텍스트(원본 피처 기준) 보완
4. val pseudo-label로 k 값 자동 최적화 (탐색 범위 k ∈ [0.5, 4.0])
5. 두 신호 합산으로 최종 이상 레벨 결정

### 4.3 이상 탐지 논리

- 잔차 임계치 초과 → 잔차 플래그
- IF 예측 결과 `-1` → IF 플래그
- 두 플래그 모두 → `HIGH`
- 하나만 → `LOW`
- 모두 정상 → `NORMAL`

### 4.4 LSTM-AE 방식과의 비교

| 항목 | LSTM-AE (구버전) | 잔차 기반 (현행) |
|:--|:--|:--|
| 이상 신호 | 재구성 오차 | 예측 잔차 |
| 게이트웨이 장애 적합성 | 낮음 | 높음 |
| 임계치 최적화 | 고정 k=2.0 | val F1으로 자동 탐색 |
| 검증 AUC | ~0.547 | 개선 기대 |

### 4.5 하이퍼파라미터

- IF contamination: `0.15`
- k 탐색 범위: `0.5 ~ 4.0` (0.1 간격)
- pseudo-label 구간: `2022-05-06 ~ 2022-07-14`

---

## 5. 모델 3: IF + LSTM-AE 이상 탐지 (`scripts/train/train_anomaly_ifae.py`) — 구버전

> ⚠️ 잔차 기반 방식(모델 2)으로 대체됨. 참조용으로 보존.

### 5.1 LSTM-AE 구조

`LSTMAutoencoder`는 다음을 포함합니다.

- 2-layer LSTM 인코더
- Attention 기반 시퀀스 풀링
- Latent bottleneck (hidden → 32)
- 2-layer LSTM 디코더
- 출력 재구성층

### 5.2 하이퍼파라미터

- 시퀀스 길이: `24`
- 배치 크기: `1024`
- 학습률: `1e-3`
- 히든 차원: `128`
- latent dim: `32`
- 드롭아웃: `0.2`
- epochs: `100`
- early stopping patience: `15`
- IF contamination: `0.15`
- threshold k: `2.0` (고정)

### 5.3 주요 결과

- 검증 AUC: 약 `0.547`
- 검증 F1: 약 `0.294`
- 검증 Precision: 약 `0.244`
- 검증 Recall: 약 `0.368`

---

## 6. 모델 4: 전체 계량기 일괄 학습 (`scripts/train/train_all_meters.py`)

### 6.1 목표

`ems.cr_measurement_1h`에서 81개 개별 계량기를 순차 학습합니다. 각 계량기마다 VMD-LSTM(예측) + 잔차 IF(이상 탐지) 모델을 독립적으로 학습합니다.

### 6.2 특징

- VMD 생략 (속도 우선): lag 피처(24h/48h/168h/336h)로 대체
- 학습 데이터 2000행 미만 계량기 자동 스킵
- k 값 val F1으로 자동 최적화
- 모델 저장 경로: `outputs/models/meters/{meter_urn}/`

### 6.3 실행 옵션

```bash
python scripts/train/train_all_meters.py
python scripts/train/train_all_meters.py --skip-existing   # 기존 모델 건너뜀
python scripts/train/train_all_meters.py --meter H1.Z10    # 단일 계량기
```

### 6.4 주요 결과 (`outputs/all_meters_results.csv`)

- 분석 계량기 수: 80개
- 전체 평균 Test MAE: 7,748 W
- 중앙값 Test MAE: 1,224 W
- Test MAE 1,000 W 미만 비율: 45% (36개)
- 평균 검증 F1: 0.335

상세 계량기별 성능은 `docs/detailed_metrics.md` 참조.

---

## 7. MLflow 사용

### 7.1 실험 구성

| 스크립트 | MLflow 실험명 |
|:--|:--|
| `scripts/train/train_vmd_lstm.py` | `SSA-IPSO-LSTM` |
| `scripts/train/train_anomaly_residual.py` | `Residual-IF-Anomaly` |
| `scripts/train/train_anomaly_ifae.py` | `LSTM-AE-Anomaly` |
| `scripts/train/train_all_meters.py` | `All-Meters` |

### 7.2 기록 항목

- 파라미터: 모델 하이퍼파라미터, 스케일러, sequence 길이 등
- 메트릭: 손실, MAE, RMSE, WAPE, MAPE, F1, AUC, Precision, Recall
- 아티팩트: 학습된 모델 파일, 스케일러 파일

### 7.3 저장 위치

- 학습 스크립트는 **로컬** `mlruns/` 에 기록 (스크립트 위치 기준 자동 설정)
- `scripts/report/upload_to_mlflow.py` 실행 시 원격 서버 `http://121.134.46.24:5000` 으로 푸시
- 원격 업로드 실험명: `VMD-LSTM-Grid`, `Residual-IF-Anomaly`

---

## 8. 코드베이스 구성

```
ML/
├── data_loader.py              # DB 연결, 데이터 로드, 피처 생성, 분할 (공통)
├── inference.py                # 모델 정의 + 예측/이상탐지 추론 함수 (공통)
├── requirements_runpod.txt
├── scripts/
│   ├── train/
│   │   ├── train_vmd_lstm.py           # grid_P VMD-LSTM 예측 학습
│   │   ├── train_anomaly_residual.py   # 잔차+IF 이상 탐지 학습 (현행)
│   │   ├── train_anomaly_ifae.py       # LSTM-AE+IF 이상 탐지 학습 (구버전)
│   │   └── train_all_meters.py         # 81개 계량기 일괄 학습
│   ├── report/
│   │   ├── upload_to_mlflow.py         # 모델을 원격 MLflow 서버에 업로드
│   │   ├── generate_detailed_report.py # 계량기별 상세 HTML 리포트 생성
│   │   ├── generate_html_report.py     # 팀 요약 HTML 리포트 생성
│   │   ├── generate_documentation_pdf.py
│   │   └── update_wape.py              # MLflow에 WAPE 메트릭 소급 업데이트
│   └── plot/
│       ├── visualize.py                # 예측/이상탐지 결과 시각화 (CLI)
│       ├── plot_db.py                  # DB 원시 데이터 탐색 플롯
│       ├── plot_distributions.py       # 계량기 분포 시각화
│       ├── plot_exemplars.py           # 대표 계량기 예시 플롯
│       └── plot_h1_ze20.py             # H1.ZE20 계량기 개별 분석
├── docs/
│   ├── project_documentation.md
│   ├── detailed_metrics.md             # 계량기별 성능 Best/Worst 요약
│   └── project_documentation.pdf
└── outputs/
    ├── models/
    │   ├── vmd_lstm_grid_electricity.pt
    │   ├── vmd_lstm_scaler.pkl
    │   ├── residual_threshold.pkl
    │   ├── residual_iforest.pkl
    │   ├── residual_if_scaler.pkl
    │   ├── anomaly_lstmae.pt           # 구버전 (LSTM-AE)
    │   ├── anomaly_iforest.pkl
    │   ├── anomaly_scaler.pkl
    │   └── meters/{meter_urn}/         # 계량기별 독립 모델
    ├── all_meters_results.csv
    ├── all_meters_summary.csv
    ├── all_meters_detailed_report.html
    └── team_report.html
```

---

## 9. 실행 방법

```bash
pip install -r requirements_runpod.txt

# 프로젝트 루트가 /workspace/project/ML 인 경우 (RunPod 기준)
export PYTHONPATH=/workspace

# 예측 모델 학습
python scripts/train/train_vmd_lstm.py

# 이상 탐지 학습 (잔차 기반 — 현행, VMD-LSTM 학습 완료 후 실행)
python scripts/train/train_anomaly_residual.py

# 전체 계량기 일괄 학습
python scripts/train/train_all_meters.py

# 추론
python inference.py --start 2022-01-01 --end 2022-12-31
python inference.py --mode forecast --start 2023-01-01 --end 2023-12-31
python inference.py --mode anomaly  --start 2022-05-01 --end 2022-08-01

# 시각화
python scripts/plot/visualize.py --start 2022-01-01 --end 2022-12-31

# 원격 MLflow 업로드
python scripts/report/upload_to_mlflow.py
```

> **주의**: `from project.ML.*` 절대 경로 임포트를 사용하므로 `PYTHONPATH`에 프로젝트 상위 디렉토리를 반드시 추가해야 합니다. CPU 환경에서는 epoch당 수 분 소요 — GPU(RunPod) 사용 권장.

---

## 10. 권장 개선 방향

### 10.1 예측 모델

- 테스트 MAPE가 높으므로 추후 개선 필요 (val 25% → test 57%)
- VMD 분해 구성을 재검토하거나 더 많은 시퀀스 길이, 추가 피처 실험
- 정규화 정책과 학습/평가 분포 차이 원인 파악

### 10.2 이상 탐지 모델

- pseudo-label 정의를 확장하여 더 많은 이상 사례 확보 (현재 val 구간 1개만 존재)
- 임계치 k 탐색 범위와 IF contamination을 계량기 특성별로 조정

### 10.3 MLflow

- 파일 스토어(`mlruns/`) 대신 원격 서버(`http://121.134.46.24:5000`) 단일 사용으로 통일
- 중요한 Run에 `mlflow.log_model`을 추가해 모델 레지스트리 등록 준비

---

## 11. 요약

이 프로젝트는 에너지 시스템에서 전력 소비 예측과 장애 기반 이상 탐지를 동시에 다루는 데이터 사이언스 파이프라인입니다. 종합 전력(`grid_P`)에 대해 VMD-LSTM 예측과 잔차 기반 IF 이상 탐지를 수행하며, 동일 파이프라인을 81개 개별 계량기에 일괄 적용합니다. 모델 정의는 `inference.py`에서 공유하고, 공통 데이터 처리는 `data_loader.py`에서 담당합니다.
