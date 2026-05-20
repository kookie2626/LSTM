"""DB에서 모델 학습용 데이터를 로드하는 공통 모듈.

팀 통일 기준 (2026-05-18 확정):
  - 해상도  : 1시간 (ems.reduced_measurement_1h)
  - 타겟    : grid_P (electricity/total/P, W 단위)
  - 피처    : grid_P, pv_P, chp_P, Ta, Igm + 시간 sin/cos 6개 = 11개
  - 분할    : train 2018~2020 / val 2021 / test 2022~2023
  - 결측    : 0으로 채움
  - 게이트웨이 장애 4개 구간 학습 데이터에서 제외
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

CONNECT_KWARGS = {
    "host":     os.environ["DB_HOST"],
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "dbname":   os.environ["DB_NAME"],
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

# ── 데이터 분할 구간 ─────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01"
VAL_END     = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2023-12-31"

# ── 게이트웨이 장애 구간 (학습 데이터에서 제외) ─────────────────────────────
GATEWAY_FAILURES = [
    ("2020-02-13", "2020-03-06"),   # Workshop gateway #1
    ("2020-08-20", "2020-09-17"),   # Emission lab gateway
    ("2021-11-15", "2021-12-10"),   # Distribution gateway
    ("2022-05-06", "2022-07-14"),   # Workshop gateway #2
]

# ── 피처 / 타겟 ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "grid_P", "pv_P", "chp_P", "Ta", "Igm",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
]
TARGET_COL = "grid_P"


def _query(sql: str) -> pd.DataFrame:
    with psycopg.connect(**CONNECT_KWARGS) as conn:
        return pd.read_sql(sql, conn)


def load_raw() -> pd.DataFrame:
    """전력(grid/pv/chp) + 기상(Ta/Igm) 데이터를 피벗하여 반환."""
    sql = """
        SELECT ts, category, subcategory, measurement, value
        FROM ems.reduced_measurement_1h
        WHERE
            (category = 'electricity' AND subcategory IN ('total', 'pv', 'chp')
             AND measurement = 'P')
            OR
            (category = 'weather' AND subcategory = 'weather'
             AND measurement IN ('Ta', 'Igm'))
        ORDER BY ts
    """
    df_long = _query(sql)
    df_long["col"] = df_long["subcategory"] + "_" + df_long["measurement"]
    df = df_long.pivot_table(
        index="ts", columns="col", values="value", aggfunc="first"
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    df = df.rename(columns={
        "total_P":   "grid_P",
        "pv_P":      "pv_P",
        "chp_P":     "chp_P",
        "weather_Ta":  "Ta",
        "weather_Igm": "Igm",
    })
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 sin/cos 피처 추가 + 결측값 0 처리."""
    ts = df.index

    df["hour_sin"]  = np.sin(2 * np.pi * ts.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * ts.hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * ts.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * ts.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

    # 결측값 0으로 채움 (팀 기준)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


def _mask_gateway_failures(df: pd.DataFrame) -> pd.DataFrame:
    """게이트웨이 장애 구간 행 제거."""
    mask = pd.Series(False, index=df.index)
    for start, end in GATEWAY_FAILURES:
        mask |= (df.index >= start) & (df.index <= end)
    return df[~mask]


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train / val / test 분할 반환.

    - train: 게이트웨이 장애 구간 제외
    - val / test: 제외 없음 (평가 구간은 그대로 사용)
    """
    train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
    train = _mask_gateway_failures(train)

    val  = df[(df.index >= VAL_START)  & (df.index <= VAL_END)]
    test = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    return train, val, test
