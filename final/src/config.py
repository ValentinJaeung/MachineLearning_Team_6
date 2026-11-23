"""모든 상수를 한 곳에 모아 둔다.

이 파일 밖에서 숫자를 직접 적는 코드는 없다. 경계값·연도·시드를 바꾸고 싶으면
여기만 고치면 모든 단계가 같은 값을 읽는다.

재현성
------
모든 난수는 SEED 하나에서 나온다. 각 스크립트는 맨 앞에서 set_seeds() 를 부른다.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# 경로
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent          # final/
DATA_DIR = ROOT / "data"                               # 중간 산출물
RESULTS_DIR = ROOT / "results"                         # 보고서에 쓰는 수치 (csv)
FIGURE_DIR = ROOT / "figures"                          # 그림

LABEL_DIR = DATA_DIR / "label"                         # step01: 정제 + 10구간 라벨
FEATURE_DIR = DATA_DIR / "feature"                     # step02: 파생 변수 2개
SCEN_DIR = DATA_DIR / "scen"                           # step03: 유형 변수 2열
PREP_DIR = DATA_DIR / "prep"                           # step05: 설계행렬
MODEL_DIR = DATA_DIR / "model"                         # step06~08: 모델·보정기

# BTS 원자료. 이미 내려받아 둔 것을 쓴다(이 프로젝트에서 다시 받지 않는다).
# 다른 곳에 있으면 환경변수 FINAL_RAW_DIR 로 덮어쓸 수 있다.
RAW_DIR = Path(os.environ.get("FINAL_RAW_DIR", ROOT.parent / "data" / "raw"))

ALL_DIRS = [DATA_DIR, RESULTS_DIR, FIGURE_DIR,
            LABEL_DIR, FEATURE_DIR, SCEN_DIR, PREP_DIR, MODEL_DIR]

# --------------------------------------------------------------------------- #
# 재현성
# --------------------------------------------------------------------------- #
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """파이썬·numpy·torch 의 난수를 모두 고정한다."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 데이터 범위와 분할
# --------------------------------------------------------------------------- #
MONTHS = (6, 7, 8)                       # 여름만 — 계절마다 지연의 원인이 다르다
TRAIN_YEARS = (2022, 2023, 2024)
VAL_YEAR = 2025
TEST_YEAR = 2026
YEARS = TRAIN_YEARS + (VAL_YEAR, TEST_YEAR)

SPLITS = ("train", "val", "test")
SPLIT_YEARS = {"train": TRAIN_YEARS, "val": (VAL_YEAR,), "test": (TEST_YEAR,)}

# --------------------------------------------------------------------------- #
# 정답: 10구간
# --------------------------------------------------------------------------- #
# 경계 8개 → 지연 9칸(class 0..8) + 미도착 1칸(class 9) = 10 클래스
#   class 0 : ArrDelay <= 0          (조기·정시)
#   class 1 : 0  < d <= 15
#   class 2 : 15 < d <= 30
#   class 3 : 30 < d <= 60
#   class 4 : 60 < d <= 90
#   class 5 : 90 < d <= 120
#   class 6 : 120 < d <= 180
#   class 7 : 180 < d <= 300
#   class 8 : d > 300
#   class 9 : 미도착 (결항, 또는 회항 후 목적지에 끝내 닿지 못한 편)
EDGES = [0, 15, 30, 60, 90, 120, 180, 300]
N_DELAY_BINS = len(EDGES) + 1            # 9
NOARRIVE = N_DELAY_BINS                  # 9
N_CLASS = N_DELAY_BINS + 1               # 10

CLASS_NAMES = [
    "≤0분(조기·정시)", "0–15분", "15–30분", "30–60분", "60–90분",
    "90–120분", "120–180분", "180–300분", "300분+", "미도착",
]

# --------------------------------------------------------------------------- #
# 유형 변수 (그날의 유형)
# --------------------------------------------------------------------------- #
SCENARIO_NAMES = ["정상일", "항공사귀책일", "연결편붕괴일", "공역혼잡일", "날씨일"]

# '실패'의 정의: 15분을 넘는 지연(= class 2 이상) 또는 미도착.
# 업계에서 쓰는 15분 기준과 같은 자리에서 끊는다.
FAIL_FROM_CLASS = 2

# 원인별 지연 분(分). SecurityDelay 는 양이 미미해 쓰지 않는다.
CAUSES = ["carrier", "late_aircraft", "nas", "weather"]
CAUSE_COLUMN = {
    "carrier": "CarrierDelay",
    "late_aircraft": "LateAircraftDelay",
    "nas": "NASDelay",
    "weather": "WeatherDelay",
}

# 그 (공항, 날짜) 에 이 편수 미만이면 유형을 매기지 않는다.
# 몇 편으로 그날을 규정할 수 없기 때문이고, 라벨하지 못한 행은 버린다.
SCENARIO_MIN_FLIGHTS = 20

# --------------------------------------------------------------------------- #
# 전처리
# --------------------------------------------------------------------------- #
# 공항 원-핫: 학습 연도 운항이 이 편수 미만인 공항만 '기타' 로 묶는다.
#
# 처음에는 "누적 운항 95% 지점에서 끊기" 를 쓰려 했는데, step03 이 20편 미만
# 공항-날짜를 버린 뒤라 공항이 139개만 남아 있고, 95% 지점은 **운항 2만 편짜리
# 공항까지** '기타' 로 묶어 버렸다. 묶는 목적은 열 하나가 학습 편수 부족으로
# 잡음이 되는 것을 막는 것이므로, 기준을 누적 비율이 아니라 **최소 운항 편수**
# 로 바꾼다. 누적 곡선은 results/step04_airport_coverage.csv 에 그대로 남겨
# 두 기준을 비교할 수 있게 한다.
#
# 2000편의 근거: 10구간 확률을 추정하려면 가장 드문 칸(300분+, 약 0.6%)이
# 열 안에서 열 번 남짓은 나와야 한다. 2000편이면 약 12편이다.
MIN_AIRPORT_FLIGHTS = 2000
AIRPORT_COVERAGE = 0.95        # 비교용으로만 기록한다 (선택에는 쓰지 않는다)
OTHER_LABEL = "기타"

# 공휴일까지 남은 일수는 이 값에서 자른다 (±일).
HOLIDAY_CLIP_DAYS = 21

# --------------------------------------------------------------------------- #
# 모델 탐색 범위 (한 번에 하나씩 바꾼다)
# --------------------------------------------------------------------------- #
LOGISTIC_C = [0.1, 1.0, 10.0]
LOGISTIC_MAX_ITER = 400

MLP_HIDDEN = [(256, 64), (512, 128, 32)]
MLP_LR = [3e-4, 1e-3]
MLP_BATCH = 8192
MLP_MAX_EPOCHS = 8
MLP_DROPOUT = 0.2

XGB_MAX_DEPTH = [3, 6, 10]
XGB_ETA = [0.05, 0.10]
XGB_ROUNDS = 400
XGB_EARLY_STOPPING = 40
XGB_BASE = {
    "objective": "multi:softprob",
    "num_class": N_CLASS,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# --------------------------------------------------------------------------- #
# 서비스
# --------------------------------------------------------------------------- #
# 요구 확률. 하나로 고정한다 — 검증과 서술을 단순하게 유지하기 위해서.
CONFIDENCE = 0.80

# 25칸 중 테스트 편수가 이 값 미만인 칸은 '판단 불가' 로 적는다.
MIN_CELL_FLIGHTS = 1000
