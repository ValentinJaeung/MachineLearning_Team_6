"""
One place for every setting the 16 steps share.

The variable groups below are the ones the leakage audit checks.
Keeping them here, rather than re-listing columns inside each step, is what
makes the leakage rule checkable: a reviewer can read ONE list to see what the
model is allowed to see.

  A1_INPUT    green  - known at booking time          -> model input candidates
  A2_ID       green  - identifiers                    -> joins and splits only
  B_OUTCOME   red    - what happened                  -> builds the target only
  C_POSTHOC   yellow - post-hoc detail                -> builds "day state" only

B and C never enter a model matrix. A2 never enters a model matrix.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

SEED = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"             # step 1  - BTS monthly parquet
CLEAN_DIR = DATA_DIR / "clean"         # step 2  - times, keys, y_class
SCHEDULE_DIR = DATA_DIR / "schedule"   # step 3  - timetable-derived features
SCENARIO_DIR = DATA_DIR / "scenario"   # step 6  - airport-day type
DATASET_DIR = DATA_DIR / "dataset"     # step 7  - one table, split by year
PREP_DIR = DATA_DIR / "prep"           # step 9  - two design matrices
MODEL_DIR = DATA_DIR / "model"         # steps 11-14 - fitted models

RESULTS_DIR = PROJECT_ROOT / "results"   # small csv/json, committed to git
FIGURE_DIR = PROJECT_ROOT / "figures"    # png, committed to git

# --------------------------------------------------------------------------- #
# Period and split
# --------------------------------------------------------------------------- #

YEARS = list(range(2021, 2027))        # 2021 is the Y-1 source for 2022 only
MONTHS = (6, 7, 8)                     # summer only: one axis of variation

# Features that look back (step 3's sched_pad) read year Y-1 only, so 2021 is
# downloaded but never split into train/val/test.
SPLIT_YEARS = {"train": [2022, 2023, 2024], "val": [2025], "test": [2026]}
SPLITS = ["train", "val", "test"]

# --------------------------------------------------------------------------- #
# Target: ten arrival outcome bins
# --------------------------------------------------------------------------- #

# upper edges in minutes of arrival delay; the ninth bin (300+) is open
EDGES = [0, 15, 30, 60, 90, 120, 180, 300]
N_DELAY_BINS = len(EDGES) + 1          # 9 delay bins, classes 0..8
NOARRIVE = N_DELAY_BINS                # class 9: cancelled, or diverted and
N_CLASS = N_DELAY_BINS + 1             #          never reached the destination

CLASS_NAMES = [
    "≤0분(조기·정시)", "0–15분", "15–30분", "30–60분", "60–90분",
    "90–120분", "120–180분", "180–300분", "300분+", "미도착(결항·회항실패)",
]

# --------------------------------------------------------------------------- #
# Variable groups
# --------------------------------------------------------------------------- #

# A-1: everything a passenger can see on the booking screen.
A1_INPUT = [
    "Month",
    "DayOfWeek",
    "CRSDepTime",
    "CRSArrTime",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Distance",
    "CRSElapsedTime",
]

# A-2: identifiers. Used to join and to split. Never model inputs.
A2_ID = ["FlightDate", "Flight_Number_Reporting_Airline", "Tail_Number"]

# B: the outcome. Builds y_class. Never a model input.
B_OUTCOME = ["ArrDelay", "Cancelled", "Diverted", "DivReachedDest", "DivArrDelay"]

# C: post-hoc detail. Builds the airport-day state and the scenario type.
# Never a model input.
C_POSTHOC = [
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "CancellationCode", "DepDelay", "DepTime", "ArrTime",
    "ActualElapsedTime", "AirTime", "TaxiOut", "TaxiIn",
]

CAUSE_COLUMNS = [
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay",
]

# --------------------------------------------------------------------------- #
# Model feature groups
# --------------------------------------------------------------------------- #

# The nine values the ticket itself shows, derived from A-1 only.
BASE_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DayOfWeek",
    "Month",
    "crs_dep_hour",
    "crs_arr_hour",
    "Distance",
    "CRSElapsedTime",
]

# Three features derived from the published timetable and last year's public
# record (step 3). Each carries information the nine above cannot rebuild.
SCHEDULE_FEATURES = [
    "sched_pad",                 # scheduled block time minus last year's actual
    "dest_arr_load",             # arrivals scheduled into this airport-hour
    "days_to_nearest_holiday",   # signed days, clipped to +/-21
]

# The twelve inputs the reported model uses.
MODEL_FEATURES = BASE_FEATURES + SCHEDULE_FEATURES

# The two scenario columns, used by model B in step 14. Not known at booking.
SCENARIO_FEATURES = ["scen_origin", "scen_dest"]

# --------------------------------------------------------------------------- #
# Scenario types
# --------------------------------------------------------------------------- #

SCENARIO_NAMES = ["정상일", "항공사귀책일", "연결편붕괴일", "공역혼잡일", "날씨일"]
SCENARIO_MIN_FLIGHTS = 20     # a day with fewer flights cannot characterise an airport

# --------------------------------------------------------------------------- #
# Other settings decided by EDA (step 8) and read back by later steps
# --------------------------------------------------------------------------- #

EDA_SETTINGS_PATH = RESULTS_DIR / "step08_settings.json"


def set_seeds(seed: int = SEED) -> None:
    """Seed every library the pipeline uses, so a re-run reproduces the run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for path in (
        CLEAN_DIR, SCHEDULE_DIR, SCENARIO_DIR,
        DATASET_DIR, PREP_DIR, MODEL_DIR, RESULTS_DIR, FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
