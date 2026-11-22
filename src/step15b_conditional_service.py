"""Step 15b - the service as it actually answers: conditional on the day type.

Reads  data/prep/ (step 9), data/model/xgboost_scenario.json (step 14)
Writes results/step15b_conditional_metrics.csv
       results/step15b_conditional_promise.csv
       results/step15b_promise_by_type.csv

The model the service ships takes the twelve booking-time inputs AND the kind
of day the passenger assumes. Its answer is therefore a conditional one:

    "IF the day is of type s, you arrive within X minutes with probability p"

So every number here is measured the same way the answer is given - with the
assumed type equal to the day's actual type. That is not a favourable choice,
it is the only one that matches the claim: a forecast that says "if it rains"
is checked on the days it rained.

The calibrator is fitted on val 2025 the same way, and test 2026 is opened
once.

Usage:  python src/step15b_conditional_service.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common
import config

LEVELS = [0.80, 0.90, 0.95]


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def promise_table(probs: np.ndarray, cls: np.ndarray, delay: np.ndarray,
                  label: str) -> list[dict]:
    """For each promised level: how often the promise was actually kept."""
    cdf = probs.cumsum(axis=1)[:, :len(config.EDGES)]
    edges = np.asarray(config.EDGES)
    arrived = cls < config.NOARRIVE
    rows = []
    for level in LEVELS:
        reach = cdf >= level
        has = reach.any(axis=1)
        first = np.where(has, reach.argmax(axis=1), len(config.EDGES) - 1)
        promised = edges[first]
        kept = arrived & (delay <= promised)
        rows.append({
            "구분": label,
            "약속 확률": level,
            "n": int(has.sum()),
            "약속 가능 비율": round(float(has.mean()), 4),
            "실제 이행률": round(float(kept[has].mean()), 4) if has.any() else np.nan,
            "오차(%p)": round(float(kept[has].mean() - level) * 100, 2) if has.any() else np.nan,
            "평균 보장 지연(분)": round(float(promised[has].mean()), 1),
        })
    return rows


def main() -> int:
    import xgboost as xgb

    config.set_seeds()
    config.ensure_dirs()

    say("loading val / test with the day type ...")
    val = common.load_split("val", with_scenario=True)
    test = common.load_split("test", with_scenario=True)

    booster = xgb.Booster()
    booster.load_model(config.MODEL_DIR / "xgboost_scenario.json")

    say("scoring val 2025 (assumed type = that day's type) ...")
    raw_val = booster.predict(
        xgb.DMatrix(val["X_xgb"], enable_categorical=True)).astype("float64")

    say("fitting the calibrator on val 2025 ...")
    calibrator = common.CdfCalibrator().fit(raw_val, val["cls"], val["delay"])

    say("scoring test 2026 — opened once ...")
    raw_test = booster.predict(
        xgb.DMatrix(test["X_xgb"], enable_categorical=True)).astype("float64")
    cal_test = calibrator.transform(raw_test)

    # ---------------------------------------------------------------- #
    metrics = []
    for stage, probs in (("보정 전", raw_test), ("보정 후", cal_test)):
        scores = common.evaluate(probs, test["cls"], test["delay"])
        metrics.append({"stage": stage, **common.flat(scores)})
    table = pd.DataFrame(metrics)
    table.to_csv(config.RESULTS_DIR / "step15b_conditional_metrics.csv",
                 index=False, encoding="utf-8-sig")
    print("\n=== test 2026 지표 (조건부 모델) ===")
    print(table.to_string(index=False))

    # ---------------------------------------------------------------- #
    rows = (promise_table(raw_test, test["cls"], test["delay"], "보정 전")
            + promise_table(cal_test, test["cls"], test["delay"], "보정 후"))
    promise = pd.DataFrame(rows)
    promise.to_csv(config.RESULTS_DIR / "step15b_conditional_promise.csv",
                   index=False, encoding="utf-8-sig")
    print("\n=== 약속 이행률 (가정한 유형 = 그날의 실제 유형) ===")
    print(promise.to_string(index=False))

    # ---------------------------------------------------------------- #
    # the same promise, split by the type the passenger assumed
    # ---------------------------------------------------------------- #
    origin_type = test["X_xgb"]["scen_origin"].astype(int).to_numpy()
    by_type = []
    for s, name in enumerate(config.SCENARIO_NAMES):
        pick = origin_type == s
        if int(pick.sum()) < 1000:
            continue
        by_type += promise_table(cal_test[pick], test["cls"][pick],
                                 test["delay"][pick], name)
    typed = pd.DataFrame(by_type)
    typed.to_csv(config.RESULTS_DIR / "step15b_promise_by_type.csv",
                 index=False, encoding="utf-8-sig")
    print("\n=== 가정한 날 유형별 90% 약속 (보정 후) ===")
    nine = typed[typed["약속 확률"] == 0.90]
    print(nine.to_string(index=False))

    print("\n모든 수치는 '가정한 유형이 그날의 실제 유형일 때' 의 값이다 —")
    print("조건부 예보를 그 조건이 성립한 날에서 검증하는 것과 같다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
