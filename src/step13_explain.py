"""
Step 13 - what the model looks at, and what the passenger can actually change.

Reads  data/model/xgboost_scenario.json (step 14), data/prep/ (step 9)
Writes results/step13_shap.csv, results/step13_whatif_hour.csv
       figures/step13_shap.png, figures/step13_whatif_hour.png

Two questions, and each one is a line on the service screen.

1. WHICH INPUTS MOVE THE DISTRIBUTION
   TreeSHAP on the fitted booster, averaged over the ten classes. This is the
   explainability the course asks for, and it is also the check that the model
   is not keying on something silly.

2. WHAT IF THE SAME FLIGHT LEFT AT A DIFFERENT HOUR
   The same rows are re-scored with only crs_dep_hour (and the scheduled
   arrival hour that follows from it) changed. Everything else is held fixed,
   so the difference is attributable to the hour alone.

   This is not decoration: it IS the product. A passenger choosing a flight
   cannot change the airline's fleet or the weather, but they can take an
   earlier departure, and this measures exactly what that buys.

Scored on validation 2025. Test 2026 is not opened here.

Usage:  python src/step13_explain.py [--shap-rows N] [--whatif-rows N]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import common
import config
import plotting
from plotting import plt

LABELS = {
    "crs_dep_hour": "출발 시각", "crs_arr_hour": "도착 시각",
    "Reporting_Airline": "항공사", "Origin": "출발 공항", "Dest": "도착 공항",
    "DayOfWeek": "요일", "Month": "월", "Distance": "거리",
    "CRSElapsedTime": "예정 비행시간",
    "A_absorb_rate": "흡수율", "B1_turnaround_mean": "평균 쉬는 시간",
    "B2_day_seq_mean": "하루 중 순번", "B3_first_flight_rate": "첫 편 비율",
    "C_bad_day_rel_fail": "나쁜 날 상대 실패",
}
# what a passenger can still change at the moment of booking
CHANGEABLE = {"crs_dep_hour", "Reporting_Airline", "DayOfWeek"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shap-rows", type=int, default=60_000)
    parser.add_argument("--whatif-rows", type=int, default=120_000)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    booster = xgb.Booster()
    booster.load_model(config.MODEL_DIR / "xgboost_scenario.json")
    val = common.load_split("val", drop=drop, with_scenario=True)
    rng = np.random.default_rng(config.SEED)

    # ------------------------------------------------------------------ #
    # 1. TreeSHAP
    # ------------------------------------------------------------------ #
    index = rng.choice(len(val["cls"]), size=min(args.shap_rows, len(val["cls"])),
                       replace=False)
    frame = val["X_xgb"].iloc[index]
    matrix = xgb.DMatrix(frame, enable_categorical=True)
    columns = list(frame.columns)
    print(f"TreeSHAP on {len(frame):,} validation rows ...")

    # Exact TreeSHAP is one reason XGBoost was chosen, but some builds refuse
    # pred_contribs on categorical splits. Fall back to the booster's own gain
    # importance rather than losing the step, and say which one was used.
    try:
        shap = booster.predict(matrix, pred_contribs=True)
        shap = np.asarray(shap).reshape(len(frame), config.N_CLASS, -1)
        importance = np.abs(shap[:, :, :len(columns)]).mean(axis=(0, 1))
        method = "TreeSHAP (exact)"
    except Exception as exc:                                   # noqa: BLE001
        print(f"  pred_contribs unavailable ({type(exc).__name__}: {exc});"
              f" falling back to gain importance")
        gain = booster.get_score(importance_type="total_gain")
        total = sum(gain.values()) or 1.0
        importance = np.array([gain.get(c, 0.0) / total for c in columns])
        method = "total_gain (TreeSHAP 사용 불가)"
    print(f"  importance method: {method}")

    table = (pd.DataFrame({
        "feature": columns,
        "label": [LABELS.get(c, c) for c in columns],
        "mean_abs_shap": importance.round(6),
        "method": method,
        "passenger_can_change": [c in CHANGEABLE for c in columns],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    table.to_csv(config.RESULTS_DIR / "step13_shap.csv",
                 index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plot = table.iloc[::-1]
    colours = ["#2b8a3e" if c else "#adb5bd"
               for c in plot["passenger_can_change"]]
    ax.barh(plot["label"], plot["mean_abs_shap"], color=colours)
    ax.set_xlabel("평균 |SHAP| (10개 구간 평균)" if method.startswith("TreeSHAP")
                  else "총 gain 비중")
    ax.set_title("Step 13 — 분포를 움직이는 변수 (초록 = 예약 시점에 바꿀 수 있는 것)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step13_shap.png")
    plt.close(fig)

    print("\n=== 변수 중요도 (TreeSHAP) ===")
    print(table.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 2. what-if on the departure hour
    # ------------------------------------------------------------------ #
    index = rng.choice(len(val["cls"]),
                       size=min(args.whatif_rows, len(val["cls"])), replace=False)
    base = val["X_xgb"].iloc[index].reset_index(drop=True)
    # the scheduled block time is held fixed, so moving departure moves arrival
    block_hours = (base["crs_arr_hour"].to_numpy().astype(int)
                   - base["crs_dep_hour"].to_numpy().astype(int))

    rows = []
    print(f"\nwhat-if: re-scoring {len(base):,} flights at every departure hour ...")
    for hour in range(5, 24):
        trial = base.copy()
        trial["crs_dep_hour"] = hour
        trial["crs_arr_hour"] = np.clip(hour + block_hours, 0, 23)
        probs = booster.predict(xgb.DMatrix(trial, enable_categorical=True))
        promise = common.guaranteed_minutes(probs, 0.90)
        rows.append({
            "crs_dep_hour": hour,
            "ontime_rate": float(probs[:, 0].mean()),
            "noarrive_rate": float(probs[:, config.NOARRIVE].mean()),
            "guaranteed_90_min": float(np.nanmean(promise)),
            "no_guarantee_share": float(np.isnan(promise).mean()),
        })
        print(f"  {hour:02d}시  정시 {rows[-1]['ontime_rate']:.3f}  "
              f"90% 보장 +{rows[-1]['guaranteed_90_min']:.0f}분  "
              f"보장 불가 {rows[-1]['no_guarantee_share']:.1%}")

    whatif = pd.DataFrame(rows)
    whatif.to_csv(config.RESULTS_DIR / "step13_whatif_hour.csv",
                  index=False, encoding="utf-8-sig")

    fig, ax1 = plt.subplots(figsize=(9.6, 5.2))
    ax1.plot(whatif["crs_dep_hour"], whatif["ontime_rate"], "o-",
             color="#4c6ef5", label="정시·조기 확률")
    ax1.plot(whatif["crs_dep_hour"], whatif["noarrive_rate"], "^-",
             color="#c92a2a", label="미도착 확률")
    ax1.set_xlabel("예정 출발 시(時) — 다른 조건은 전부 그대로")
    ax1.set_ylabel("확률")
    ax2 = ax1.twinx()
    ax2.bar(whatif["crs_dep_hour"], whatif["guaranteed_90_min"],
            alpha=0.22, color="#e8590c", label="90% 보장 지연(분)")
    ax2.set_ylabel("90% 보장 지연(분)", color="#e8590c")
    ax2.grid(False)
    lines = ax1.get_lines() + [ax2.containers[0]]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left",
               fontsize=9)
    first, last = whatif.iloc[0], whatif.iloc[-1]
    ax1.set_title(
        f"Step 13 — 같은 편이 시각만 바뀌면: "
        f"{int(first['crs_dep_hour'])}시 정시 {first['ontime_rate']:.0%}"
        f"(보장 +{first['guaranteed_90_min']:.0f}분) → "
        f"{int(last['crs_dep_hour'])}시 {last['ontime_rate']:.0%}"
        f"(+{last['guaranteed_90_min']:.0f}분)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step13_whatif_hour.png")
    plt.close(fig)

    best = whatif.loc[whatif["guaranteed_90_min"].idxmin()]
    worst = whatif.loc[whatif["guaranteed_90_min"].idxmax()]
    print("\n=== what-if 요약 ===")
    print(f"  {int(best['crs_dep_hour'])}시 편의 90% 보장 지연 "
          f"+{best['guaranteed_90_min']:.0f}분 vs "
          f"{int(worst['crs_dep_hour'])}시 편 +{worst['guaranteed_90_min']:.0f}분 "
          f"— 차이 {worst['guaranteed_90_min'] - best['guaranteed_90_min']:.0f}분.")
    print("  '언제 늦느냐'는 시각이 정한다. 이 한 줄이 서비스 화면의 마지막 줄이다.")
    top = table.iloc[0]
    print(f"  SHAP 1위는 {top['label']}({top['feature']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
