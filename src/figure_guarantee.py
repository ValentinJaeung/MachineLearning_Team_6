"""The method in one picture: one flight's distribution becomes a promise.

Reads  data/prep/ (step 9), data/model/xgboost.json (step 11),
       data/model/production_calibrator.joblib (step 12)
Writes figures/method_guarantee.png
       results/figure_guarantee_example.csv

Every number the service shows comes out of ONE output - the ten-bin
distribution for a single flight. This figure draws that output twice for the
same flight:

  left   the ten bins themselves, which is all the model produces
  right  the same thing summed up, which is the only curve a passenger needs:
         "by how many minutes am I there with probability p?"

The three guarantee times (80 / 90 / 95%) are read off the right panel. The
curve stops short of 1.0 because of the tenth bin - never arrived - and a
flight whose curve never reaches the required level gets "보장 불가" rather
than an invented time. That is the whole method.

One flight, not a comparison. How the answer moves with the departure hour is
measured on 120,000 flights in step 13 with everything else held fixed; doing
it here with two hand-picked rows would be a weaker version of the same claim.

Usage:  python src/figure_guarantee.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

LEVELS = [0.80, 0.90, 0.95]
COLOURS = {0.80: "#2b8a3e", 0.90: "#1c7ed6", 0.95: "#c92a2a"}


def pick_example(meta: pd.DataFrame) -> int:
    """The median flight of the busiest route: an ordinary, real test row."""
    routes = meta["route_key"].to_numpy()
    busiest = pd.Series(routes).value_counts().index[0]
    rows = np.flatnonzero(routes == busiest)
    return int(rows[len(rows) // 2])


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    import xgboost as xgb_lib

    test = common.load_split("test", with_scenario=True)
    booster = xgb_lib.Booster()
    booster.load_model(config.MODEL_DIR / "xgboost_scenario.json")
    calibrator = joblib.load(config.MODEL_DIR / "conditional_calibrator.joblib")

    row = pick_example(test["meta"])
    raw = booster.predict(
        xgb_lib.DMatrix(test["X_xgb"].iloc[[row]], enable_categorical=True)
    ).astype("float64")
    probs = calibrator.transform(raw)[0]

    meta = test["meta"].iloc[row]
    xgb_row = test["X_xgb"].iloc[row]
    assumed = config.SCENARIO_NAMES[int(xgb_row["scen_origin"])]
    assumed_dest = config.SCENARIO_NAMES[int(xgb_row["scen_dest"])]
    title = (f"{meta['route_key']} · {str(meta['FlightDate'])[:10]} · "
             f"{int(xgb_row['crs_dep_hour']):02d}시 출발 · "
             f"가정: 출발지 {assumed} / 도착지 {assumed_dest}")

    edges = np.asarray(config.EDGES, dtype="float64")
    cdf = probs.cumsum()[:len(config.EDGES)]

    fig, (ax_bars, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5.4),
                                          gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: the model's actual output -------------------------------- #
    ax_bars.bar(range(config.N_CLASS), probs, color="#4c6ef5", alpha=0.85)
    ax_bars.bar([config.NOARRIVE], [probs[config.NOARRIVE]], color="#c92a2a",
                alpha=0.9)
    ax_bars.set_xticks(range(config.N_CLASS))
    ax_bars.set_xticklabels(config.CLASS_NAMES, rotation=55, ha="right",
                            fontsize=8)
    ax_bars.set_ylabel("확률")
    ax_bars.set_title("모델이 내놓는 것은 이 10칸뿐이다", fontsize=11)
    for i, v in enumerate(probs):
        if v >= 0.02:
            ax_bars.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=8)
    ax_bars.annotate(f"미도착 {probs[config.NOARRIVE]:.1%}\n(결항·회항실패)",
                     (config.NOARRIVE, probs[config.NOARRIVE]),
                     textcoords="offset points", xytext=(-10, 26), ha="right",
                     fontsize=9, color="#c92a2a",
                     arrowprops={"arrowstyle": "->", "color": "#c92a2a",
                                 "lw": 0.8})

    # --- right: the same numbers, cumulated ----------------------------- #
    ax_cdf.plot(edges, cdf, "o-", color="#1c7ed6", lw=2)
    record = {"route": str(meta["route_key"]),
              "date": str(meta["FlightDate"])[:10],
              "crs_dep_hour": int(xgb_row["crs_dep_hour"]),
              "assumed_origin": assumed, "assumed_dest": assumed_dest}
    for level in LEVELS:
        minutes = float(common.guaranteed_minutes(probs[None, :], level)[0])
        record[f"guarantee_{int(level * 100)}"] = minutes
        colour = COLOURS[level]
        if np.isnan(minutes):
            continue
        ax_cdf.plot([0, minutes], [level, level], "--", color=colour, lw=1)
        ax_cdf.plot([minutes, minutes], [0, level], "--", color=colour, lw=1)
        ax_cdf.plot([minutes], [level], "o", color=colour, ms=7)
        ax_cdf.annotate(f"{int(level * 100)}% 보장 → 예정 +{minutes:.0f}분",
                        (minutes, level), textcoords="offset points",
                        xytext=(10, -14), fontsize=9.5, color=colour)
    record["reachable"] = float(cdf[-1])

    ax_cdf.axhline(1.0, color="#868e96", lw=0.8, ls=":")
    ax_cdf.annotate(f"곡선은 {cdf[-1]:.1%} 에서 멈춘다 — 나머지 "
                    f"{1 - cdf[-1]:.1%} 가 미도착 위험이고,\n"
                    "보장 시각이 아니라 경고로 따로 표시한다",
                    (0.97, 0.10), xycoords="axes fraction", ha="right",
                    fontsize=9, color="#495057")
    ax_cdf.set_xlabel("예정 도착 대비 지연(분)")
    ax_cdf.set_ylabel("그 시각까지 도착할 확률")
    ax_cdf.set_title("누적하면 — 손님이 보는 단 하나의 곡선", fontsize=11)
    ax_cdf.set_ylim(0, 1.04)
    ax_cdf.set_xlim(0, edges[-1] * 1.02)
    ax_cdf.grid(alpha=0.25)

    fig.suptitle("가정 하나로 답이 정해진다 — " + title
                 + chr(10) + "(test 2026, 보정 후)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "method_guarantee.png")
    plt.close(fig)

    table = pd.DataFrame([record])
    out = config.RESULTS_DIR / "figure_guarantee_example.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print("\n요구 확률을 올리면 보장 시각이 뒤로 밀린다. 그 교환이 이 서비스가 파는 것이다.")
    print("곡선이 1.0 에 닿지 않는 만큼이 결항·회항실패 위험이고, 화면에 따로 표시한다.")
    print(f"figure: {config.FIGURE_DIR / 'method_guarantee.png'}")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
