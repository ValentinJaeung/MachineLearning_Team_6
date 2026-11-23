"""Step 16b - one real flight from test 2026, shown under all five assumptions.

Reads  data/prep/ (step 9), data/model/ (steps 14, 12)
Writes results/step16b_case.csv          the chosen flight, five rows
       results/step16b_case_search.csv   how many flights passed each filter
       figures/step16b_case.png

Section 6.1.2 says the answer moves when the assumption moves. An average over
1.8 million flights proves that it moves on average; it does not let a reader
see it happen. This step picks ONE flight out of the test year and shows the
whole screen for it.

The example is chosen, not sampled, and the choice is made on conditions
written down here BEFORE the search runs:

  1. a route where the passenger has a real choice of flights to shift to
     (the 24 busiest routes of the test year, as step 16 uses for the demo)
  2. a 90% guarantee exists under ALL five assumptions, so the table has no
     holes and the five numbers are comparable
  3. the five guarantees are non-decreasing along the order the aggregate
     what-if produced (정상일 < 연결편붕괴일 < 공역혼잡일 ~ 항공사귀책일 <
     날씨일), so the picture does not contradict the population it came from
  4. 정상일 is a clean baseline: its 90% guarantee is at most 15 minutes
  5. among those, the widest 날씨일 - 정상일 gap

The widest gap of 246 flights is not what a reader should expect to see, so
the search file also records the median and quartiles of that gap among the
survivors. A showcase next to the distribution it was drawn from is a
showcase; a showcase on its own is a claim.

Rules 2-5 are a showcase, not a measurement, and rule 5 in particular picks
the most favourable flight there is. The report has to say so where the figure
appears, and this step prints the funnel - how many flights each rule removed
- so that "the most favourable of N" is a number the reader can see rather
than a claim they have to trust.

Nothing here feeds a metric. test 2026 was opened once, in step 15; this step
only displays rows of it that were already scored there.

Usage:  python src/step16b_case_study.py [--routes N] [--limit N]
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

LEVELS = [0.80, 0.90, 0.95]
BASELINE = 0            # 정상일
WORST = 4               # 날씨일
# the order the aggregate what-if produced, worst last (step14_whatif.csv)
ORDER = [0, 2, 3, 1, 4]
CLEAN_BASELINE_MAX = 15.0


def guarantees(probs: np.ndarray) -> dict[float, np.ndarray]:
    """90% and the other two promise levels, in minutes, for every row."""
    return {level: common.guaranteed_minutes(probs, level) for level in LEVELS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    import xgboost as xgb

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    test = common.load_split("test", args.limit, drop=drop, with_scenario=True)
    meta = test["meta"].reset_index(drop=True)

    route = meta["Origin"].astype(str) + "-" + meta["Dest"].astype(str)
    busiest = route.value_counts().head(args.routes).index.tolist()
    index = np.flatnonzero(route.isin(busiest).to_numpy())
    print(f"후보: {args.routes} 개 노선 · {len(index):,} 편")

    booster = xgb.Booster()
    booster.load_model(config.MODEL_DIR / "xgboost_scenario.json")
    calibrator = joblib.load(config.MODEL_DIR / "conditional_calibrator.joblib")

    frame = test["X_xgb"].iloc[index].reset_index(drop=True)
    levels = np.full((len(index), 5, len(LEVELS)), np.nan)
    noarrive = np.empty((len(index), 5))
    cdfs = np.empty((len(index), 5, len(config.EDGES)))
    for s in range(5):
        trial = frame.copy()
        for column in ("scen_origin", "scen_dest"):
            trial[column] = pd.Categorical(
                [str(s)] * len(trial), categories=[str(i) for i in range(5)])
        raw = booster.predict(xgb.DMatrix(trial, enable_categorical=True)
                              ).astype("float64")
        probs = calibrator.transform(raw)
        for j, level in enumerate(LEVELS):
            levels[:, s, j] = common.guaranteed_minutes(probs, level)
        noarrive[:, s] = probs[:, config.NOARRIVE]
        cdfs[:, s, :] = probs.cumsum(axis=1)[:, :len(config.EDGES)]
        print(f"  {config.SCENARIO_NAMES[s]} 계산 완료")

    # ---- the funnel, in the order the docstring states it ----------------- #
    g90 = levels[:, :, LEVELS.index(0.90)]
    steps = [("후보 편", np.ones(len(index), dtype=bool))]

    complete = ~np.isnan(g90).any(axis=1)
    steps.append(("다섯 가정 모두 90% 보장 가능", complete))

    ordered = g90[:, ORDER]
    monotone = complete & (np.diff(ordered, axis=1) >= 0).all(axis=1)
    steps.append(("집계와 같은 순서 (단조)", monotone))

    clean = monotone & (g90[:, BASELINE] <= CLEAN_BASELINE_MAX)
    steps.append((f"정상일 90% 보장 ≤ {CLEAN_BASELINE_MAX:.0f}분", clean))

    survivors = np.flatnonzero(clean)
    if not len(survivors):
        raise SystemExit("조건을 통과한 편이 없다 — 조건을 고치고 다시 적어야 한다")

    spread = g90[:, WORST] - g90[:, BASELINE]
    best = int(survivors[np.argmax(spread[survivors])])
    row = int(index[best])

    kept = spread[survivors]
    funnel = pd.DataFrame([{"단계": name, "남은 편": int(mask.sum()),
                            "남은 비율": round(float(mask.mean()), 4)}
                           for name, mask in steps]
                          + [{"단계": "통과한 편의 폭 — 중앙값", "남은 편": len(kept),
                              "남은 비율": round(float(np.median(kept)), 1)},
                             {"단계": "통과한 편의 폭 — 1사분위", "남은 편": len(kept),
                              "남은 비율": round(float(np.quantile(kept, .25)), 1)},
                             {"단계": "통과한 편의 폭 — 3사분위", "남은 편": len(kept),
                              "남은 비율": round(float(np.quantile(kept, .75)), 1)},
                             {"단계": "고른 편의 폭 (최댓값)", "남은 편": 1,
                              "남은 비율": round(float(kept.max()), 1)}])
    funnel.to_csv(config.RESULTS_DIR / "step16b_case_search.csv", index=False,
                  encoding="utf-8-sig")

    # ---- the chosen flight ------------------------------------------------ #
    flight = meta.iloc[row]
    # the scenario columns of the untouched frame are the day that happened
    true_origin = int(frame.iloc[best]["scen_origin"])
    true_dest = int(frame.iloc[best]["scen_dest"])
    actual = float(test["delay"][row])
    date = str(flight["FlightDate"])[:10]
    name = (f"{flight['Origin']}-{flight['Dest']} · {date} · "
            f"{flight['Reporting_Airline']}"
            f"{flight['Flight_Number_Reporting_Airline']}")

    table = pd.DataFrame([{
        "가정한 날": config.SCENARIO_NAMES[s],
        "80% 보장": round(float(levels[best, s, 0]), 1),
        "90% 보장": round(float(levels[best, s, 1]), 1),
        "95% 보장": ("보장 불가" if np.isnan(levels[best, s, 2])
                   else round(float(levels[best, s, 2]), 1)),
        "정상일 대비": round(float(g90[best, s] - g90[best, BASELINE]), 1),
        "미도착 확률": round(float(noarrive[best, s]), 4),
        "그날의 실제 유형인가": ("출발지·도착지" if s == true_origin == true_dest
                       else "출발지" if s == true_origin
                       else "도착지" if s == true_dest else ""),
    } for s in ORDER])
    table.insert(0, "편", name)
    table.to_csv(config.RESULTS_DIR / "step16b_case.csv", index=False,
                 encoding="utf-8-sig")

    # ---- figure: the screen itself, then the answer it produces -------- #
    edges = np.asarray(config.EDGES, dtype="float64")
    labels = [config.SCENARIO_NAMES[s] for s in ORDER]
    colours = ["#2b8a3e", "#f59f00", "#e8590c", "#c92a2a", "#862e9c"]

    fig, (ax_cdf, ax_bar) = plt.subplots(
        1, 2, figsize=(13.4, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    ax_cdf.axhline(0.90, color="#1864ab", linewidth=1.3, linestyle="--")
    ax_cdf.annotate("90% 선 — 이 선과 만나는 지점이 답이다", (edges[0], 0.90),
                    textcoords="offset points", xytext=(6, -14), ha="left",
                    fontsize=9, color="#1864ab")
    for k, sc in enumerate(ORDER):
        ax_cdf.plot(edges, cdfs[best, sc], "o-", color=colours[k],
                    markersize=4, linewidth=1.9, label=labels[k])
        minutes = levels[best, sc, LEVELS.index(0.90)]
        if not np.isnan(minutes):
            ax_cdf.plot([minutes, minutes], [0, 0.90], color=colours[k],
                        linewidth=1.0, alpha=0.55)
            ax_cdf.plot([minutes], [0.90], "v", color=colours[k], markersize=8)
    ax_cdf.set_xlabel("예정 도착 대비 지연 (분)")
    ax_cdf.set_ylabel("그 안에 도착할 확률")
    ax_cdf.set_ylim(0.3, 1.02)
    ax_cdf.set_title("가정이 곡선을 내리고, 90% 선과 만나는 지점이 밀린다",
                     fontsize=10)
    ax_cdf.legend(fontsize=9, loc="lower right")
    ax_cdf.grid(alpha=0.25)

    g = [levels[best, sc, LEVELS.index(0.90)] for sc in ORDER]
    positions = np.arange(len(ORDER))[::-1]
    ax_bar.barh(positions, g, color=colours, height=0.6)
    for k, value in enumerate(g):
        ax_bar.annotate(f"+{value:.0f}분", (value, positions[k]),
                        textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=10, fontweight="bold",
                        color=colours[k])
        if np.isnan(levels[best, ORDER[k], LEVELS.index(0.95)]):
            ax_bar.annotate("95% 보장 불가", (0, positions[k]),
                            textcoords="offset points", xytext=(8, 0),
                            va="center", fontsize=8.5, color="#ffffff")
    ax_bar.set_yticks(positions)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlim(0, max(g) * 1.28)
    ax_bar.set_xlabel("90% 보장 지연 (분)")
    ax_bar.set_title(f"답이 {max(g) - min(g):.0f}분 움직인다", fontsize=10)
    ax_bar.grid(alpha=0.25, axis="x")

    fig.suptitle(f"Step 16b — {name} · 예정 도착 "
                 f"{int(flight['crs_arr_min']) // 60:02d}:"
                 f"{int(flight['crs_arr_min']) % 60:02d} · "
                 f"실제 {actual:+.0f}분 (그날은 "
                 f"{config.SCENARIO_NAMES[true_origin]}이었다)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step16b_case.png")
    plt.close(fig)

    # ---- what it says ----------------------------------------------------- #
    print("\n" + "=" * 84)
    print(funnel.to_string(index=False))
    print(f"\n고른 편: {name}")
    print(f"  예정 출발 {int(flight['crs_dep_min']) // 60:02d}:"
          f"{int(flight['crs_dep_min']) % 60:02d} · "
          f"예정 도착 {int(flight['crs_arr_min']) // 60:02d}:"
          f"{int(flight['crs_arr_min']) % 60:02d}")
    print(f"  그날의 실제 유형: 출발지 "
          f"{config.SCENARIO_NAMES[true_origin]} · 도착지 "
          f"{config.SCENARIO_NAMES[true_dest]}")
    print(f"  실제 도착 지연: {actual:+.0f}분\n")
    print(table.drop(columns=["편"]).to_string(index=False))
    gap = float(g90[best, WORST] - g90[best, BASELINE])
    print(f"\n가정만 바꾸면 90% 보장 시각이 {gap:.0f}분 움직인다 "
          f"({g90[best, BASELINE]:.0f}분 → {g90[best, WORST]:.0f}분)")
    print(f"조건을 통과한 {len(survivors):,} 편 중 그 폭이 가장 큰 편을 골랐다 — "
          f"대표값이 아니라 가장 잘 보이는 예다.")
    print(f"  같은 조건을 통과한 편들의 폭: 중앙값 {np.median(kept):.0f}분 "
          f"(1사분위 {np.quantile(kept, .25):.0f}분 · "
          f"3사분위 {np.quantile(kept, .75):.0f}분)")
    print(f"  95% 는 비관적 가정에서 보장 불가다 — 숨기지 말고 그렇게 표시한다.")
    print(f"\nresults: {config.RESULTS_DIR / 'step16b_case.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
