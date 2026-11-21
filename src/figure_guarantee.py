"""The method in one picture: one flight's distribution becomes a promise.

Reads  data/prep/ (step 9), data/model/xgboost.json (step 11),
       data/model/production_calibrator.joblib (step 12)
Writes figures/method_guarantee.png
       results/figure_guarantee_example.csv

Every number the service shows comes out of ONE output - the ten-bin
distribution for a single flight. This figure draws that output twice for the
same flight:

  left   the ten bins themselves, the model's answer
  right  the same thing summed up, which is the only curve a passenger needs:
         "by how many minutes am I there with probability p?"

The three guarantee times (80 / 90 / 95%) are read off the right panel, and
the tenth bin - never arrived - is why the curve stops short of 1.0. A flight
whose curve never reaches the required level gets "보장 불가" instead of an
invented time.

Two flights are drawn side by side, an early one and a late one, because the
whole product is the gap between them.

Usage:  python src/figure_guarantee.py
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

LEVELS = [0.80, 0.90, 0.95]
EARLY_HOUR = 6
LATE_HOUR = 19
MIN_ROWS = 200


def pick_example(meta: pd.DataFrame, xgb: pd.DataFrame, hour: int) -> int:
    """One real test flight: a busy route at the given scheduled hour."""
    busiest = meta["route_key"].value_counts()
    at_hour = xgb["crs_dep_hour"].to_numpy() == hour
    for route in busiest.index:
        rows = np.flatnonzero((meta["route_key"].to_numpy() == route) & at_hour)
        if len(rows) >= MIN_ROWS:
            return int(rows[len(rows) // 2])
    raise SystemExit(f"no route with {MIN_ROWS}+ flights at {hour}시")


def panel(ax_bars, ax_cdf, probs: np.ndarray, label: str, colour: str) -> dict:
    edges = np.asarray(config.EDGES, dtype="float64")
    cdf = probs.cumsum()[:len(config.EDGES)]

    ax_bars.bar(range(config.N_CLASS), probs, color=colour, alpha=0.85)
    ax_bars.set_xticks(range(config.N_CLASS))
    ax_bars.set_xticklabels(config.CLASS_NAMES, rotation=55, ha="right",
                            fontsize=7)
    ax_bars.set_ylabel("확률")
    ax_bars.set_title(f"{label} — 모델의 출력 10칸", fontsize=10)
    ax_bars.annotate(f"미도착 {probs[config.NOARRIVE]:.1%}",
                     (config.NOARRIVE, probs[config.NOARRIVE]),
                     textcoords="offset points", xytext=(-6, 10),
                     ha="right", fontsize=8, color="#c92a2a")

    ax_cdf.plot(edges, cdf, "o-", color=colour, label=label)
    out = {"label": label}
    for level in LEVELS:
        minutes = float(common.guaranteed_minutes(probs[None, :], level)[0])
        out[f"guarantee_{int(level * 100)}"] = minutes
        if np.isnan(minutes):
            continue
        ax_cdf.plot([minutes, minutes], [0, level], ":", color=colour, lw=1)
        ax_cdf.annotate(f"{int(level * 100)}% → +{minutes:.0f}분",
                        (minutes, level), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color=colour)
    out["reachable"] = float(cdf[-1])
    return out


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    import xgboost as xgb_lib

    test = common.load_split("test")
    booster = xgb_lib.Booster()
    booster.load_model(config.MODEL_DIR / "xgboost.json")
    calibrator = joblib.load(config.MODEL_DIR / "production_calibrator.joblib")

    rows = {f"{EARLY_HOUR:02d}시 출발": pick_example(test["meta"], test["X_xgb"],
                                                  EARLY_HOUR),
            f"{LATE_HOUR:02d}시 출발": pick_example(test["meta"], test["X_xgb"],
                                                 LATE_HOUR)}
    index = list(rows.values())
    raw = booster.predict(
        xgb_lib.DMatrix(test["X_xgb"].iloc[index], enable_categorical=True)
    ).astype("float64")
    probs = calibrator.transform(raw)

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax_cdf = fig.add_subplot(gs[:, 1])
    records = []
    for i, ((label, row), colour) in enumerate(zip(rows.items(),
                                                   ("#2b8a3e", "#c92a2a"))):
        ax_bars = fig.add_subplot(gs[i, 0])
        record = panel(ax_bars, ax_cdf, probs[i], label, colour)
        meta = test["meta"].iloc[row]
        record["route"] = str(meta["route_key"])
        record["date"] = str(meta["FlightDate"])[:10]
        records.append(record)

    ax_cdf.axhline(1.0, color="#868e96", lw=0.8, ls="--")
    ax_cdf.set_xlabel("예정 도착 대비 지연(분)")
    ax_cdf.set_ylabel("그 시각까지 도착할 확률")
    ax_cdf.set_title("같은 출력을 누적하면 — 손님이 보는 단 하나의 곡선", fontsize=10)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.legend(fontsize=9, loc="lower right")
    ax_cdf.grid(alpha=0.25)

    fig.suptitle("출력 하나에서 서비스 화면 전부가 나온다 (test 2026, 보정 후)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "method_guarantee.png")
    plt.close(fig)

    table = pd.DataFrame(records)
    out = config.RESULTS_DIR / "figure_guarantee_example.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print("\n같은 노선·같은 분포 형식인데 보장 시각이 다르다 — 차이를 만드는 것은 출발 시각이다.")
    print(f"figure: {config.FIGURE_DIR / 'method_guarantee.png'}")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
