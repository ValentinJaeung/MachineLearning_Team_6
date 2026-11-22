"""Step 11g - the XGBoost search, re-run under the conditions the comparison
actually uses.

Reads  data/prep/ (step 9)
Writes results/step11g_xgb_full.csv
       figures/step11g_xgb_full.png

Why this step exists.

The sweeps in step 11 were written to show the SHAPE of overfitting, so they
cut corners that are fine for a shape and not fine for a choice:

  * they fit on every third training row (SAMPLE = 3) at eta 0.15, while the
    model that actually ships is fitted on all of it at eta 0.08;
  * they record validation log loss only.

Meanwhile step 11d swept the network on the full training set and recorded the
whole metric set. So the two searches were not comparable, and - the part that
matters - no XGBoost configuration was ever scored on CDF calibration error,
which is the metric section 4.4 named as the selection metric before any
numbers were seen. The tree was being compared on that metric at exactly one
configuration while the network got to put forward the best of six.

This step closes that. The same twelve configurations, plus the production one,
fitted on the full training set with the production early-stopping rule, each
scored with the full metric set. After it, every configuration of every model
has been measured the same way, and the selection rule can be applied without
the answer depending on who got the bigger search.

The step 11 depth study stays as it is - it is a picture of the train/val gap,
it is honest about being a one-third sample, and re-running it at full size
would not change its shape.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11g_xgb_full.py [--rounds 400]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

import step11_models as step11

# the knob that sets capacity, then the small grid on top of the chosen depth -
# the same twelve the one-third sweep used, so the two can be laid side by side
DEPTHS = (3, 6, 8, 10, 12)
SEARCH_ETA = step11.SEARCH_ETA          # 0.15
GRID = ([{"eta": e, "min_child_weight": 50} for e in (0.05, 0.10, 0.20)]
        + [{"eta": SEARCH_ETA, "min_child_weight": m} for m in (10, 200)]
        + [{"eta": SEARCH_ETA, "min_child_weight": 50,
            "colsample_bytree": c} for c in (0.5, 1.0)])


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def label_of(setting: dict) -> str:
    base = {"max_depth": 6, "eta": SEARCH_ETA, "min_child_weight": 50,
            "subsample": 0.8, "colsample_bytree": 0.8}
    changed = {k: v for k, v in setting.items() if base.get(k) != v}
    if not changed:
        return "기준 d6 eta0.15"
    return " ".join(f"{k}={v}" for k, v in changed.items())


def fit_one(dtrain, dval, val, setting: dict, rounds: int,
            patience: int) -> dict:
    import xgboost as xgb

    history: dict = {}
    started = time.time()
    booster = xgb.train(step11.xgb_params(**setting), dtrain,
                        num_boost_round=rounds,
                        evals=[(dtrain, "train"), (dval, "val")],
                        evals_result=history, early_stopping_rounds=patience,
                        verbose_eval=False)
    seconds = time.time() - started
    best = int(booster.best_iteration)
    probs = booster.predict(dval, iteration_range=(0, best + 1))
    scores = common.evaluate(probs, val["cls"], val["delay"])
    train_loss = float(history["train"]["mlogloss"][best])
    return {"설정": label_of(setting),
            **{k: setting.get(k, step11.xgb_params()[k])
               for k in ("max_depth", "eta", "min_child_weight",
                         "colsample_bytree")},
            "best_iteration": best,
            "train_mlogloss": round(train_loss, 5),
            **common.flat(scores),
            "gap": round(scores["log_loss"] - train_loss, 5),
            "fit_seconds": round(seconds, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    import xgboost as xgb

    say("loading ...")
    train = common.load_split("train", args.limit)
    val = common.load_split("val", args.limit)
    say(f"  train {len(train['cls']):,}행 (전량)  val {len(val['cls']):,}행")

    dtrain = xgb.DMatrix(train["X_xgb"], label=train["cls"],
                         enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"],
                       enable_categorical=True)

    settings = ([{"max_depth": d, "eta": SEARCH_ETA} for d in DEPTHS]
                + GRID
                + [{"eta": 0.08}])          # the production configuration
    seen, unique = set(), []
    for setting in settings:
        key = tuple(sorted(step11.xgb_params(**setting).items()))
        if key not in seen:
            seen.add(key)
            unique.append(setting)

    path = config.RESULTS_DIR / "step11g_xgb_full.csv"
    path.unlink(missing_ok=True)
    rows = []
    for i, setting in enumerate(unique, 1):
        row = fit_one(dtrain, dval, val, setting, args.rounds, args.patience)
        rows.append(row)
        step11.append(path, row)          # written as they arrive
        say(f"  [{i}/{len(unique)}] {row['설정']}: "
            f"log loss {row['log_loss']}  CDF 오차 {row['cdf_ece']}  "
            f"({row['best_iteration']}회, {row['fit_seconds']:.0f}초)")

    table = pd.DataFrame(rows)

    # ---- figure ---------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    deep = table[table["설정"].str.startswith("max_depth")
                 | (table["설정"] == "기준 d6 eta0.15")]
    deep = deep.sort_values("max_depth")
    ax1.plot(deep["max_depth"], deep["train_mlogloss"], "o-", label="train",
             color="#4c6ef5")
    ax1.plot(deep["max_depth"], deep["log_loss"], "s-", label="val",
             color="#c92a2a")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("multiclass log loss")
    ax1.set_title("전량 학습에서의 깊이 — 격차가 벌어지는 모양", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.scatter(table["log_loss"], table["cdf_ece"], s=46, color="#1864ab",
                zorder=3)
    for _, row in table.iterrows():
        ax2.annotate(row["설정"], (row["log_loss"], row["cdf_ece"]),
                     textcoords="offset points", xytext=(0, 7), ha="center",
                     fontsize=7)
    pick = table.loc[table["log_loss"].idxmin()]
    ax2.scatter([pick["log_loss"]], [pick["cdf_ece"]], s=220, facecolors="none",
                edgecolors="#c92a2a", linewidths=2, zorder=4)
    ax2.set_xlabel("val log loss")
    ax2.set_ylabel("val CDF 보정오차")
    ax2.set_title("설정 13종 — 두 지표가 같은 방향이 아니다", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.suptitle("Step 11g — XGBoost 탐색을 최종 학습과 같은 조건에서 "
                 "(val 2025, 전량)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11g_xgb_full.png")
    plt.close(fig)

    # ---- what it says ---------------------------------------------------- #
    print("\n" + "=" * 100)
    print(table[["설정", "max_depth", "eta", "best_iteration", "train_mlogloss",
                 "log_loss", "cdf_ece", "coverage_90", "auc_late", "gap"]]
          .to_string(index=False))
    print(f"\nlog loss 로 고르면: {pick['설정']}  "
          f"log loss {pick['log_loss']}  CDF 오차 {pick['cdf_ece']}")
    by_ece = table.loc[table["cdf_ece"].idxmin()]
    print(f"CDF 오차로 고르면: {by_ece['설정']}  "
          f"log loss {by_ece['log_loss']}  CDF 오차 {by_ece['cdf_ece']}")
    print(f"\n탐색 폭 — log loss {table['log_loss'].max() - table['log_loss'].min():.5f}"
          f"  CDF 오차 {table['cdf_ece'].max() - table['cdf_ece'].min():.5f}")
    print("두 지표가 같은 설정을 고르지 않으면, 어느 지표로 고를지를 "
          "미리 정해 둔 것이 그 자리에서 값을 한다.")
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
