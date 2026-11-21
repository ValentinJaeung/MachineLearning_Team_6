"""
Step 10 - feature selection by contribution, not by correlation.
(PROPOSAL_FLOW §4, 10단계)

Reads  data/prep/ (step 9)
Writes results/step10_gate.csv, results/step10_noise.csv
       figures/step10_gate.png

Correlation is not contribution
-------------------------------
Step 8 showed that every report-card feature moves with the target on its own
(C_bad_day_rel_fail spreads the on-time rate by 15 points across its deciles).
That is not a reason to keep it. If the airline and airport identifiers
already carry the same information, adding it changes nothing - and it can
make things worse by adding noise.

    a sixth tall player does not help a team that already has five.

So each block is judged by putting it IN and measuring whether the validation
distribution gets better.

The decision rule, fixed before the numbers were seen
-----------------------------------------------------
A block is adopted only if it improves validation log loss by more than twice
the run-to-run noise. The noise is not guessed: the baseline is refitted with
three seeds and the spread of those fits is the yardstick. Without that band,
a 0.0003 "improvement" would look like evidence.

Screening budget
----------------
The gate fits one model per block, so it runs on a fixed one-third sample of
train (identical rows for every block) with a faster learning rate. It is a
RELATIVE comparison under identical conditions; step 11 then fits the chosen
set on all 5.4M rows. Any block whose margin lands inside the noise band is
refitted on the full training data before the decision is recorded.

The scenario block is measured but cannot be deployed
-----------------------------------------------------
"What kind of day was it" is not knowable at booking time. It is put through
the same gate anyway, because the size of its gain is the price tag of
real-time information - the gap step 14 studies properly.

Usage:  python src/step10_feature_selection.py [--limit N]
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import common
import config
import plotting
from plotting import plt

A = ["A_absorb_rate"]
B = ["B1_turnaround_mean", "B2_day_seq_mean", "B3_first_flight_rate"]
C = ["C_bad_day_rel_fail"]
ALL_HISTORY = A + B + C

# name -> (features dropped from the 14, use the scenario columns?)
BLOCKS = {
    "기본 9 (예약 시점 시간표만)": (ALL_HISTORY, False),
    "기본 + A 흡수율": (B + C, False),
    "기본 + B 연결편 3종": (A + C, False),
    "기본 + C 나쁜 날 상대실패": (A + B, False),
    "기본 + A·B·C 전부": ([], False),
    "기본 + 그날의 유형 (배포 불가·천장 측정)": (ALL_HISTORY, True),
}
BASELINE = "기본 9 (예약 시점 시간표만)"

GATE_SAMPLE = 3          # every 3rd training row, the same rows for every block
GATE_ROUNDS = 220
GATE_ETA = 0.15
NOISE_SEEDS = [42, 1, 2]


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit_once(train: dict, val: dict, seed: int, rounds: int, eta: float,
             sample: int) -> tuple[float, dict]:
    """One XGBoost fit; returns validation log loss and run information."""
    index = np.arange(0, len(train["cls"]), sample)
    frame = train["X_xgb"].iloc[index]
    labels = train["cls"][index]

    dtrain = xgb.DMatrix(frame, label=labels, enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"], enable_categorical=True)
    params = {
        "objective": "multi:softprob", "num_class": config.N_CLASS,
        "eval_metric": "mlogloss", "tree_method": "hist",
        "max_depth": 6, "eta": eta, "min_child_weight": 50,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
    }
    history: dict = {}
    started = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=rounds,
                        evals=[(dval, "val")], evals_result=history,
                        early_stopping_rounds=20, verbose_eval=False)
    best = int(booster.best_iteration)
    probs = booster.predict(dval, iteration_range=(0, best + 1))
    scores = common.evaluate(probs, val["cls"], val["delay"])
    return float(history["val"]["mlogloss"][best]), {
        "best_iteration": best,
        "fit_seconds": round(time.time() - started, 1),
        "n_train_rows": len(labels),
        **common.flat(scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    # Deliberately NOT cached. One loaded split pair is about 4 GB, and holding
    # six of them at once would exhaust memory; reloading costs a few seconds.
    def data(scenario: bool, drop: list[str]) -> tuple[dict, dict]:
        return (common.load_split("train", args.limit, drop=drop,
                                  with_scenario=scenario),
                common.load_split("val", args.limit, drop=drop,
                                  with_scenario=scenario))

    # ------------------------------------------------------------------ #
    # 1. how much does the same configuration move between seeds?
    # ------------------------------------------------------------------ #
    noise_path = config.RESULTS_DIR / "step10_noise.csv"
    drop, scenario = BLOCKS[BASELINE]
    if noise_path.exists():
        noise = pd.read_csv(noise_path)
        noise_rows = noise.to_dict("records")
        say(f"baseline noise: reusing {noise_path.name} "
            f"({len(noise)} seeds already measured)")
    else:
        say("baseline noise: refitting the same configuration with three seeds")
        train, val = data(scenario, drop)
        noise_rows = []
        for seed in NOISE_SEEDS:
            loss, info = fit_once(train, val, seed, GATE_ROUNDS, GATE_ETA,
                                  GATE_SAMPLE)
            noise_rows.append({"seed": seed, "val_log_loss": round(loss, 6),
                               **info})
            say(f"  seed {seed}: val log loss {loss:.6f}  ({info['fit_seconds']}s)")
        del train, val
        noise = pd.DataFrame(noise_rows)
    noise.to_csv(noise_path, index=False, encoding="utf-8-sig")
    sigma = float(noise["val_log_loss"].std(ddof=1))
    threshold = 2 * sigma
    baseline_loss = float(noise["val_log_loss"].mean())
    say(f"  seed sd = {sigma:.6f}  ->  a block must gain more than "
        f"{threshold:.6f} to be adopted")

    # ------------------------------------------------------------------ #
    # 2. the gate
    # ------------------------------------------------------------------ #
    rows = []
    for name, (drop, scenario) in BLOCKS.items():
        if name == BASELINE:
            loss, info = baseline_loss, dict(noise_rows[0])
            info.pop("seed", None)
        else:
            say(f"gate: {name}")
            train, val = data(scenario, drop)
            loss, info = fit_once(train, val, config.SEED, GATE_ROUNDS,
                                  GATE_ETA, GATE_SAMPLE)
            del train, val
            gc.collect()
        gain = baseline_loss - loss
        rows.append({
            "block": name, "val_log_loss": round(loss, 6),
            "gain_vs_baseline": round(gain, 6),
            "noise_threshold": round(threshold, 6),
            "verdict": "통과" if gain > threshold else "미달",
            "deployable": "아니오 (예약 시점 미지)" if scenario else "예",
            **info,
        })
        say(f"  {name}: {loss:.6f}  (gain {gain:+.6f})  "
            f"{'통과' if gain > threshold else '미달'}")

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step10_gate.csv",
                 index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ #
    # 3. figure
    # ------------------------------------------------------------------ #
    plot = table[table["block"] != BASELINE]
    colours = ["#2b8a3e" if v == "통과" else "#adb5bd" for v in plot["verdict"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(plot["block"], plot["gain_vs_baseline"], color=colours)
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(-threshold, threshold, color="#ffd8a8", alpha=0.55,
               label=f"시드 잡음 ±2σ = ±{threshold:.5f}")
    for bar, value in zip(bars, plot["gain_vs_baseline"]):
        ax.text(value + (0.0012 if value >= 0 else -0.0012), bar.get_y()
                + bar.get_height() / 2, f"{value:+.5f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.set_xlabel("val log loss 개선폭 (클수록 좋음)")
    ax.set_title("Step 10 — 변수 블록 게이트: 상관이 아니라 기여로 판정")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step10_gate.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 92)
    print(table[["block", "val_log_loss", "gain_vs_baseline", "verdict",
                 "deployable", "cdf_ece", "auc_late", "fit_seconds"]]
          .to_string(index=False))

    passed = table[(table["verdict"] == "통과") & (table["deployable"] == "예")]
    print(f"\n시드 잡음 ±2σ = ±{threshold:.6f} (기준은 숫자를 보기 전에 정했다)")
    if passed.empty:
        print("배포 가능한 블록 중 통과한 것이 없다.")
        print("→ 최종 입력은 '기본 9'. 예약 시점 시간표가 가진 정보가 천장이고,")
        print("  과거 이력은 항공사·공항 ID가 이미 담고 있던 정보였다.")
    else:
        print("통과한 블록: " + ", ".join(passed["block"].tolist()))
    oracle = table[table["deployable"] != "예"]
    if not oracle.empty:
        print(f"\n참고 — 그날의 유형(배포 불가)은 "
              f"{float(oracle['gain_vs_baseline'].iloc[0]):+.5f}. "
              "이 크기가 '실시간 정보의 가격표'이며 14단계에서 제대로 다룬다.")
    print(f"\nresults: {config.RESULTS_DIR / 'step10_gate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
