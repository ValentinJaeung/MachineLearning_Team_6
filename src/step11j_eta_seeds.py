"""Step 11j - is eta=0.05 really better than the deployed eta=0.08, or is that
0.00038 a coin landing the same way twice?

Reads  data/prep/ (step 9)
       results/step11c_seed_check.csv   (the eta=0.08 arm, already measured)
Writes results/step11j_eta_seeds.csv
       results/step11j_verdict.csv
       figures/step11j_eta_seeds.png

Step 11g swept thirteen XGBoost settings on the full training set and the rule
in step 11i - pick the configuration by validation log loss - points at
eta=0.05 (1.40435) rather than the eta=0.08 that is actually deployed
(1.40473). The difference is 0.00038. The seed spread of this model's log loss
is sd 0.000152, so the threshold is 0.000304, and 0.00038 clears it by a
quarter. That is thin, and the six best settings are separated by 0.00004 to
0.00014 - all of them well inside the band. A ranking whose consecutive gaps
are noise is a ranking of noise.

Adopting eta=0.05 means refitting the conditional model, the calibrator and
the whole service evaluation, so the question is worth one measurement rather
than one argument. This is the same paired-seed design step 11c_seed_check
used for the polynomial expansion, where the apparent gain also failed to
survive three seeds.

Three seeds, same seeds as the existing eta=0.08 arm, so the comparison is
paired rather than two separate averages.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11j_eta_seeds.py
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

CHALLENGER = 0.05            # what step 11g's log loss pick wants
DEPLOYED = 0.08              # what is in data/model/xgboost.json today
SEEDS = [42, 1, 2]           # the same seeds step11c_seed_check used


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit_one(dtrain, dval, val, eta: float, seed: int, rounds: int,
            patience: int) -> dict:
    import xgboost as xgb

    history: dict = {}
    started = time.time()
    booster = xgb.train(step11.xgb_params(seed=seed, eta=eta), dtrain,
                        num_boost_round=rounds,
                        evals=[(dtrain, "train"), (dval, "val")],
                        evals_result=history, early_stopping_rounds=patience,
                        verbose_eval=False)
    seconds = time.time() - started
    best = int(booster.best_iteration)
    probs = booster.predict(dval, iteration_range=(0, best + 1))
    scores = common.evaluate(probs, val["cls"], val["delay"])
    return {"eta": eta, "seed": seed, "best_iteration": best,
            "train_mlogloss": round(float(history["train"]["mlogloss"][best]),
                                    5),
            **common.flat(scores), "fit_seconds": round(seconds, 1)}


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
    dtrain = xgb.DMatrix(train["X_xgb"], label=train["cls"],
                         enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"],
                       enable_categorical=True)

    # the deployed arm was already fitted under three seeds in step 11c
    old = pd.read_csv(config.RESULTS_DIR / "step11c_seed_check.csv")
    old = old[old["arm"] == "base"].copy()
    old["eta"] = DEPLOYED
    rows = [{"eta": DEPLOYED, "seed": int(r["seed"]),
             "best_iteration": int(r["best_iteration"]),
             "log_loss": r["log_loss"], "brier": r["brier"],
             "cdf_ece": r["cdf_ece"], "coverage_90": r["coverage_90"],
             "auc_late": r["auc_late"], "fit_seconds": r["fit_seconds"],
             "출처": "step11c_seed_check.csv"}
            for _, r in old.iterrows()]
    say(f"기존 eta={DEPLOYED} 팔: 시드 {sorted(old['seed'])} "
        f"(step11c_seed_check.csv 에서 가져옴)")

    path = config.RESULTS_DIR / "step11j_eta_seeds.csv"
    path.unlink(missing_ok=True)
    for seed in SEEDS:
        row = fit_one(dtrain, dval, val, CHALLENGER, seed, args.rounds,
                      args.patience)
        row["출처"] = "이 단계"
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        say(f"  eta={CHALLENGER} seed {seed}: log loss {row['log_loss']}  "
            f"CDF 오차 {row['cdf_ece']}  ({row['best_iteration']}회, "
            f"{row['fit_seconds']:.0f}초)")

    table = pd.DataFrame(rows)

    # ---- paired comparison ------------------------------------------------ #
    wide = table.pivot_table(index="seed", columns="eta",
                             values=["log_loss", "cdf_ece", "coverage_90"])
    diff = wide["log_loss"][CHALLENGER] - wide["log_loss"][DEPLOYED]
    sd_within = float(table[table["eta"] == DEPLOYED]["log_loss"].std(ddof=1))
    band = 2 * sd_within

    verdict = {
        "도전 설정": f"eta={CHALLENGER}",
        "현행 설정": f"eta={DEPLOYED}",
        "도전 log loss 평균": round(float(wide["log_loss"][CHALLENGER].mean()),
                                5),
        "현행 log loss 평균": round(float(wide["log_loss"][DEPLOYED].mean()), 5),
        "짝지은 차이 평균": round(float(diff.mean()), 5),
        "짝지은 차이 부호 일치": int((diff < 0).sum()),
        "시드 수": len(SEEDS),
        "임계값": round(band, 5),
        "도전 cdf_ece 평균": round(float(wide["cdf_ece"][CHALLENGER].mean()), 5),
        "현행 cdf_ece 평균": round(float(wide["cdf_ece"][DEPLOYED].mean()), 5),
    }
    wins_everywhere = int((diff < 0).sum()) == len(SEEDS)
    beats_band = abs(float(diff.mean())) > band
    verdict["판정"] = ("도전 설정 채택" if (wins_everywhere and beats_band)
                    else "구분 불가 — 현행 유지")
    verdict["채택 eta"] = (CHALLENGER if verdict["판정"] == "도전 설정 채택"
                        else DEPLOYED)
    pd.DataFrame([verdict]).to_csv(
        config.RESULTS_DIR / "step11j_verdict.csv", index=False,
        encoding="utf-8-sig")

    # ---- figure ----------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    for eta, colour in ((DEPLOYED, "#1864ab"), (CHALLENGER, "#e8590c")):
        part = table[table["eta"] == eta].sort_values("seed")
        ax1.plot(part["seed"].astype(str), part["log_loss"], "o-",
                 color=colour, label=f"eta={eta}", markersize=8)
    ax1.set_xlabel("시드")
    ax1.set_ylabel("val log loss")
    ax1.set_title("시드마다 같은 쪽이 이기는가", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.bar([str(s) for s in diff.index], diff.values,
            color=["#2f9e44" if v < 0 else "#c92a2a" for v in diff.values])
    ax2.axhline(0, color="#343a40", linewidth=1)
    ax2.axhspan(-band, band, color="#868e96", alpha=0.18)
    ax2.annotate(f"잡음 폭 ±{band:.5f}", (len(diff) - 0.5, band),
                 textcoords="offset points", xytext=(0, 5), ha="right",
                 fontsize=9)
    ax2.set_xlabel("시드")
    ax2.set_ylabel(f"eta={CHALLENGER} − eta={DEPLOYED}  (음수면 도전 설정 승)")
    ax2.set_title("짝지은 차이 — 회색 띠 안이면 구분 불가", fontsize=10)
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Step 11j — eta {CHALLENGER} vs {DEPLOYED}, 같은 시드로 "
                 f"짝지어 (val 2025)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11j_eta_seeds.png")
    plt.close(fig)

    # ---- what it says ----------------------------------------------------- #
    print("\n" + "=" * 96)
    print(table[["eta", "seed", "best_iteration", "log_loss", "cdf_ece",
                 "coverage_90", "auc_late", "fit_seconds", "출처"]]
          .to_string(index=False))
    print(f"\neta={CHALLENGER} 평균 {verdict['도전 log loss 평균']}  vs  "
          f"eta={DEPLOYED} 평균 {verdict['현행 log loss 평균']}")
    print("짝지은 차이 (같은 시드끼리):")
    for seed, value in diff.items():
        print(f"  seed {seed}: {value:+.5f}"
              f"{'  (도전 승)' if value < 0 else '  (현행 승)'}")
    print(f"\n평균 차이 {diff.mean():+.5f} / 임계값 {band:.5f} "
          f"(현행 설정의 시드 sd {sd_within:.6f} × 2)")
    print(f"세 시드 모두 같은 쪽이 이겼는가: "
          f"{'예' if wins_everywhere else '아니오'} "
          f"({verdict['짝지은 차이 부호 일치']}/{len(SEEDS)})")
    print(f"CDF 보정오차: {verdict['도전 cdf_ece 평균']} vs "
          f"{verdict['현행 cdf_ece 평균']} — 채택 지표에서는 "
          f"{'같다' if abs(verdict['도전 cdf_ece 평균'] - verdict['현행 cdf_ece 평균']) < 0.0001 else '다르다'}")
    print(f"\n판정: {verdict['판정']}  →  eta={verdict['채택 eta']}")
    if verdict["채택 eta"] == DEPLOYED:
        print("현행 설정을 유지한다. 11g 의 0.00038 은 시드를 바꾸면 "
              "남지 않는 차이였다 — 재학습하지 않는다.")
    else:
        print("조건부 모델·보정기·서비스 평가를 이 설정으로 다시 적합해야 한다.")
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
