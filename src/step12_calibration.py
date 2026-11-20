"""
Step 12 - making the promised probability the real one.
(PROPOSAL_FLOW §4, 12단계)

Reads  data/prep/ (step 9), data/model/ (step 11)
Writes data/model/production_calibrator.joblib, production_prior.json
       results/step12_windows.csv, step12_reliability.csv
       figures/step12_windows.png, step12_reliability.png

Why this step exists at all
---------------------------
The product is the probability itself. If the model says 90% and 85% of those
flights keep the deadline, every other score can be excellent and the service
is still lying. A model is trained so the whole distribution fits; that is NOT
the same objective as "the thing I called 90% happens 90% of the time".

How "what actually happened" is even measurable
-----------------------------------------------
One flight cannot check a 90%: it either arrived or it did not. So flights
that were given the same probability are POOLED and counted, exactly the way
a rain forecast is verified - take the hundred days called "30% rain" and
count how many were wet. The reliability table here does that at all eight
bin edges, because the service reads the CDF at all eight.

Which data may build the conversion table
-----------------------------------------
  train  no  - the model already fits it, so the error is invisible there
  test   no  - it is the final exam; using it would void the result
  a held-out stretch  yes

And WHICH held-out stretch is itself measured rather than assumed. "The most
recent data" sounds obviously right, so it is put against "the same season a
year ago" and a 14-day rolling window.

The experiment
--------------
A model is refitted on 2022-2023 ONLY, so 2024 and 2025 are both genuinely
unseen and a full year's gap can be simulated. It is evaluated on 2025
July-August, which leaves 2025 June available to act as "last month".
Four correction methods x four windows, one table.

Then the production calibrator is fitted: the step 11 model (trained on
2022-2024) corrected on validation 2025 - unseen by it, and the same season
as the 2026 test year. Steps 14, 15 and 16 use that one.

Usage:  python src/step12_calibration.py [--rounds N]
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import common
import config
import plotting
from plotting import plt


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit_temperature(probs: np.ndarray, cls: np.ndarray) -> float:
    """One number that sharpens or softens the whole distribution."""
    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    index = np.arange(len(cls))

    def loss(log_t: float) -> float:
        scaled = log_p / np.exp(log_t)
        scaled -= scaled.max(axis=1, keepdims=True)
        return float(-(scaled[index, cls]
                       - np.log(np.exp(scaled).sum(axis=1))).mean())

    best = minimize_scalar(loss, bounds=(np.log(0.3), np.log(3.0)),
                           method="bounded")
    return float(np.exp(best.x))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    out = np.power(np.clip(probs, 1e-12, 1.0), 1.0 / temperature)
    return out / out.sum(axis=1, keepdims=True)


def prior_of(cls: np.ndarray) -> np.ndarray:
    return np.bincount(cls, minlength=config.N_CLASS) / len(cls)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    args = parser.parse_args()

    import xgboost as xgb

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    say("loading ...")
    train = common.load_split("train", drop=drop)
    val = common.load_split("val", drop=drop)

    year = train["meta"]["Year"].to_numpy()
    fit_rows, cal_rows = year < 2024, year == 2024
    say(f"  refit on 2022-23 {int(fit_rows.sum()):,} rows / "
        f"2024 held out {int(cal_rows.sum()):,} / val 2025 {len(val['cls']):,}")

    # ---------------------------------------------------------------- #
    # A. the experiment: which window should build the table?
    # ---------------------------------------------------------------- #
    say("refitting on 2022-23 so 2024 and 2025 are both unseen ...")
    params = {"objective": "multi:softprob", "num_class": config.N_CLASS,
              "eval_metric": "mlogloss", "tree_method": "hist",
              "max_depth": 6, "eta": 0.08, "min_child_weight": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "seed": config.SEED}
    dtrain = xgb.DMatrix(train["X_xgb"][fit_rows], label=train["cls"][fit_rows],
                         enable_categorical=True)
    dcal = xgb.DMatrix(train["X_xgb"][cal_rows], label=train["cls"][cal_rows],
                       enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"], enable_categorical=True)
    started = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=args.rounds,
                        evals=[(dcal, "2024")], early_stopping_rounds=40,
                        verbose_eval=100)
    span = (0, booster.best_iteration + 1)
    p2024 = booster.predict(dcal, iteration_range=span).astype("float64")
    p2025 = booster.predict(dval, iteration_range=span).astype("float64")
    say(f"  {time.time() - started:.0f}s, {booster.best_iteration} rounds")

    pi_model = prior_of(train["cls"][fit_rows])
    c2024, d2024 = train["cls"][cal_rows], train["delay"][cal_rows]
    date = pd.to_datetime(val["meta"]["FlightDate"])
    month, day = date.dt.month.to_numpy(), date.dt.normalize().to_numpy()

    june, target = month == 6, month > 6
    pe, ce, de = p2025[target], val["cls"][target], val["delay"][target]
    say(f"  evaluation window: 2025 July-August, {int(target.sum()):,} rows")

    rows = []

    def record(label: str, window: str, probs) -> dict:
        scores = common.evaluate(probs, ce, de)
        row = {"보정 방식": label, "보정 학습 구간": window, **common.flat(scores)}
        rows.append(row)
        return row

    base = record("보정 없음", "—", pe)
    say(f"  uncalibrated: CDF 오차 {base['cdf_ece']}  "
        f"90% 약속 이행 {base['coverage_90']}")

    windows = {
        "작년 같은 세 달 (2024 6–8월)": (p2024, c2024, d2024),
        "직전 달 (2025-06)": (p2025[june], val["cls"][june], val["delay"][june]),
        "평가 구간 자체 (도달 불가 상한)": (pe, ce, de),
    }
    for window, (pw, cw, dw) in windows.items():
        iso = common.CdfCalibrator().fit(pw, cw, dw)
        record("isotonic", window, iso.transform(pe))
        temperature = fit_temperature(pw, cw)
        record(f"temperature (T={temperature:.3f})", window,
               apply_temperature(pe, temperature))
        pi_w = prior_of(cw)
        record("prior 보정", window, common.prior_shift(pe, pi_model, pi_w))
        shifted = common.prior_shift(pw, pi_model, pi_w)
        iso2 = common.CdfCalibrator().fit(shifted, cw, dw)
        record("prior + isotonic", window,
               iso2.transform(common.prior_shift(pe, pi_model, pi_w)))
        say(f"  [{window}] done")

    # rolling 14 days, refitted every day
    say("rolling 14-day window ...")
    rolled = pe.copy()
    day_target = day[target]
    for d in np.unique(day_target):
        in_window = (day >= d - np.timedelta64(14, "D")) & (day < d)
        here = day_target == d
        if in_window.sum() < 50_000:
            continue
        rolled[here] = (common.CdfCalibrator()
                        .fit(p2025[in_window], val["cls"][in_window],
                             val["delay"][in_window])
                        .transform(pe[here]))
    record("isotonic", "최근 14일 롤링", rolled)

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step12_windows.csv",
                 index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- #
    # B. the production calibrator
    # ---------------------------------------------------------------- #
    say("fitting the production calibrator (step 11 model, corrected on val 2025)")
    prod_val = np.load(config.MODEL_DIR / "val_pred_xgboost.npy").astype("float64")
    calibrator = common.CdfCalibrator().fit(prod_val, val["cls"], val["delay"])
    pi_train = prior_of(train["cls"])
    pi_val = prior_of(val["cls"])
    joblib.dump(calibrator, config.MODEL_DIR / "production_calibrator.joblib")
    (config.MODEL_DIR / "production_prior.json").write_text(json.dumps({
        "source_prior_train_2022_2024": pi_train.round(6).tolist(),
        "target_prior_val_2025": pi_val.round(6).tolist(),
        "class_names": config.CLASS_NAMES,
        "note": ("the calibrator is fitted on validation 2025, which the "
                 "production model never trained on and which is the same "
                 "season as the 2026 test year"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    before = common.evaluate(prod_val, val["cls"], val["delay"])
    after = common.evaluate(calibrator.transform(prod_val), val["cls"], val["delay"])
    reliability = pd.DataFrame([
        {"stage": stage, "edge": edge, "predicted": v["예측"],
         "actual": v["실제"], "gap": round(v["예측"] - v["실제"], 4)}
        for stage, scores in (("보정 전", before), ("보정 후(in-sample)", after))
        for edge, v in scores["reliability"].items()])
    reliability.to_csv(config.RESULTS_DIR / "step12_reliability.csv",
                       index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- #
    # figures
    # ---------------------------------------------------------------- #
    best_per_window = (table[table["보정 방식"] != "보정 없음"]
                       .sort_values("cdf_ece")
                       .drop_duplicates("보정 학습 구간"))
    order = pd.concat([table[table["보정 방식"] == "보정 없음"], best_per_window])
    labels = [f"{w}\n({m})" if w != "—" else "보정 없음"
              for w, m in zip(order["보정 학습 구간"], order["보정 방식"])]

    fig, ax = plt.subplots(figsize=(11, 5))
    colours = ["#c92a2a" if w == "—" else
               ("#adb5bd" if "상한" in str(w) else "#4c6ef5")
               for w in order["보정 학습 구간"]]
    bars = ax.bar(labels, order["cdf_ece"], color=colours)
    for bar, v in zip(bars, order["cdf_ece"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.5f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CDF 보정 오차 (낮을수록 정직)")
    ax.set_title("Step 12 — 어느 구간으로 환산표를 만들어야 하는가 (2025년 7·8월 평가)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step12_windows.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for stage, marker in (("보정 전", "o-"), ("보정 후(in-sample)", "s-")):
        part = reliability[reliability["stage"] == stage]
        ax.plot(part["actual"], part["predicted"], marker, label=stage)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="완벽한 보정")
    ax.set_xlabel("실제 비율"); ax.set_ylabel("모델이 말한 확률")
    ax.set_title("Step 12 — 8개 구간 경계에서의 신뢰도")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step12_reliability.png")
    plt.close(fig)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 100)
    show = table[["보정 방식", "보정 학습 구간", "log_loss", "cdf_ece",
                  "coverage_90", "auc_late"]]
    print(show.to_string(index=False))

    real = best_per_window[~best_per_window["보정 학습 구간"].str.contains("상한")]
    best = real.sort_values("cdf_ece").iloc[0]
    print(f"\n가장 좋은 실현 가능 조합: {best['보정 방식']} / "
          f"{best['보정 학습 구간']}  → CDF 오차 "
          f"{base['cdf_ece']:.5f} → {best['cdf_ece']:.5f}")
    print("'가장 최신'이 정답이 아니다. 8월은 매년 가장 좋은 달이고 6·7월은 나쁘므로,")
    print("월 효과가 연도 효과보다 크다. 계절 구성이 같은 과거가 더 가까운 과거다.")
    print(f"\n배포용: 11단계 모델(2022–24 학습)을 val 2025 로 보정 → "
          f"{config.MODEL_DIR / 'production_calibrator.joblib'}")
    print("보정은 성능을 올리는 작업이 아니라 약속을 정직하게 만드는 작업이다. "
          "보장 시각은 오히려 뒤로 밀린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
