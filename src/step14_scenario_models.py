"""
Step 14 - conditioning on the kind of day: the what-if panel and the ceiling.

Reads  data/prep/ (step 9), data/model/ (step 11)
Writes data/model/xgboost_scenario.json, scenario_type_{origin,dest}.json
       results/step14_models.csv, step14_type_prediction.csv, step14_whatif.csv
       figures/step14_models.png, figures/step14_whatif.png

Three models and one composite
------------------------------
  A  P(y | x)            unconditional - what the service actually ships
  B  P(y | x, s)         conditional on the day type
  C  P(s | x)            can the day type be predicted at booking time?

  composite(y | x) = Σ_s C(s | x) · B(y | x, s)

ONE model, not five
-------------------
The day type is an extra INPUT COLUMN, not a reason to train five models.
Asking the same doctor "and if I stopped smoking?" is not the same as raising
a second doctor. Five models would split the data (most days are ordinary),
relearn the same shared rules five times, and make any difference between
types indistinguishable from the difference between five fits.

B is deliberately unusable
--------------------------
B is handed the TRUE day type, which nobody knows when booking. It cannot be
deployed. Its number is the ceiling: what perfect knowledge of the day would
be worth. The gap between A and B is the price tag of real-time information,
and it is always reported as a pair - never as "our performance".

What the composite settles
--------------------------
By the law of total probability the composite should land on A. If it does,
that is a demonstration, not an assertion, that DECOMPOSING A PREDICTION BY
CAUSE AND RECOMBINING IT CREATES NO INFORMATION. "Split it by cause and it
will get more accurate" is an intuition many people hold, so it is worth
spending a measurement on.

C is also the justification of the whole design: if the day type cannot be
predicted from booking-time information, then leaving it as an ASSUMPTION the
passenger supplies is the only honest option.

Scored on validation 2025. Test 2026 is not opened here.

Usage:  python src/step14_scenario_models.py [--rounds N] [--composite-stride N]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import common
import config
import plotting
from plotting import plt

TYPE_NAMES = config.SCENARIO_NAMES
SCEN = ["scen_origin", "scen_dest"]

# How far above majority-class guessing counts as "can be predicted". Both
# sides land within 0.001 of the majority rate, which on 1.8M rows is a
# difference of a few hundred flights either way - not a capability.
MEANINGFUL_MARGIN = 0.005


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def params(num_class: int, seed: int = config.SEED) -> dict:
    return {"objective": "multi:softprob", "num_class": num_class,
            "eval_metric": "mlogloss", "tree_method": "hist",
            "max_depth": 6, "eta": 0.08, "min_child_weight": 50,
            "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--composite-stride", type=int, default=4)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    say("loading ...")
    train = common.load_split("train", drop=drop, with_scenario=True)
    val = common.load_split("val", drop=drop, with_scenario=True)
    base_cols = [c for c in train["X_xgb"].columns if c not in SCEN]

    rows = []

    # ---------------------------------------------------------------- #
    # A - already fitted in step 11
    # ---------------------------------------------------------------- #
    probs_a = np.load(config.MODEL_DIR / "val_pred_xgboost.npy").astype("float64")
    scores = common.evaluate(probs_a, val["cls"], val["delay"])
    rows.append({"model": "A 무조건 P(y|x)", "배포 가능": "예", **common.flat(scores)})
    say(f"A: log loss {scores['log_loss']}  AUC {scores['auc_late']}")

    # ---------------------------------------------------------------- #
    # B - the oracle
    # ---------------------------------------------------------------- #
    say("B: fitting with the TRUE day type as an input (oracle) ...")
    dtrain = xgb.DMatrix(train["X_xgb"], label=train["cls"], enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"], enable_categorical=True)
    started = time.time()
    model_b = xgb.train(params(config.N_CLASS), dtrain, num_boost_round=args.rounds,
                        evals=[(dval, "val")], early_stopping_rounds=40,
                        verbose_eval=100)
    span_b = (0, model_b.best_iteration + 1)
    model_b.save_model(config.MODEL_DIR / "xgboost_scenario.json")
    probs_b = model_b.predict(dval, iteration_range=span_b).astype("float64")
    scores = common.evaluate(probs_b, val["cls"], val["delay"])
    rows.append({"model": "B 참 유형 주입 (오라클)", "배포 가능": "아니오",
                 **common.flat(scores),
                 "fit_seconds": round(time.time() - started, 1)})
    say(f"B: log loss {scores['log_loss']}  AUC {scores['auc_late']}")

    # ---------------------------------------------------------------- #
    # C - can the day type be predicted at booking time?
    # ---------------------------------------------------------------- #
    type_rows, type_models, type_probs = [], {}, {}
    for side in SCEN:
        say(f"C: predicting {side} from booking-time features ...")
        y_tr = train["X_xgb"][side].cat.codes.to_numpy().astype("int64")
        y_va = val["X_xgb"][side].cat.codes.to_numpy().astype("int64")
        dtr = xgb.DMatrix(train["X_xgb"][base_cols], label=y_tr,
                          enable_categorical=True)
        dva = xgb.DMatrix(val["X_xgb"][base_cols], label=y_va,
                          enable_categorical=True)
        model_c = xgb.train(params(5), dtr, num_boost_round=args.rounds,
                            evals=[(dva, "val")], early_stopping_rounds=30,
                            verbose_eval=False)
        model_c.save_model(config.MODEL_DIR / f"scenario_type_{side}.json")
        p = model_c.predict(dva, iteration_range=(0, model_c.best_iteration + 1))
        type_models[side], type_probs[side] = model_c, p.astype("float64")

        predicted = p.argmax(axis=1)
        majority = int(np.bincount(y_tr, minlength=5).argmax())
        accuracy = float((predicted == y_va).mean())
        baseline = float((y_va == majority).mean())
        type_rows.append({
            "side": side, "accuracy": round(accuracy, 4),
            "majority_baseline": round(baseline, 4),
            "majority_class": TYPE_NAMES[majority],
            "margin": round(accuracy - baseline, 4),
            # A bare `accuracy > baseline` would call +0.0009 a win. The margin
            # has to clear something, so it has to clear MEANINGFUL_MARGIN -
            # fixed here rather than read off the result.
            "beats_majority": bool(accuracy - baseline > MEANINGFUL_MARGIN),
            "log_loss": round(float(-np.log(np.clip(
                p[np.arange(len(y_va)), y_va], 1e-9, 1)).mean()), 5),
        })
        say(f"  {side}: accuracy {accuracy:.4f} vs majority {baseline:.4f}")

    types = pd.DataFrame(type_rows)
    types.to_csv(config.RESULTS_DIR / "step14_type_prediction.csv",
                 index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- #
    # composite - does decomposing create information?
    # ---------------------------------------------------------------- #
    stride = args.composite_stride
    pick = np.arange(0, len(val["cls"]), stride)
    say(f"composite: 25 (origin, dest) combinations over {len(pick):,} rows "
        f"(every {stride}th)")
    sub = val["X_xgb"].iloc[pick].reset_index(drop=True)
    weights_o = type_probs["scen_origin"][pick]
    weights_d = type_probs["scen_dest"][pick]

    composite = np.zeros((len(pick), config.N_CLASS))
    for s_o in range(5):
        for s_d in range(5):
            trial = sub.copy()
            trial["scen_origin"] = pd.Categorical(
                [str(s_o)] * len(trial), categories=[str(i) for i in range(5)])
            trial["scen_dest"] = pd.Categorical(
                [str(s_d)] * len(trial), categories=[str(i) for i in range(5)])
            p = model_b.predict(xgb.DMatrix(trial, enable_categorical=True),
                                iteration_range=span_b).astype("float64")
            composite += (weights_o[:, s_o] * weights_d[:, s_d])[:, None] * p
    composite /= composite.sum(axis=1, keepdims=True)

    cls_sub, delay_sub = val["cls"][pick], val["delay"][pick]
    for label, probs in (("A 무조건 (같은 부분집합)", probs_a[pick]),
                         ("B 오라클 (같은 부분집합)", probs_b[pick]),
                         ("C·B 합성 Σ C(s|x)·B(y|x,s)", composite)):
        scores = common.evaluate(probs, cls_sub, delay_sub)
        rows.append({"model": label,
                     "배포 가능": "아니오" if "오라클" in label else "예",
                     **common.flat(scores)})

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step14_models.csv",
                 index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- #
    # the product feature: the same flights under each assumption
    # ---------------------------------------------------------------- #
    say("what-if panel: the same flights under each assumed day type")
    whatif_rows = []
    for s in range(5):
        trial = sub.copy()
        for side in SCEN:          # the panel assumes the same kind of day at both ends
            trial[side] = pd.Categorical([str(s)] * len(trial),
                                         categories=[str(i) for i in range(5)])
        p = model_b.predict(xgb.DMatrix(trial, enable_categorical=True),
                            iteration_range=span_b).astype("float64")
        promise = common.guaranteed_minutes(p, 0.90)
        whatif_rows.append({
            "가정한 날": TYPE_NAMES[s],
            "정시 확률": round(float(p[:, 0].mean()), 4),
            "미도착 확률": round(float(p[:, config.NOARRIVE].mean()), 4),
            "90% 보장 지연(분)": round(float(np.nanmean(promise)), 1),
            "보장 불가 비율": round(float(np.isnan(promise).mean()), 4),
        })
        say(f"  {TYPE_NAMES[s]}: 정시 {whatif_rows[-1]['정시 확률']:.3f}  "
            f"90% 보장 +{whatif_rows[-1]['90% 보장 지연(분)']:.0f}분")

    whatif = pd.DataFrame(whatif_rows)
    whatif.to_csv(config.RESULTS_DIR / "step14_whatif.csv",
                  index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- #
    # figures
    # ---------------------------------------------------------------- #
    part = table[table["model"].str.contains("같은 부분집합|합성")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, column, title in ((axes[0], "log_loss", "val log loss (낮을수록 좋음)"),
                              (axes[1], "auc_late", "AUC (높을수록 좋음)")):
        colours = ["#c92a2a" if v == "아니오" else "#4c6ef5"
                   for v in part["배포 가능"]]
        bars = ax.bar(range(len(part)), part[column], color=colours)
        ax.set_xticks(range(len(part)),
                      [m.replace(" (같은 부분집합)", "").replace(" Σ C(s|x)·B(y|x,s)", "")
                       for m in part["model"]], fontsize=9)
        for bar, v in zip(bars, part[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.5f}",
                    ha="center", va="bottom", fontsize=9)
        low, high = part[column].min(), part[column].max()
        pad = (high - low) * 0.5 or high * 0.01
        ax.set_ylim(max(0, low - pad), high + pad)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Step 14 — 빨강은 배포 불가(참 유형을 쥐여 준 오라클)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step14_models.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.6, 5))
    x = np.arange(5)
    ax1.bar(x - 0.2, whatif["90% 보장 지연(분)"], width=0.4, color="#e8590c",
            label="90% 보장 지연(분)")
    ax1.set_ylabel("90% 보장 지연(분)", color="#e8590c")
    ax1.set_xticks(x, whatif["가정한 날"], fontsize=10)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, whatif["보장 불가 비율"], width=0.4, color="#4c6ef5",
            label="보장 불가 비율")
    ax2.set_ylabel("90% 보장 자체가 불가능한 편의 비율", color="#4c6ef5")
    ax2.grid(False)
    ax1.set_title("Step 14 — 가정이 바뀌면 답이 바뀐다 (같은 편, 유형 칸만 교체)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step14_whatif.png")
    plt.close(fig)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print(table[["model", "배포 가능", "log_loss", "cdf_ece", "auc_late"]]
          .to_string(index=False))
    print("\n" + whatif.to_string(index=False))
    print("\n" + types.to_string(index=False))

    a = table[table["model"] == "A 무조건 P(y|x)"].iloc[0]
    b = table[table["model"].str.startswith("B 참")].iloc[0]
    comp = table[table["model"].str.startswith("C·B")].iloc[0]
    a_sub = table[table["model"].str.startswith("A 무조건 (")].iloc[0]
    print(f"\n1) 오라클 격차: AUC {a['auc_late']:.5f} → {b['auc_late']:.5f} "
          f"({b['auc_late'] - a['auc_late']:+.5f}). "
          "예약 시점 정보의 천장과 '그날을 안다면'의 차이 = 실시간 정보의 가격표.")
    print(f"2) 분해는 정보를 만들지 않는다: 합성 {comp['log_loss']:.5f} vs "
          f"무조건 {a_sub['log_loss']:.5f} "
          f"({comp['log_loss'] - a_sub['log_loss']:+.5f}).")
    beat = bool(types["beats_majority"].any())
    margins = ", ".join(f"{r.side} {r.margin:+.4f}" for r in types.itertuples())
    print(f"3) 그날 유형 예측 (최빈값 대비 여유: {margins}; 기준 "
          f"{MEANINGFUL_MARGIN:+.3f}): "
          + ("최빈값을 의미 있게 이긴다" if beat else
             "최빈값 찍기를 이기지 못한다")
          + " → 시나리오를 '예측'이 아니라 '가정'으로 둔 설계가 "
          + ("재검토 필요" if beat else "데이터로 정당화된다") + ".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
