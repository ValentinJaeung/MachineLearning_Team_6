"""
Step 11c (part 2) - is the tree's polynomial gain real, or column-draw luck?

Reads  data/prep/ (step 9)
Writes results/step11c_seed_check.csv

Why this exists
---------------
step11c found that giving XGBoost the fifteen degree-2 columns moved val
log loss from 1.40473 to 1.40427, a gain of 0.00046. Our pre-registered
decision threshold is 0.000265, so by that rule this would count as a real
change - and the honest thing is to say so rather than to wave it away.

But the threshold does not strictly apply to this comparison. It was
measured in step 10 by re-seeding a model whose COLUMN SET never changed.
Here the column set grows from 12 to 27, and the deployed configuration
uses colsample_bytree = 0.8: every boosting round draws 80% of the columns
at random. Adding fifteen columns therefore changes what gets drawn, round
after round, which is a source of variation the 0.000265 figure never saw.

So the threshold has to be re-measured for THIS comparison. Both arms are
fitted at three seeds on the full training set, and the gap is judged
against the spread the seeds themselves produce.

  arm "base"  : the twelve inputs, exactly as step 11 fits them
  arm "poly"  : the same, plus the fifteen degree-2 columns built from the
                five numeric inputs

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11c_seed_check.py [--seeds 42,1,2]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common
import config

ROUNDS = 400
EARLY_STOP = 40


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def numeric_block(split) -> tuple[np.ndarray, list[str]]:
    """The five numeric columns, read back from the linear design matrix."""
    columns = split["linear_columns"]
    at = [i for i, c in enumerate(columns) if c.startswith("num__")]
    names = [columns[i].split("__", 1)[1] for i in at]
    return np.asarray(split["X_linear"][:, at].todense()), names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,1,2")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import xgboost as xgb
    from sklearn.preprocessing import PolynomialFeatures

    config.set_seeds()
    config.ensure_dirs()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    say("loading ...")
    train = common.load_split("train", args.limit, drop=drop)
    val = common.load_split("val", args.limit, drop=drop)

    num_train, names = numeric_block(train)
    num_val, _ = numeric_block(val)
    say(f"  수치 {len(names)}열 {names}")

    # the fifteen extra columns: squares and pairwise products only
    poly = PolynomialFeatures(degree=2, include_bias=False)
    extra_train = poly.fit_transform(num_train)[:, num_train.shape[1]:]
    extra_val = poly.transform(num_val)[:, num_train.shape[1]:]
    say(f"  확장 {extra_train.shape[1]}열 추가")

    frames = {"base": (train["X_xgb"], val["X_xgb"])}
    poly_train = train["X_xgb"].copy()
    poly_val = val["X_xgb"].copy()
    for i in range(extra_train.shape[1]):
        poly_train[f"poly_{i}"] = extra_train[:, i].astype("float32")
        poly_val[f"poly_{i}"] = extra_val[:, i].astype("float32")
    frames["poly"] = (poly_train, poly_val)
    del extra_train, extra_val

    rows = []
    for arm, (frame_train, frame_val) in frames.items():
        dtrain = xgb.DMatrix(frame_train, label=train["cls"],
                             enable_categorical=True)
        dval = xgb.DMatrix(frame_val, label=val["cls"],
                           enable_categorical=True)
        for seed in seeds:
            params = {"objective": "multi:softprob",
                      "num_class": config.N_CLASS, "eval_metric": "mlogloss",
                      "tree_method": "hist", "max_depth": 6, "eta": 0.08,
                      "min_child_weight": 50, "subsample": 0.8,
                      "colsample_bytree": 0.8, "seed": seed}
            started = time.time()
            booster = xgb.train(params, dtrain, num_boost_round=ROUNDS,
                                evals=[(dval, "val")],
                                early_stopping_rounds=EARLY_STOP,
                                verbose_eval=False)
            probs = booster.predict(
                dval, iteration_range=(0, booster.best_iteration + 1)
            ).astype("float64")
            scores = common.evaluate(probs, val["cls"], val["delay"])
            rows.append({"arm": arm, "seed": seed,
                         "columns": int(frame_train.shape[1]),
                         **common.flat(scores),
                         "best_iteration": int(booster.best_iteration),
                         "fit_seconds": round(time.time() - started, 1)})
            say(f"  [{arm}] seed {seed}: log loss {scores['log_loss']}  "
                f"CDF 오차 {scores['cdf_ece']}  AUC {scores['auc_late']}")
        del dtrain, dval

    table = pd.DataFrame(rows)
    path = config.RESULTS_DIR / "step11c_seed_check.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(table.to_string(index=False))

    base = table[table["arm"] == "base"]["log_loss"]
    poly_ll = table[table["arm"] == "poly"]["log_loss"]
    spread = float(pd.concat([base - base.mean(),
                              poly_ll - poly_ll.mean()]).std(ddof=1))
    gap = float(poly_ll.mean() - base.mean())
    print(f"\n기본 {base.mean():.5f} (최소 {base.min():.5f} 최대 {base.max():.5f})")
    print(f"확장 {poly_ll.mean():.5f} (최소 {poly_ll.min():.5f} 최대 {poly_ll.max():.5f})")
    print(f"평균 차이 {gap:+.5f}   시드 잡음 sd {spread:.6f}   임계값 {spread * 2:.6f}")
    verdict = "실재" if abs(gap) > spread * 2 else "구별 불가"
    print(f"→ 판정: {verdict}")
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
