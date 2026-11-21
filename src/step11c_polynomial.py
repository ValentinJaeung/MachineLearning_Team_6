"""
Step 11c - do polynomial features help?

Reads  data/prep/ (step 9)
Writes results/step11c_polynomial.csv
       figures/step11c_polynomial.png

The question
------------
The linear model draws one straight line per feature. Five of the twelve
inputs enter it as plain numbers, so any curvature in them is invisible to
it. Expanding those five into squares and pairwise products gives the linear
model the ability to bend, without changing the model family.

This is the week-5 topic (특성공학 · 규제) applied to the project's own data,
and there is a concrete reason to expect a gain on ONE feature in particular:

    days_to_nearest_holiday is SIGNED (−21 … +21).
    Its effect is plausibly U-shaped - the days on either side of a holiday
    are busy, the days far from one are not. A linear term can only say
    "later is worse" or "earlier is worse"; it cannot say "near is worse".
    The squared term can.

The same argument does not apply to the tree or the MLP: a tree splits the
same feature repeatedly and builds the bend itself, and a hidden layer is a
non-linear function by construction. So the prediction is

    logistic  : should improve
    XGBoost   : should not change
    MLP       : should not change (not refitted here; the argument is structural)

Confirming that is the point. If the expansion lifts the linear model and
leaves the tree flat, that is direct evidence for the step 9 claim that the
two encodings exist because the model families need different things - and
not merely an assertion.

What is expanded
----------------
Only the five numeric columns:

    Distance · CRSElapsedTime · sched_pad · dest_arr_load ·
    days_to_nearest_holiday

The categorical block is left alone. Expanding 350 one-hot columns to degree
two would produce ~61,000 columns of almost entirely zeros, which is a
memory problem, not a modelling idea; and a product of two indicator columns
is just a rarer indicator, which the tree already forms by splitting twice.

Degrees 1 (baseline), 2 and 3 are compared, then the best degree is given a
regularisation sweep, because adding 50 correlated columns to a linear model
is exactly the situation regularisation exists for.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11c_polynomial.py [--degrees 1,2,3] [--skip-xgb]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

import common
import config
import plotting
from plotting import plt

C_GRID = [0.1, 1.0, 10.0]
MAX_ITER = 120


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def split_blocks(matrix, columns: list[str]) -> tuple:
    """Separate the one-hot block from the numeric block."""
    numeric_at = [i for i, c in enumerate(columns) if c.startswith("num__")]
    cat_at = [i for i, c in enumerate(columns) if not c.startswith("num__")]
    names = [columns[i].split("__", 1)[1] for i in numeric_at]
    return matrix[:, cat_at], np.asarray(matrix[:, numeric_at].todense()), names


def expand(train_num, val_num, degree: int):
    """Polynomial expansion fitted on train, applied to both splits."""
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if degree == 1:
        poly_train, poly_val = train_num, val_num
        names = None
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_train = poly.fit_transform(train_num)
        poly_val = poly.transform(val_num)
        names = poly.get_feature_names_out()

    scaler = StandardScaler().fit(poly_train)        # train only
    return (scaler.transform(poly_train).astype(np.float32),
            scaler.transform(poly_val).astype(np.float32), names)


def fit_logistic(X_train, y_train, X_val, cls_val, delay_val, c: float):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=c, max_iter=MAX_ITER, solver="lbfgs",
                               random_state=config.SEED)
    started = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - started
    probs = model.predict_proba(X_val)
    scores = common.evaluate(probs, cls_val, delay_val)
    return scores, {"fit_seconds": round(seconds, 1),
                    "iterations": int(model.n_iter_[0]),
                    "converged": bool(model.n_iter_[0] < MAX_ITER)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", type=str, default="1,2,3")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    degrees = [int(d) for d in args.degrees.split(",")]

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    say("loading ...")
    train = common.load_split("train", args.limit, drop=drop)
    val = common.load_split("val", args.limit, drop=drop)

    cat_train, num_train, names = split_blocks(train["X_linear"],
                                               train["linear_columns"])
    cat_val, num_val, _ = split_blocks(val["X_linear"], val["linear_columns"])
    say(f"  one-hot {cat_train.shape[1]}열 · 수치 {num_train.shape[1]}열 "
        f"{names}")

    rows = []
    path = config.RESULTS_DIR / "step11c_polynomial.csv"
    path.unlink(missing_ok=True)

    def record(row: dict) -> None:
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ #
    # 1. degree sweep for the linear model
    # ------------------------------------------------------------------ #
    for degree in degrees:
        poly_train, poly_val, poly_names = expand(num_train, num_val, degree)
        X_train = sparse.hstack([cat_train, sparse.csr_matrix(poly_train)],
                                format="csr")
        X_val = sparse.hstack([cat_val, sparse.csr_matrix(poly_val)],
                              format="csr")
        say(f"다항 차수 {degree}: 수치 {num_train.shape[1]} → "
            f"{poly_train.shape[1]}열, 전체 {X_train.shape[1]}열")
        scores, info = fit_logistic(X_train, train["cls"], X_val,
                                    val["cls"], val["delay"], c=1.0)
        record({"model": "M1 로지스틱", "degree": degree, "C": 1.0,
                "numeric_columns": int(poly_train.shape[1]),
                "total_columns": int(X_train.shape[1]),
                **common.flat(scores), **info})
        say(f"  -> log loss {scores['log_loss']}  CDF 오차 {scores['cdf_ece']}"
            f"  AUC {scores['auc_late']}  ({info['fit_seconds']}초)")
        del X_train, X_val, poly_train, poly_val

    # ------------------------------------------------------------------ #
    # 2. regularisation sweep at the best degree
    # ------------------------------------------------------------------ #
    table = pd.DataFrame(rows)
    best_degree = int(table.loc[table["log_loss"].idxmin(), "degree"])
    say(f"가장 좋은 차수 {best_degree} → 규제 강도 비교")
    if best_degree > 1:
        poly_train, poly_val, _ = expand(num_train, num_val, best_degree)
        X_train = sparse.hstack([cat_train, sparse.csr_matrix(poly_train)],
                                format="csr")
        X_val = sparse.hstack([cat_val, sparse.csr_matrix(poly_val)],
                              format="csr")
        for c in C_GRID:
            if c == 1.0:
                continue
            scores, info = fit_logistic(X_train, train["cls"], X_val,
                                        val["cls"], val["delay"], c=c)
            record({"model": "M1 로지스틱", "degree": best_degree, "C": c,
                    "numeric_columns": int(poly_train.shape[1]),
                    "total_columns": int(X_train.shape[1]),
                    **common.flat(scores), **info})
            say(f"  C={c}: log loss {scores['log_loss']}")
        del X_train, X_val, poly_train, poly_val

    # ------------------------------------------------------------------ #
    # 3. does the tree need it?
    # ------------------------------------------------------------------ #
    if not args.skip_xgb and best_degree > 1:
        import xgboost as xgb

        say("XGBoost: 같은 확장을 트리에 줘도 달라지는가")
        poly = None
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=best_degree, include_bias=False)
        extra_train = poly.fit_transform(num_train)[:, num_train.shape[1]:]
        extra_val = poly.transform(num_val)[:, num_train.shape[1]:]
        extra_names = [f"poly_{i}" for i in range(extra_train.shape[1])]

        frame_train = train["X_xgb"].copy()
        frame_val = val["X_xgb"].copy()
        for i, name in enumerate(extra_names):
            frame_train[name] = extra_train[:, i].astype("float32")
            frame_val[name] = extra_val[:, i].astype("float32")

        params = {"objective": "multi:softprob", "num_class": config.N_CLASS,
                  "eval_metric": "mlogloss", "tree_method": "hist",
                  "max_depth": 6, "eta": 0.08, "min_child_weight": 50,
                  "subsample": 0.8, "colsample_bytree": 0.8, "seed": config.SEED}
        dtrain = xgb.DMatrix(frame_train, label=train["cls"],
                             enable_categorical=True)
        dval = xgb.DMatrix(frame_val, label=val["cls"], enable_categorical=True)
        started = time.time()
        booster = xgb.train(params, dtrain, num_boost_round=400,
                            evals=[(dval, "val")], early_stopping_rounds=40,
                            verbose_eval=100)
        probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1)
                                ).astype("float64")
        scores = common.evaluate(probs, val["cls"], val["delay"])
        record({"model": "M2 XGBoost", "degree": best_degree, "C": None,
                "numeric_columns": int(extra_train.shape[1] + num_train.shape[1]),
                "total_columns": int(frame_train.shape[1]),
                **common.flat(scores),
                "fit_seconds": round(time.time() - started, 1),
                "iterations": int(booster.best_iteration)})
        say(f"  -> log loss {scores['log_loss']}  AUC {scores['auc_late']}")

    # ------------------------------------------------------------------ #
    table = pd.DataFrame(rows)
    linear = table[table["model"] == "M1 로지스틱"]
    sweep = linear[linear["C"] == 1.0].sort_values("degree")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(sweep["degree"], sweep["log_loss"], "o-", color="#4c6ef5")
    for d, v in zip(sweep["degree"], sweep["log_loss"]):
        axes[0].annotate(f"{v:.5f}", (d, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)
    axes[0].set_xticks(sweep["degree"])
    axes[0].set_xlabel("다항 차수 (수치형 5변수만)")
    axes[0].set_ylabel("val log loss")
    axes[0].set_title("로지스틱 — 곡선을 그릴 수 있게 해 주면")

    reg = linear[linear["degree"] == best_degree].sort_values("C")
    if len(reg) > 1:
        axes[1].semilogx(reg["C"], reg["log_loss"], "s-", color="#e8590c")
        axes[1].set_xlabel("C (클수록 규제 약함)")
        axes[1].set_ylabel("val log loss")
        axes[1].set_title(f"차수 {best_degree} 에서의 규제 강도")
    else:
        axes[1].axis("off")
    fig.suptitle("Step 11c — 다항 특성이 도움이 되는가", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11c_polynomial.png")
    plt.close(fig)

    print("\n" + "=" * 100)
    print(table[["model", "degree", "C", "numeric_columns", "total_columns",
                 "log_loss", "cdf_ece", "auc_late", "fit_seconds"]]
          .to_string(index=False))

    base = float(sweep.loc[sweep["degree"] == 1, "log_loss"].iloc[0])
    best = float(linear["log_loss"].min())
    print(f"\n로지스틱: 차수 1 {base:.5f} → 최선 {best:.5f} ({best - base:+.5f})")
    tree = table[table["model"] == "M2 XGBoost"]
    if not tree.empty:
        print(f"XGBoost: 같은 확장을 줬을 때 {float(tree['log_loss'].iloc[0]):.5f} "
              f"— 11단계의 기본 입력 결과와 비교할 것")
    print("\n트리와 MLP 는 비선형을 스스로 만든다. 다항 확장이 선형 모델만 움직인다면,")
    print("그것이 9단계에서 두 벌로 전처리한 이유를 주장이 아니라 측정으로 보여 준다.")
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
