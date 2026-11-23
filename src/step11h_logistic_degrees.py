"""Step 11h - the logistic search, re-run to convergence.

Reads  data/prep/ (step 9)
Writes results/step11h_logistic_degrees.csv
       figures/step11h_logistic_degrees.png

Step 11e settled the question at degree 1, C=1: lbfgs converges at 151
iterations, and the extra 31 iterations past the old cap of 120 bought -0.00003
of log loss, an eighth of the pre-registered threshold. Stopping at 120 had
cost nothing measurable.

That is one configuration out of five. Step 11c capped all five at 120, and
degrees 2 and 3 expand the five numeric columns to twenty and fifty-five - a
wider problem can take more iterations to settle, so the degree comparison
might be comparing three differently-unfinished fits. Degree 3 came last by
0.00259; it would have to find 0.00259 in its remaining iterations to change
the ranking, which is eighty times what degree 1 found. Unlikely is not
measured, so this measures it, and the two C settings with it.

After this step every logistic configuration has run to convergence, every
XGBoost configuration has been fitted on the full training set (step 11g), and
every MLP configuration has its seed spread measured (step 11f) - so the model
comparison is between models rather than between search budgets.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11h_logistic_degrees.py [--max-iter 400]
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

import step11c_polynomial as poly

# (degree, C) -> what step 11c got with the cap at 120 iterations
CAPPED = {(1, 1.0): 1.40592, (2, 1.0): 1.40613, (3, 1.0): 1.40846,
          (1, 0.1): 1.40594, (1, 10.0): 1.40589}
THRESHOLD = 0.000304   # sd 0.000152 x 2, step11c_seed_check.csv base arm


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    from sklearn.linear_model import LogisticRegression

    # the same inputs and the same expansion step 11c used, so the only thing
    # that differs between the two tables is the iteration cap
    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    say("loading ...")
    train = common.load_split("train", args.limit, drop=drop)
    val = common.load_split("val", args.limit, drop=drop)

    cat_train, num_train, names = poly.split_blocks(train["X_linear"],
                                                    train["linear_columns"])
    cat_val, num_val, _ = poly.split_blocks(val["X_linear"],
                                            val["linear_columns"])
    say(f"  one-hot {cat_train.shape[1]}열 · 수치 {num_train.shape[1]}열 "
        f"{names}")

    rows = []
    path = config.RESULTS_DIR / "step11h_logistic_degrees.csv"
    path.unlink(missing_ok=True)
    cache: dict = {}

    for degree, c in CAPPED:
        if degree not in cache:
            cache.clear()               # one expansion in memory at a time
            cache[degree] = poly.expand(num_train, num_val, degree)
        poly_train, poly_val, _ = cache[degree]
        X_train = sparse.hstack([cat_train, sparse.csr_matrix(poly_train)],
                                format="csr")
        X_val = sparse.hstack([cat_val, sparse.csr_matrix(poly_val)],
                              format="csr")

        model = LogisticRegression(C=c, max_iter=args.max_iter,
                                   solver="lbfgs", random_state=config.SEED)
        started = time.time()
        model.fit(X_train, train["cls"])
        seconds = time.time() - started
        used = int(model.n_iter_[0])
        scores = common.evaluate(model.predict_proba(X_val), val["cls"],
                                 val["delay"])
        gained = round(CAPPED[(degree, c)] - scores["log_loss"], 5)
        row = {"설정": f"차수 {degree} · C {c:g}", "degree": degree, "C": c,
               "numeric_columns": int(poly_train.shape[1]),
               "total_columns": int(X_train.shape[1]),
               "iterations_used": used,
               "converged": bool(used < args.max_iter),
               **common.flat(scores),
               "capped_120_log_loss": CAPPED[(degree, c)],
               "gained_by_converging": gained,
               "fit_seconds": round(seconds, 1)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        say(f"  {row['설정']} ({row['numeric_columns']}열): {used}회 사용, "
            f"수렴={row['converged']}  log loss {row['log_loss']} "
            f"(120회 {CAPPED[(degree, c)]}, {gained:+.5f})  "
            f"CDF 오차 {row['cdf_ece']}  ({row['fit_seconds']:.0f}초)")
        del X_train, X_val

    table = pd.DataFrame(rows)

    # ---- figure ---------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.7))
    deg = table[table["C"] == 1.0].sort_values("degree").reset_index(drop=True)
    width = 0.36
    x = np.arange(len(deg))
    ax1.bar(x - width / 2, deg["capped_120_log_loss"], width,
            label="120회에서 끊음", color="#adb5bd")
    ax1.bar(x + width / 2, deg["log_loss"], width, label="수렴까지",
            color="#4c6ef5")
    for i, row in deg.iterrows():
        ax1.annotate(f"{row['iterations_used']}회",
                     (i + width / 2, row["log_loss"]),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=8)
    low = float(min(deg["log_loss"].min(), deg["capped_120_log_loss"].min()))
    high = float(max(deg["log_loss"].max(), deg["capped_120_log_loss"].max()))
    ax1.set_ylim(low - (high - low) * 0.35, high + (high - low) * 0.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"차수 {d}\n({c}열)" for d, c
                         in zip(deg["degree"], deg["numeric_columns"])])
    ax1.set_ylabel("val log loss")
    ax1.set_title("끝까지 돌려도 차수 순위는 그대로인가", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, axis="y")

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
    ax2.set_title("설정 5종 — 빨간 원 = log loss 로 고른 것", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.suptitle("Step 11h — 로지스틱 탐색을 수렴까지 (val 2025)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11h_logistic_degrees.png")
    plt.close(fig)

    # ---- what it says ---------------------------------------------------- #
    print("\n" + "=" * 104)
    print(table[["설정", "numeric_columns", "iterations_used", "converged",
                 "log_loss", "capped_120_log_loss", "gained_by_converging",
                 "cdf_ece", "auc_late"]].to_string(index=False))

    before = deg.sort_values("capped_120_log_loss")["degree"].tolist()
    after = deg.sort_values("log_loss")["degree"].tolist()
    print(f"\n120회 기준 차수 순위: {before}")
    print(f"수렴 기준 차수 순위: {after}")
    print("순위가 같다 — 차수 비교는 그대로 쓴다." if before == after
          else "순위가 바뀐다 — 차수 비교를 수렴 기준으로 다시 쓴다.")

    biggest = table.loc[table["gained_by_converging"].abs().idxmax()]
    print(f"수렴으로 가장 많이 움직인 설정: {biggest['설정']} "
          f"({biggest['gained_by_converging']:+.5f}, 임계값 "
          f"{abs(biggest['gained_by_converging']) / THRESHOLD:.1f}배)")
    print(f"log loss 로 고른 설정: {pick['설정']}  "
          f"log loss {pick['log_loss']}  CDF 오차 {pick['cdf_ece']}")
    by_ece = table.loc[table["cdf_ece"].idxmin()]
    print(f"CDF 오차로 고르면: {by_ece['설정']}  "
          f"log loss {by_ece['log_loss']}  CDF 오차 {by_ece['cdf_ece']}")
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
