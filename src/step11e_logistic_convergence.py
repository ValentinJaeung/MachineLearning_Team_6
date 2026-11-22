"""Step 11e - letting the logistic model finish.

Reads  data/prep/ (step 9)
Writes results/step11e_logistic_convergence.csv
       figures/step11e_logistic_convergence.png

Step 11 reports the logistic model at max_iter=120, where lbfgs had not yet
converged. The tree is allowed to stop at its own best iteration and the
network at its own best epoch, so stopping the linear model at an arbitrary
120 is not the same treatment - it is the one model being compared before it
finished.

This step runs it to convergence (or until the extra iterations stop buying
anything) and reports the number the comparison should use.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11e_logistic_convergence.py [--iters 120,300,600]
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


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fit(train, val, max_iter: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs",
                               random_state=config.SEED)
    started = time.time()
    model.fit(train["X_linear"], train["cls"])
    seconds = time.time() - started
    probs = model.predict_proba(val["X_linear"])
    scores = common.evaluate(probs, val["cls"], val["delay"])
    used = int(model.n_iter_[0])
    return {"max_iter": max_iter, "iterations_used": used,
            "converged": bool(used < max_iter),
            **common.flat(scores),
            "fit_seconds": round(seconds, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=str, default="120,300,600")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    budgets = [int(v) for v in args.iters.split(",")]

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    say("loading ...")
    train = common.load_split("train", args.limit)
    val = common.load_split("val", args.limit)

    rows = []
    for budget in budgets:
        row = fit(train, val, budget)
        rows.append(row)
        say(f"  max_iter {budget}: {row['iterations_used']}회 사용, "
            f"수렴={row['converged']}  log loss {row['log_loss']}  "
            f"CDF 오차 {row['cdf_ece']}  ({row['fit_seconds']:.0f}초)")
        if row["converged"]:
            say("  수렴했다 — 더 늘려도 같은 해에 도달한다")
            break

    table = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "step11e_logistic_convergence.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(table["iterations_used"], table["log_loss"], "o-", color="#4c6ef5")
    for x, y, c in zip(table["iterations_used"], table["log_loss"],
                       table["converged"]):
        ax.annotate(f"{y:.5f}" + ("  (수렴)" if c else ""), (x, y),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9)
    ax.set_xlabel("lbfgs 반복 횟수")
    ax.set_ylabel("val log loss")
    ax.set_title("Step 11e — 로지스틱을 끝까지 돌리면 (val 2025)", fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11e_logistic_convergence.png")
    plt.close(fig)

    print("\n" + "=" * 90)
    print(table[["max_iter", "iterations_used", "converged", "log_loss",
                 "cdf_ece", "auc_late", "fit_seconds"]].to_string(index=False))
    first, last = table.iloc[0], table.iloc[-1]
    print(f"\n120회에서 {first['log_loss']} → "
          f"{last['iterations_used']}회에서 {last['log_loss']} "
          f"({last['log_loss'] - first['log_loss']:+.5f})")
    print("비교에 쓸 숫자는 끝까지 돌린 쪽이다.")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
