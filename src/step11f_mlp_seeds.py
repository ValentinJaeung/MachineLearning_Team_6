"""Step 11f - is the MLP's calibration a property of the model, or of where
early stopping happened to stop?

Reads  data/prep/ (step 9)
Writes results/step11f_mlp_seeds.csv
       results/step11f_mlp_epochs.csv
       figures/step11f_mlp_seeds.png

Why this step exists.

Step 11d swept the MLP and one configuration (256-64, Adam 1e-3) came out at
CDF calibration error 0.00500 - better than the tree's 0.00654, which is the
metric section 4.4 declared as the selection metric before any numbers were
seen. If that 0.00500 is real, the chosen model has to change.

Two things make it suspect, and both are measurable rather than arguable:

  1. The network's early stopping picks its epoch by validation LOG LOSS, and
     the validation log loss curve is almost flat (1.40034 - 1.40353 across
     eight epochs). A nearly flat curve means the winning epoch is close to
     arbitrary - and the calibration error that comes along with that epoch is
     then arbitrary too.

  2. The pre-registered noise threshold (sd 0.000152, so 0.000304) was measured
     on the TREE's log loss. Nobody measured the seed-to-seed spread of the
     NETWORK's calibration error, which is the quantity now being compared.

So: refit both candidate configurations under three seeds, and record the whole
per-epoch trace of both metrics. That answers it directly - if the seed spread
of cdf_ece swamps the 0.00154 gap to the tree, the two models are tied on the
selection metric and the sweep found a lucky epoch, not a better model.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11f_mlp_seeds.py [--epochs 8]
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

import step11d_mlp_search as mlp

CONFIGS = {"256-64 @ lr 3e-4": ((256, 64), 3e-4),
           "256-64 @ lr 1e-3": ((256, 64), 1e-3)}
SEEDS = [42, 1, 2]
XGB_ECE = 0.00654


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def quick(probs, cls, delay):
    """log loss and CDF calibration error only - no rank statistic."""
    p = np.clip(probs, 1e-9, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    n = len(cls)
    logloss = float(-np.log(p[np.arange(n), cls]).mean())
    cdf = p.cumsum(axis=1)
    arrived = cls < config.N_CLASS - 1
    ece = 0.0
    for i, edge in enumerate(common.EDGES):
        ece += abs(float(cdf[:, i].mean())
                   - float((arrived & (delay <= edge)).mean()))
    return logloss, ece / len(common.EDGES)


def run(train, val, widths, lr, seed, epochs):
    """One fit, returning the per-epoch trace and the early-stopping pick."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = mlp.build(train["X_linear"].shape[1], widths)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_train = train["X_linear"]
    y_train = torch.from_numpy(train["cls"].astype(np.int64))
    order = np.arange(X_train.shape[0])
    rng = np.random.default_rng(seed)

    def predict(matrix):
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, matrix.shape[0], 65536):
                chunk = matrix[start:start + 65536].toarray().astype(np.float32)
                out.append(torch.softmax(model(torch.from_numpy(chunk)),
                                         dim=1).numpy())
        return np.concatenate(out)

    started = time.time()
    trace = []
    best = (np.inf, None, -1)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), mlp.BATCH):
            index = np.sort(order[start:start + mlp.BATCH])
            chunk = torch.from_numpy(X_train[index].toarray().astype(np.float32))
            optimiser.zero_grad()
            loss = loss_fn(model(chunk), y_train[index])
            loss.backward()
            optimiser.step()
        probs = predict(val["X_linear"])
        logloss, ece = quick(probs, val["cls"], val["delay"])
        trace.append({"epoch": epoch, "log_loss": round(logloss, 5),
                      "cdf_ece": round(ece, 5)})
        if logloss < best[0]:
            best = (logloss, probs, epoch)

    scores = common.evaluate(best[1], val["cls"], val["delay"])
    return trace, {"best_epoch": best[2], **common.flat(scores),
                   "fit_seconds": round(time.time() - started, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    say("loading ...")
    train = common.load_split("train", args.limit)
    val = common.load_split("val", args.limit)

    rows, traces = [], []
    for label, (widths, lr) in CONFIGS.items():
        for seed in SEEDS:
            trace, result = run(train, val, widths, lr, seed, args.epochs)
            rows.append({"설정": label, "seed": seed, **result})
            for point in trace:
                traces.append({"설정": label, "seed": seed, **point})
            say(f"  {label}  seed {seed}: 최적 에폭 {result['best_epoch']}  "
                f"log loss {result['log_loss']}  "
                f"CDF 오차 {result['cdf_ece']}  "
                f"({result['fit_seconds']:.0f}초)")

    table = pd.DataFrame(rows)
    epochs = pd.DataFrame(traces)
    out = config.RESULTS_DIR / "step11f_mlp_seeds.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    epochs.to_csv(config.RESULTS_DIR / "step11f_mlp_epochs.csv",
                  index=False, encoding="utf-8-sig")

    # ---- figure ---------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colours = {42: "#4c6ef5", 1: "#e8590c", 2: "#2f9e44"}
    styles = {"256-64 @ lr 3e-4": "-", "256-64 @ lr 1e-3": "--"}
    for keys, part in epochs.groupby(["설정", "seed"], sort=False):
        label, seed = keys
        for ax, column in ((ax1, "log_loss"), (ax2, "cdf_ece")):
            ax.plot(part["epoch"], part[column], styles[label] + "o",
                    color=colours[seed], markersize=4, linewidth=1.3,
                    label=f"{label}  seed {seed}" if ax is ax1 else None)
    for _, row in table.iterrows():
        pick = epochs[(epochs["설정"] == row["설정"])
                      & (epochs["seed"] == row["seed"])
                      & (epochs["epoch"] == row["best_epoch"])]
        ax1.plot(pick["epoch"], pick["log_loss"], "*", color="#c92a2a",
                 markersize=15, zorder=5)
        ax2.plot(pick["epoch"], pick["cdf_ece"], "*", color="#c92a2a",
                 markersize=15, zorder=5)

    ax2.axhline(XGB_ECE, color="#343a40", linestyle=":", linewidth=1.6)
    ax2.annotate(f"XGBoost {XGB_ECE:.5f}", (args.epochs, XGB_ECE),
                 textcoords="offset points", xytext=(-6, 6), ha="right",
                 fontsize=9, color="#343a40")

    ax1.set_xlabel("에폭")
    ax1.set_ylabel("val log loss")
    ax1.set_title("조기 종료가 보는 값 — 거의 평평하다", fontsize=10)
    ax2.set_xlabel("에폭")
    ax2.set_ylabel("val CDF 보정오차")
    ax2.set_title("같은 구간에서 채택 지표는 요동친다", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)
    fig.suptitle("Step 11f — ★ = 조기 종료가 고른 에폭 (val 2025)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11f_mlp_seeds.png")
    plt.close(fig)

    # ---- verdict --------------------------------------------------------- #
    print("\n" + "=" * 92)
    print(table[["설정", "seed", "best_epoch", "log_loss", "cdf_ece",
                 "coverage_90", "auc_late"]].to_string(index=False))

    print("\n시드 3개의 표준편차 (같은 설정, 같은 데이터)")
    for label, part in table.groupby("설정", sort=False):
        sd_ll = float(part["log_loss"].std(ddof=1))
        sd_ece = float(part["cdf_ece"].std(ddof=1))
        print(f"  {label}: log loss sd {sd_ll:.5f} (2배 {2 * sd_ll:.5f})  "
              f"CDF 오차 sd {sd_ece:.5f} (2배 {2 * sd_ece:.5f})  "
              f"에폭 {sorted(part['best_epoch'])}")

    low = float(table["cdf_ece"].min())
    sd_all = float(table["cdf_ece"].std(ddof=1))
    print(f"\nXGBoost {XGB_ECE:.5f} 와 MLP 최선 {low:.5f} 의 차이: "
          f"{XGB_ECE - low:.5f}")
    print(f"MLP 설정·시드 전체의 CDF 오차 표준편차: {sd_all:.5f} "
          f"→ 2배 임계값 {2 * sd_all:.5f}")
    print("에폭별 변동폭:")
    for label, part in epochs.groupby("설정", sort=False):
        print(f"  {label}: {part['cdf_ece'].min():.5f} ~ "
              f"{part['cdf_ece'].max():.5f} "
              f"(폭 {part['cdf_ece'].max() - part['cdf_ece'].min():.5f})")
    print("차이가 임계값보다 작으면, 11d 가 찾은 것은 더 좋은 모델이 아니라 "
          "운 좋은 에폭이다.")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
