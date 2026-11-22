"""Step 11d - tuning the MLP, so that "each model was pushed as far as it goes"
is a measured statement rather than a claim.

Reads  data/prep/ (step 9)
Writes results/step11d_mlp_width.csv
       results/step11d_mlp_search.csv
       figures/step11d_mlp_search.png

The tree got a depth sweep and a seven-point grid; the linear model got three
polynomial degrees and three regularisation strengths. The MLP had neither -
it was fitted once at 512-128-32 with Adam 3e-4 and early stopping. Comparing
a tuned tree with an untuned network is a comparison of effort, not of model
families, and it cuts the wrong way here: the untuned network already has the
best log loss, so leaving it untuned understates it.

Two sweeps, mirroring what the tree got:

  1. CAPACITY - the width of the hidden stack, which is the network's
     equivalent of max_depth. Same question: where does extra capacity stop
     buying validation loss and start buying training loss only?

  2. LEARNING RATE - at the best width. Adam's step size is the one knob that
     routinely moves a network by more than noise.

Early stopping stays on everywhere (the best epoch is reported), so a
configuration that overfits is not punished for epochs it never used.

Everything is fitted on train 2022-24 and scored on validation 2025.
Test 2026 is not opened here.

Usage:  python src/step11d_mlp_search.py [--epochs N]
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

WIDTHS = {
    "작게 256-64": (256, 64),
    "기본 512-128-32": (512, 128, 32),
    "크게 1024-256-64": (1024, 256, 64),
}
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
BASE_LR = 3e-4
DROPOUT = 0.2
BATCH = 8192


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build(n_features: int, widths: tuple[int, ...]):
    from torch import nn

    layers, last = [], n_features
    for i, w in enumerate(widths):
        layers += [nn.Linear(last, w), nn.ReLU()]
        if i < len(widths) - 1:          # no dropout right before the head
            layers.append(nn.Dropout(DROPOUT))
        last = w
    layers.append(nn.Linear(last, config.N_CLASS))
    return nn.Sequential(*layers)


def fit_one(train, val, widths: tuple[int, ...], lr: float,
            epochs: int) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(config.SEED)
    model = build(train["X_linear"].shape[1], widths)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_train = train["X_linear"]
    y_train = torch.from_numpy(train["cls"].astype(np.int64))
    order = np.arange(X_train.shape[0])
    rng = np.random.default_rng(config.SEED)

    def predict(matrix) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, matrix.shape[0], 65536):
                chunk = matrix[start:start + 65536].toarray().astype(np.float32)
                out.append(torch.softmax(model(torch.from_numpy(chunk)),
                                         dim=1).numpy())
        return np.concatenate(out)

    started = time.time()
    best = (np.inf, None, -1, np.inf)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), BATCH):
            index = np.sort(order[start:start + BATCH])
            chunk = torch.from_numpy(X_train[index].toarray().astype(np.float32))
            optimiser.zero_grad()
            loss = loss_fn(model(chunk), y_train[index])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
        train_loss = total / len(order)
        probs = predict(val["X_linear"])
        val_loss = float(-np.log(np.clip(
            probs[np.arange(len(val["cls"])), val["cls"]], 1e-9, 1)).mean())
        if val_loss < best[0]:
            best = (val_loss, probs, epoch, train_loss)

    scores = common.evaluate(best[1], val["cls"], val["delay"])
    return {"hidden": "-".join(str(w) for w in widths),
            "params": sum(p.numel() for p in model.parameters()),
            "lr": lr, "best_epoch": best[2],
            "train_loss": round(best[3], 5),
            **common.flat(scores),
            "gap": round(best[0] - best[3], 5),
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
    say(f"  train {train['X_linear'].shape[0]:,} x "
        f"{train['X_linear'].shape[1]} 열")

    # ---- 1. capacity ---------------------------------------------------- #
    say("은닉층 폭 쓸기 (트리의 max_depth 에 해당) ...")
    width_rows = []
    for label, widths in WIDTHS.items():
        row = {"설정": label, **fit_one(train, val, widths, BASE_LR,
                                      args.epochs)}
        width_rows.append(row)
        say(f"  {label}: val {row['log_loss']}  train {row['train_loss']}  "
            f"격차 {row['gap']}  ({row['params']:,} 파라미터, "
            f"{row['fit_seconds']:.0f}초)")
    widths_table = pd.DataFrame(width_rows)
    widths_table.to_csv(config.RESULTS_DIR / "step11d_mlp_width.csv",
                        index=False, encoding="utf-8-sig")

    best_label = str(widths_table.loc[widths_table["log_loss"].idxmin(), "설정"])
    best_widths = WIDTHS[best_label]
    say(f"가장 좋은 폭: {best_label} → 학습률 쓸기")

    # ---- 2. learning rate ----------------------------------------------- #
    lr_rows = []
    for lr in LEARNING_RATES:
        if lr == BASE_LR:
            row = dict(widths_table.loc[widths_table["설정"] == best_label]
                       .iloc[0])
            row["설정"] = f"lr {lr:g}"
        else:
            row = {"설정": f"lr {lr:g}",
                   **fit_one(train, val, best_widths, lr, args.epochs)}
            say(f"  lr {lr:g}: val {row['log_loss']}  "
                f"({row['fit_seconds']:.0f}초)")
        lr_rows.append(row)
    lr_table = pd.DataFrame(lr_rows)

    table = pd.concat([widths_table, lr_table], ignore_index=True)
    out = config.RESULTS_DIR / "step11d_mlp_search.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")

    # ---- figure ---------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(widths_table["params"], widths_table["train_loss"], "o-",
             label="train", color="#4c6ef5")
    ax1.plot(widths_table["params"], widths_table["log_loss"], "s-",
             label="val", color="#c92a2a")
    for x, y, lab in zip(widths_table["params"], widths_table["log_loss"],
                         widths_table["설정"]):
        ax1.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("파라미터 수 (로그)")
    ax1.set_ylabel("cross entropy")
    ax1.set_title("MLP 용량을 키우면 — train 만 내려간다면 과적합", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.semilogx(lr_table["lr"], lr_table["log_loss"], "o-", color="#e8590c")
    for x, y in zip(lr_table["lr"], lr_table["log_loss"]):
        ax2.annotate(f"{y:.5f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9)
    ax2.set_xlabel("Adam 학습률 (로그)")
    ax2.set_ylabel("val log loss")
    ax2.set_title(f"{best_label} 에서의 학습률", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.suptitle("Step 11d — MLP 도 끝까지 밀어 본다 (val 2025)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11d_mlp_search.png")
    plt.close(fig)

    print("\n" + "=" * 100)
    print(table[["설정", "hidden", "lr", "params", "best_epoch",
                 "train_loss", "log_loss", "cdf_ece", "auc_late", "gap"]]
          .to_string(index=False))
    span = float(table["log_loss"].max() - table["log_loss"].min())
    best = table.loc[table["log_loss"].idxmin()]
    print(f"\n탐색한 설정 {len(table)}종의 폭: {span:.5f}")
    print(f"최선: {best['설정']} ({best['hidden']}, lr {best['lr']:g}) "
          f"val log loss {best['log_loss']}  CDF 오차 {best['cdf_ece']}")
    print(f"\nresults: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
