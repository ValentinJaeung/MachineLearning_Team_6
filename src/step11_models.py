"""
Step 11 - three models, one split, one set of metrics.

Reads  data/prep/ (step 9)
Writes data/model/{model}.*            fitted models and validation predictions
       results/step11_models.csv       the comparison table
       results/step11_reliability.csv  predicted vs actual at every CDF edge
       results/step11_xgb_depth.csv    overfitting against tree depth
       results/step11_xgb_search.csv   hyper-parameter search
       results/step11_curves_*.csv     learning curves
       figures/step11_*.png            four figures

The three models
----------------
  M1 다항 로지스틱 회귀   softmax over the one-hot matrix          (week 6)
  M2 XGBoost            multi:softprob over the category frame   (week 10)
  M3 MLP                softmax head on the one-hot matrix       (deep learning)

All three output the same ten probabilities. That is what makes them
comparable at all: the service reads a distribution, so a model that emitted
a single number could not be scored on the same axis.

Everything is scored on validation 2025. Test 2026 stays shut until step 15.

What to look for besides the score
----------------------------------
- how CLOSE the three land. If they converge, the ceiling is the feature set,
  not the model, and more tuning is the wrong place to spend effort.
- how the train-validation gap grows with tree depth (overfitting).
- fitting time, which decides how many experiments are affordable at all.
- whether the model can be explained (step 13 needs exact TreeSHAP).

The searches run on a fixed one-third sample of train so that many settings
can be compared; the three reported models are fitted on all 5.4M rows.

Usage:  python src/step11_models.py [--skip logistic,mlp,depth,search]
                                    [--rounds N] [--epochs N] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

SAMPLE = 3                  # every 3rd row, for the depth and search studies
SEARCH_ROUNDS = 220
SEARCH_ETA = 0.15


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def chosen_features() -> list[str]:
    """Nothing is dropped: step 9 writes exactly the twelve inputs."""
    print(f"[입력] {len(config.MODEL_FEATURES)}개: {config.MODEL_FEATURES}")
    return []


def append(path, row: dict) -> None:
    """Write results as they arrive, so an interrupted run still leaves output."""
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).to_csv(
        path, index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# M2 XGBoost
# --------------------------------------------------------------------------- #


def xgb_params(seed: int = config.SEED, **over) -> dict:
    params = {
        "objective": "multi:softprob", "num_class": config.N_CLASS,
        "eval_metric": "mlogloss", "tree_method": "hist",
        "max_depth": 6, "eta": 0.08, "min_child_weight": 50,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
    }
    params.update(over)
    return params


def run_xgboost(train, val, rounds: int) -> tuple[np.ndarray, dict]:
    import xgboost as xgb

    dtrain = xgb.DMatrix(train["X_xgb"], label=train["cls"], enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"], enable_categorical=True)
    history: dict = {}
    started = time.time()
    booster = xgb.train(xgb_params(), dtrain, num_boost_round=rounds,
                        evals=[(dtrain, "train"), (dval, "val")],
                        evals_result=history, early_stopping_rounds=40,
                        verbose_eval=50)
    seconds = time.time() - started
    pd.DataFrame({
        "round": np.arange(1, len(history["train"]["mlogloss"]) + 1),
        "train_mlogloss": history["train"]["mlogloss"],
        "val_mlogloss": history["val"]["mlogloss"],
    }).to_csv(config.RESULTS_DIR / "step11_curves_xgb.csv",
              index=False, encoding="utf-8-sig")

    booster.save_model(config.MODEL_DIR / "xgboost.json")
    best = int(booster.best_iteration)
    probs = booster.predict(dval, iteration_range=(0, best + 1))
    return probs, {"fit_seconds": round(seconds, 1), "best_iteration": best,
                   "train_mlogloss": round(history["train"]["mlogloss"][best], 5)}


def depth_study(train, val) -> None:
    """How the train-validation gap opens up as the trees get deeper."""
    import xgboost as xgb

    index = np.arange(0, len(train["cls"]), SAMPLE)
    dtrain = xgb.DMatrix(train["X_xgb"].iloc[index], label=train["cls"][index],
                         enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"], enable_categorical=True)
    path = config.RESULTS_DIR / "step11_xgb_depth.csv"
    path.unlink(missing_ok=True)
    for depth in (3, 6, 8, 10, 12):
        history: dict = {}
        booster = xgb.train(xgb_params(max_depth=depth, eta=SEARCH_ETA), dtrain,
                            num_boost_round=SEARCH_ROUNDS,
                            evals=[(dtrain, "train"), (dval, "val")],
                            evals_result=history, early_stopping_rounds=20,
                            verbose_eval=False)
        best = int(booster.best_iteration)
        row = {"max_depth": depth,
               "train_mlogloss": round(history["train"]["mlogloss"][best], 5),
               "val_mlogloss": round(history["val"]["mlogloss"][best], 5),
               "gap": round(history["val"]["mlogloss"][best]
                            - history["train"]["mlogloss"][best], 5),
               "best_iteration": best}
        append(path, row)
        say(f"  depth {depth:>2}: val {row['val_mlogloss']}  "
            f"train {row['train_mlogloss']}  gap {row['gap']}")


def hyper_search(train, val) -> None:
    """A small grid, so the comparison is not against an untuned opponent."""
    import xgboost as xgb

    index = np.arange(0, len(train["cls"]), SAMPLE)
    dtrain = xgb.DMatrix(train["X_xgb"].iloc[index], label=train["cls"][index],
                         enable_categorical=True)
    dval = xgb.DMatrix(val["X_xgb"], label=val["cls"], enable_categorical=True)
    path = config.RESULTS_DIR / "step11_xgb_search.csv"
    path.unlink(missing_ok=True)

    grid = ([{"eta": e, "min_child_weight": 50} for e in (0.05, 0.10, 0.20)]
            + [{"eta": SEARCH_ETA, "min_child_weight": m} for m in (10, 200)]
            + [{"eta": SEARCH_ETA, "min_child_weight": 50,
                "colsample_bytree": c} for c in (0.5, 1.0)])
    for setting in grid:
        history: dict = {}
        booster = xgb.train(xgb_params(**setting), dtrain,
                            num_boost_round=SEARCH_ROUNDS,
                            evals=[(dval, "val")], evals_result=history,
                            early_stopping_rounds=20, verbose_eval=False)
        best = int(booster.best_iteration)
        row = {**xgb_params(**setting), "best_iteration": best,
               "val_mlogloss": round(history["val"]["mlogloss"][best], 5)}
        for drop in ("objective", "num_class", "eval_metric", "tree_method", "seed"):
            row.pop(drop)
        append(path, row)
        say(f"  {setting}: val {row['val_mlogloss']}")


# --------------------------------------------------------------------------- #
# M1 logistic
# --------------------------------------------------------------------------- #


def run_logistic(train, val, max_iter: int) -> tuple[np.ndarray, dict]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=max_iter, solver="lbfgs",
                               random_state=config.SEED)
    started = time.time()
    model.fit(train["X_linear"], train["cls"])
    seconds = time.time() - started
    probs = model.predict_proba(val["X_linear"])
    np.save(config.MODEL_DIR / "logistic_coef.npy", model.coef_)
    return probs, {"fit_seconds": round(seconds, 1),
                   "iterations": int(model.n_iter_[0]),
                   "converged": bool(model.n_iter_[0] < max_iter)}


# --------------------------------------------------------------------------- #
# M3 MLP
# --------------------------------------------------------------------------- #


def run_mlp(train, val, epochs: int) -> tuple[np.ndarray, dict]:
    import torch
    from torch import nn

    torch.manual_seed(config.SEED)
    n_features = train["X_linear"].shape[1]
    model = nn.Sequential(
        nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 32), nn.ReLU(),
        nn.Linear(32, config.N_CLASS),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    X_train = train["X_linear"]
    y_train = torch.from_numpy(train["cls"].astype(np.int64))
    order = np.arange(X_train.shape[0])
    rng = np.random.default_rng(config.SEED)
    batch = 8192

    def predict(matrix) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, matrix.shape[0], 65536):
                chunk = matrix[start:start + 65536].toarray().astype(np.float32)
                out.append(torch.softmax(model(torch.from_numpy(chunk)), dim=1).numpy())
        return np.concatenate(out)

    started = time.time()
    history, best = [], (np.inf, None, -1)
    path = config.RESULTS_DIR / "step11_curves_mlp.csv"
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch):
            index = np.sort(order[start:start + batch])
            chunk = torch.from_numpy(X_train[index].toarray().astype(np.float32))
            optimiser.zero_grad()
            loss = loss_fn(model(chunk), y_train[index])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
        probs = predict(val["X_linear"])
        val_loss = float(-np.log(np.clip(
            probs[np.arange(len(val["cls"])), val["cls"]], 1e-9, 1)).mean())
        history.append({"epoch": epoch, "train_loss": round(total / len(order), 5),
                        "val_mlogloss": round(val_loss, 5)})
        pd.DataFrame(history).to_csv(path, index=False, encoding="utf-8-sig")
        say(f"  epoch {epoch}: train {total / len(order):.5f}  val {val_loss:.5f}")
        if val_loss < best[0]:
            best = (val_loss, probs, epoch)
            torch.save(model.state_dict(), config.MODEL_DIR / "mlp.pt")

    return best[1], {"fit_seconds": round(time.time() - started, 1),
                     "best_epoch": best[2]}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, column, title, better in (
            (axes[0], "log_loss", "val log loss", "낮을수록 좋음"),
            (axes[1], "cdf_ece", "CDF 보정 오차", "낮을수록 좋음"),
            (axes[2], "auc_late", "AUC (15분+ 지연)", "높을수록 좋음")):
        bars = ax.bar(table["model"], table[column], color="#4c6ef5")
        for bar, v in zip(bars, table[column]):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.5f}",
                    ha="center", va="bottom", fontsize=9)
        low, high = table[column].min(), table[column].max()
        pad = max((high - low) * 0.45, high * 0.01)
        ax.set_ylim(max(0, low - pad), high + pad)
        ax.set_title(f"{title}\n({better})", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Step 11 — 같은 분할·같은 지표로 모델 3종 비교 (val 2025)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11_model_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    xgb_curve = config.RESULTS_DIR / "step11_curves_xgb.csv"
    if xgb_curve.exists():
        curve = pd.read_csv(xgb_curve)
        axes[0].plot(curve["round"], curve["train_mlogloss"], label="train")
        axes[0].plot(curve["round"], curve["val_mlogloss"], label="val")
        axes[0].set_xlabel("부스팅 라운드"); axes[0].set_ylabel("multi log loss")
        axes[0].set_title("XGBoost 학습 곡선"); axes[0].legend(fontsize=9)
    mlp_curve = config.RESULTS_DIR / "step11_curves_mlp.csv"
    if mlp_curve.exists():
        curve = pd.read_csv(mlp_curve)
        axes[1].plot(curve["epoch"], curve["train_loss"], "o-", label="train")
        axes[1].plot(curve["epoch"], curve["val_mlogloss"], "s-", label="val")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("cross entropy")
        axes[1].set_title("MLP 학습 곡선"); axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11_learning_curves.png")
    plt.close(fig)

    depth_path = config.RESULTS_DIR / "step11_xgb_depth.csv"
    if depth_path.exists():
        depth = pd.read_csv(depth_path)
        fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
        ax1.plot(depth["max_depth"], depth["train_mlogloss"], "o-", label="train")
        ax1.plot(depth["max_depth"], depth["val_mlogloss"], "s-", label="val")
        ax1.set_xlabel("max_depth"); ax1.set_ylabel("multi log loss")
        ax1.legend(loc="center left", fontsize=9)
        ax2 = ax1.twinx()
        ax2.bar(depth["max_depth"], depth["gap"], alpha=0.2, color="#c92a2a")
        ax2.set_ylabel("val − train (과적합 폭)", color="#c92a2a")
        ax2.grid(False)
        span = depth["val_mlogloss"].max() - depth["val_mlogloss"].min()
        ax1.set_title(f"트리 깊이와 과적합 — 깊이를 3→12로 바꿔도 val 폭은 {span:.4f}")
        fig.tight_layout()
        fig.savefig(config.FIGURE_DIR / "step11_depth_overfit.png")
        plt.close(fig)

    rel_path = config.RESULTS_DIR / "step11_reliability.csv"
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        for model in rel["model"].unique():
            part = rel[rel["model"] == model]
            ax.plot(part["actual"], part["predicted"], "o-", label=model)
        lim = [0, 1]
        ax.plot(lim, lim, "k--", lw=1, label="완벽한 보정")
        ax.set_xlabel("실제 비율"); ax.set_ylabel("모델이 말한 확률")
        ax.set_title("Step 11 — 보정 전 신뢰도 (8개 구간 경계, val 2025)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(config.FIGURE_DIR / "step11_reliability_before.png")
        plt.close(fig)


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--skip", type=str, default="")
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    drop = chosen_features()
    say("loading ...")
    train = common.load_split("train", args.limit, drop=drop)
    val = common.load_split("val", args.limit, drop=drop)
    say(f"  train {len(train['cls']):,} / val {len(val['cls']):,}  "
        f"linear {train['X_linear'].shape[1]} cols, "
        f"xgb {train['X_xgb'].shape[1]} cols")

    table_path = config.RESULTS_DIR / "step11_models.csv"
    rel_path = config.RESULTS_DIR / "step11_reliability.csv"
    if not {"logistic", "xgboost", "mlp"} <= skip:
        table_path.unlink(missing_ok=True)
        rel_path.unlink(missing_ok=True)

    np.save(config.MODEL_DIR / "val_cls.npy", val["cls"])
    np.save(config.MODEL_DIR / "val_delay.npy", val["delay"])

    def record(label: str, probs: np.ndarray, info: dict) -> None:
        scores = common.evaluate(probs, val["cls"], val["delay"])
        append(table_path, {"model": label, **common.flat(scores), **info})
        for edge, values in scores["reliability"].items():
            append(rel_path, {"model": label, "edge": edge,
                              "predicted": values["예측"],
                              "actual": values["실제"],
                              "gap": round(values["예측"] - values["실제"], 4)})
        say(f"  -> {label}: log loss {scores['log_loss']}  "
            f"CDF 오차 {scores['cdf_ece']}  AUC {scores['auc_late']}")

    if "xgboost" not in skip:
        say("M2 XGBoost ...")
        probs, info = run_xgboost(train, val, args.rounds)
        np.save(config.MODEL_DIR / "val_pred_xgboost.npy", probs.astype("float32"))
        record("M2 XGBoost", probs, info)

    if "logistic" not in skip:
        say("M1 다항 로지스틱 ...")
        probs, info = run_logistic(train, val, args.max_iter)
        np.save(config.MODEL_DIR / "val_pred_logistic.npy", probs.astype("float32"))
        record("M1 로지스틱", probs, info)

    if "mlp" not in skip:
        say("M3 MLP ...")
        probs, info = run_mlp(train, val, args.epochs)
        np.save(config.MODEL_DIR / "val_pred_mlp.npy", probs.astype("float32"))
        record("M3 MLP", probs, info)

    if "depth" not in skip:
        say("트리 깊이별 과적합 ...")
        depth_study(train, val)
    if "search" not in skip:
        say("하이퍼파라미터 탐색 ...")
        hyper_search(train, val)

    table = pd.read_csv(table_path)
    make_figures(table)

    print("\n" + "=" * 100)
    print(table.to_string(index=False))

    best = table.sort_values("log_loss").iloc[0]
    fastest = table.sort_values("fit_seconds").iloc[0]
    span = float(table["log_loss"].max() - table["log_loss"].min())
    print(f"\nlog loss 최저: {best['model']} ({best['log_loss']})")
    print(f"가장 빠름: {fastest['model']} ({fastest['fit_seconds']}초)")
    print(f"세 모델의 log loss 폭: {span:.5f}")
    print("→ 폭이 작으면 천장은 모델이 아니라 변수다. 더 튜닝할 곳이 아니라,")
    print("  예약 시점 정보가 원래 거기까지라는 뜻이다(14단계에서 확인).")

    (config.MODEL_DIR / "chosen.json").write_text(json.dumps(
        {"dropped_features": drop,
         "xgb_columns": list(train["X_xgb"].columns),
         "linear_columns": train["linear_columns"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
