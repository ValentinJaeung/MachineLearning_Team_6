"""step06 — 세 모델을 학습하고 튜닝한다.

읽기  data/prep/
쓰기  data/model/{family}_best.*            계열별 가장 좋은 모델
      data/model/val_probs_{family}.npy     그 모델의 검증 구간 예측
      results/step06_search.csv             모든 적합의 기록
      results/step06_best.csv               계열별 채택 설정

탐색은 '한 번에 하나씩'
-----------------------
기준 설정을 하나 정하고 하이퍼파라미터를 **하나만** 바꿔 본다. 격자 전체를
돌리면 적합 횟수가 수십 배가 되는데, 어느 값이 효과가 있었는지는 하나씩
바꿔 봐도 알 수 있다.

  XGBoost   기준 max_depth=6, eta=0.05  →  깊이 3 / 깊이 10 / eta 0.10
  MLP       기준 (256,64), lr=3e-4      →  (512,128,32) / lr 1e-3
  로지스틱   규제 강도 C = 0.1 / 1 / 10

세 계열에 같은 조건을 준다
--------------------------
같은 학습 구간 전체, 같은 분할, 같은 지표(검증 log loss), 그리고 비슷한 횟수의
탐색. 드문 구간의 비중을 키우는 처리(클래스 가중치·리샘플링)는 **쓰지 않는다** —
모델이 내는 확률이 실제 비율에서 벗어나면 우리가 파는 것이 거짓이 된다.

과적합·과소적합은 학습 log loss 와 검증 log loss 를 나란히 적어 확인한다.

실행  python src/step06_train.py [--only xgboost|logistic|mlp]
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import common
import config


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def run_xgboost(y_train, y_val, delay_val) -> tuple[list[dict], dict]:
    X_train, X_val = common.load_tree("train"), common.load_tree("val")
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    del X_train, X_val

    base = {"max_depth": 6, "eta": 0.05}
    settings = [
        ("기준 depth6 eta0.05", base),
        ("depth 3", {**base, "max_depth": 3}),
        ("depth 10", {**base, "max_depth": 10}),
        ("eta 0.10", {**base, "eta": 0.10}),
    ]

    rows, best = [], None
    for name, override in settings:
        params = {**config.XGB_BASE, **override, "seed": config.SEED}
        say(f"XGBoost [{name}] 적합 ...")
        started = time.time()
        history: dict = {}
        booster = xgb.train(
            params, dtrain, num_boost_round=config.XGB_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=config.XGB_EARLY_STOPPING,
            evals_result=history, verbose_eval=False)
        span = (0, booster.best_iteration + 1)
        probs = booster.predict(dval, iteration_range=span).astype("float64")
        row = {
            "family": "xgboost", "설정": name,
            "max_depth": params["max_depth"], "eta": params["eta"],
            "train_log_loss": round(history["train"]["mlogloss"][booster.best_iteration], 6),
            "val_log_loss": round(common.log_loss(probs, y_val), 6),
            "best_iteration": int(booster.best_iteration + 1),
            "fit_seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        say(f"  train {row['train_log_loss']}  val {row['val_log_loss']}  "
            f"({row['best_iteration']} rounds, {row['fit_seconds']}s)")
        if best is None or row["val_log_loss"] < best["row"]["val_log_loss"]:
            best = {"row": row, "probs": probs, "model": booster, "span": span}

    best["model"].save_model(config.MODEL_DIR / "xgboost_best.json")
    (config.MODEL_DIR / "xgboost_best_span.json").write_text(
        json.dumps({"iteration_range": list(best["span"])}), encoding="utf-8")
    np.save(config.MODEL_DIR / "val_probs_xgboost.npy",
            best["probs"].astype("float32"))
    return rows, best["row"]


# --------------------------------------------------------------------------- #
# 다항 로지스틱 회귀
# --------------------------------------------------------------------------- #
def run_logistic(y_train, y_val, delay_val) -> tuple[list[dict], dict]:
    from sklearn.linear_model import LogisticRegression

    X_train, X_val = common.load_linear("train"), common.load_linear("val")

    rows, best = [], None
    for C in config.LOGISTIC_C:
        say(f"로지스틱 [C={C}] 적합 ...")
        started = time.time()
        model = LogisticRegression(
            C=C, solver="lbfgs", max_iter=config.LOGISTIC_MAX_ITER,
            random_state=config.SEED)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val).astype("float64")
        row = {
            "family": "로지스틱", "설정": f"C={C}", "C": C,
            "train_log_loss": round(
                common.log_loss(model.predict_proba(X_train).astype("float64"),
                                y_train), 6),
            "val_log_loss": round(common.log_loss(probs, y_val), 6),
            "iterations": int(np.max(model.n_iter_)),
            "converged": bool(np.max(model.n_iter_) < config.LOGISTIC_MAX_ITER),
            "fit_seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        say(f"  train {row['train_log_loss']}  val {row['val_log_loss']}  "
            f"({row['iterations']} iters, 수렴 {row['converged']}, "
            f"{row['fit_seconds']}s)")
        if best is None or row["val_log_loss"] < best["row"]["val_log_loss"]:
            best = {"row": row, "probs": probs, "model": model}

    joblib.dump(best["model"], config.MODEL_DIR / "logistic_best.joblib")
    np.save(config.MODEL_DIR / "val_probs_logistic.npy",
            best["probs"].astype("float32"))
    return rows, best["row"]


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
def run_mlp(y_train, y_val, delay_val) -> tuple[list[dict], dict]:
    import torch
    from torch import nn

    X_train, X_val = common.load_linear("train"), common.load_linear("val")
    n_features = X_train.shape[1]
    yt = torch.from_numpy(y_train)

    def batches(matrix, order, size):
        for start in range(0, len(order), size):
            idx = order[start:start + size]
            dense = torch.from_numpy(matrix[idx].toarray())
            yield idx, dense

    def predict(model, matrix) -> np.ndarray:
        model.eval()
        out = np.empty((matrix.shape[0], config.N_CLASS), dtype="float64")
        order = np.arange(matrix.shape[0])
        with torch.no_grad():
            for idx, dense in batches(matrix, order, config.MLP_BATCH * 2):
                out[idx] = torch.softmax(model(dense), dim=1).numpy()
        return out

    def build(hidden) -> nn.Module:
        layers, width = [], n_features
        for i, size in enumerate(hidden):
            layers += [nn.Linear(width, size), nn.ReLU()]
            if i < len(hidden) - 1:                 # 마지막 은닉층엔 드롭아웃 없음
                layers.append(nn.Dropout(config.MLP_DROPOUT))
            width = size
        layers.append(nn.Linear(width, config.N_CLASS))
        return nn.Sequential(*layers)

    base_hidden, base_lr = config.MLP_HIDDEN[0], config.MLP_LR[0]
    settings = [
        (f"기준 {base_hidden} lr{base_lr:g}", base_hidden, base_lr),
        (f"넓게 {config.MLP_HIDDEN[1]}", config.MLP_HIDDEN[1], base_lr),
        (f"lr {config.MLP_LR[1]:g}", base_hidden, config.MLP_LR[1]),
    ]

    rows, best = [], None
    for name, hidden, lr in settings:
        config.set_seeds()
        say(f"MLP [{name}] 적합 ...")
        started = time.time()
        model = build(hidden)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        generator = np.random.default_rng(config.SEED)

        best_epoch, best_val, best_state, best_train = -1, np.inf, None, np.nan
        for epoch in range(1, config.MLP_MAX_EPOCHS + 1):
            model.train()
            order = generator.permutation(X_train.shape[0])
            running, seen = 0.0, 0
            for idx, dense in batches(X_train, order, config.MLP_BATCH):
                optimiser.zero_grad()
                loss = loss_fn(model(dense), yt[idx])
                loss.backward()
                optimiser.step()
                running += loss.detach().item() * len(idx)
                seen += len(idx)
            val_loss = common.log_loss(predict(model, X_val), y_val)
            say(f"    epoch {epoch}: train {running / seen:.6f}  val {val_loss:.6f}")
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                best_train = running / seen
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        probs = predict(model, X_val)
        row = {
            "family": "MLP", "설정": name, "hidden": str(hidden), "lr": lr,
            "params": int(sum(p.numel() for p in model.parameters())),
            "train_log_loss": round(float(best_train), 6),
            "val_log_loss": round(common.log_loss(probs, y_val), 6),
            "best_epoch": best_epoch,
            "fit_seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        say(f"  train {row['train_log_loss']}  val {row['val_log_loss']}  "
            f"(epoch {best_epoch}, {row['fit_seconds']}s)")
        if best is None or row["val_log_loss"] < best["row"]["val_log_loss"]:
            best = {"row": row, "probs": probs,
                    "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "hidden": hidden}

    torch.save({"state_dict": best["state"], "hidden": list(best["hidden"]),
                "n_features": n_features}, config.MODEL_DIR / "mlp_best.pt")
    np.save(config.MODEL_DIR / "val_probs_mlp.npy",
            best["probs"].astype("float32"))
    return rows, best["row"]


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["xgboost", "logistic", "mlp"],
                        action="append")
    args = parser.parse_args()
    families = args.only or ["xgboost", "logistic", "mlp"]

    config.set_seeds()
    config.ensure_dirs()

    y_train, y_val = common.load_y("train"), common.load_y("val")
    delay_val = common.load_meta("val")["arr_delay_min"].to_numpy(dtype="float64")
    say(f"학습 {len(y_train):,} 편 / 검증 {len(y_val):,} 편")

    runner = {"xgboost": run_xgboost, "logistic": run_logistic, "mlp": run_mlp}
    all_rows, bests = [], []
    for family in families:
        rows, best = runner[family](y_train, y_val, delay_val)
        all_rows += rows
        bests.append(best)

    # 기존 기록과 합쳐서 저장한다(계열별로 따로 돌릴 수 있게)
    search_path = config.RESULTS_DIR / "step06_search.csv"
    table = pd.DataFrame(all_rows)
    if search_path.exists() and args.only:
        old = pd.read_csv(search_path)
        table = pd.concat([old[~old["family"].isin(table["family"])], table],
                          ignore_index=True)
    table.to_csv(search_path, index=False, encoding="utf-8-sig")

    best_path = config.RESULTS_DIR / "step06_best.csv"
    best_table = pd.DataFrame(bests)
    if best_path.exists() and args.only:
        old = pd.read_csv(best_path)
        best_table = pd.concat(
            [old[~old["family"].isin(best_table["family"])], best_table],
            ignore_index=True)
    best_table.to_csv(best_path, index=False, encoding="utf-8-sig")

    print("\n=== 탐색 기록 ===")
    print(table.to_string(index=False))
    print("\n=== 계열별 채택 ===")
    print(best_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
