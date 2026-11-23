"""step06b — XGBoost 를 조기 종료가 실제로 걸릴 때까지 돌린다.

읽기  data/prep/{train,val}_tree.parquet
쓰기  results/step06b_xgb_long.csv        라운드 상한을 올린 결과
      results/step06b_xgb_curve.csv       라운드별 train/val log loss
      figures/step06b_xgb_long.png        학습 곡선
      data/model/xgboost_long.json        모델
      data/model/val_probs_xgboost_long.npy

왜 이 스텝이 있는가
-------------------
step06 에서 XGBoost 두 설정(기준 depth6 eta0.05, depth 3)이 **400라운드를 다 쓰고
끝났다.** 조기 종료(40라운드 인내)가 걸리지 않았다는 뜻이고, 그래서 "더 돌리면
좋아질 여지가 있었다" 는 말을 한계로 적어야 했다.

이 스텝은 그 한계를 **측정한다.** 라운드 상한만 3000으로 올리고 나머지는 전부
step06 과 같게 둔다 — 같은 설정, 같은 시드, 같은 조기 종료 인내(40).
조기 종료가 실제로 걸리는 지점이 어디인지, 그때 검증 log loss 가 MLP(1.351091)를
넘는지 확인한다.

바뀐 것은 하나뿐이다
--------------------
    num_boost_round : 400  →  3000

조기 종료 인내는 40 그대로다. 그래서 "조기 종료 없이" 가 아니라 **"조기 종료가
스스로 걸릴 때까지"** 돌리는 것이다. 상한을 없애면 영원히 돌 수 있으므로
3000으로 둔다 — 그 전에 멈추면 상한이 아무 역할도 하지 않았다는 뜻이다.

선택 규칙은 그대로다
--------------------
**검증 구간 log loss 가 가장 낮은 계열.** 평가 구간(2026)은 여기서 열지 않는다.
이 결과로 순위가 바뀌면 그 사실을 보고하고, 채택을 바꿀지는 사람이 정한다.

step06 의 결과 파일을 덮어쓰지 않는다. 원래 실험을 그대로 남겨 두고 비교한다.

실행  python src/step06b_xgb_long.py [--rounds 3000] [--only depth6|depth3]
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

MLP_VAL = 1.351091          # step06 에서 채택된 MLP 의 검증 log loss
STEP06_VAL = {              # step06 에서 400라운드로 끝난 두 설정
    "기준 depth6 eta0.05": 1.351794,
    "depth 3": 1.360516,
}

SETTINGS = {
    "depth6": ("기준 depth6 eta0.05", {"max_depth": 6, "eta": 0.05}),
    "depth3": ("depth 3", {"max_depth": 3, "eta": 0.05}),
}


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--only", choices=list(SETTINGS), nargs="*")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    keys = args.only or list(SETTINGS)

    y_train, y_val = common.load_y("train"), common.load_y("val")
    say(f"학습 {len(y_train):,} 편 / 검증 {len(y_val):,} 편")
    dtrain = xgb.DMatrix(common.load_tree("train"), label=y_train,
                         enable_categorical=True)
    dval = xgb.DMatrix(common.load_tree("val"), label=y_val,
                       enable_categorical=True)

    rows, curves, best = [], [], None
    for key in keys:
        name, override = SETTINGS[key]
        params = {**config.XGB_BASE, **override, "seed": config.SEED}
        say(f"XGBoost [{name}] — 최대 {args.rounds} 라운드, 인내 "
            f"{config.XGB_EARLY_STOPPING} ...")
        started = time.time()
        history: dict = {}
        booster = xgb.train(
            params, dtrain, num_boost_round=args.rounds,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=config.XGB_EARLY_STOPPING,
            evals_result=history, verbose_eval=100)

        span = (0, booster.best_iteration + 1)
        probs = booster.predict(dval, iteration_range=span).astype("float64")
        used = len(history["val"]["mlogloss"])
        stopped_itself = used < args.rounds

        row = {
            "설정": name,
            "라운드 상한": args.rounds,
            "실제 돌린 라운드": used,
            "최적 라운드": int(booster.best_iteration + 1),
            "스스로 멈췄는가": stopped_itself,
            "train_log_loss": round(
                history["train"]["mlogloss"][booster.best_iteration], 6),
            "val_log_loss": round(common.log_loss(probs, y_val), 6),
            "step06_val_log_loss(400라운드)": STEP06_VAL[name],
            "MLP 대비": round(common.log_loss(probs, y_val) - MLP_VAL, 6),
            "fit_seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        for i, (tr, va) in enumerate(zip(history["train"]["mlogloss"],
                                         history["val"]["mlogloss"]), start=1):
            curves.append({"설정": name, "라운드": i,
                           "train": round(tr, 6), "val": round(va, 6)})

        say(f"  최적 {row['최적 라운드']} 라운드 / 돌린 {used} 라운드"
            f"{' (스스로 멈춤)' if stopped_itself else ' (상한에서 끊김)'}")
        say(f"  train {row['train_log_loss']}  val {row['val_log_loss']}"
            f"  — 400라운드 때 {STEP06_VAL[name]}"
            f" / MLP 대비 {row['MLP 대비']:+.6f}")

        if best is None or row["val_log_loss"] < best["row"]["val_log_loss"]:
            best = {"row": row, "probs": probs, "model": booster, "span": span}

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step06b_xgb_long.csv",
                 index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(config.RESULTS_DIR / "step06b_xgb_curve.csv",
                                index=False, encoding="utf-8-sig")

    best["model"].save_model(config.MODEL_DIR / "xgboost_long.json")
    (config.MODEL_DIR / "xgboost_long_span.json").write_text(
        json.dumps({"iteration_range": list(best["span"])}), encoding="utf-8")
    np.save(config.MODEL_DIR / "val_probs_xgboost_long.npy",
            best["probs"].astype("float32"))

    print("\n=== 라운드 상한을 올린 결과 (검증 2025) ===")
    print(table.to_string(index=False))

    winner = best["row"]
    print(f"\nMLP 검증 log loss      {MLP_VAL:.6f}")
    print(f"XGBoost(400라운드)     {STEP06_VAL[winner['설정']]:.6f}")
    print(f"XGBoost(길게)          {winner['val_log_loss']:.6f}")
    if winner["val_log_loss"] < MLP_VAL:
        print(f"\n→ 순위가 바뀐다. XGBoost 가 MLP 보다 "
              f"{MLP_VAL - winner['val_log_loss']:.6f} 낮다.")
    else:
        print(f"\n→ 순위는 그대로다. MLP 가 여전히 "
              f"{winner['val_log_loss'] - MLP_VAL:.6f} 낮다.")

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 4.4),
                             squeeze=False)
    curve_frame = pd.DataFrame(curves)
    for ax, row in zip(axes[0], rows):
        part = curve_frame[curve_frame["설정"] == row["설정"]]
        ax.plot(part["라운드"], part["train"], color="#adb5bd", label="학습")
        ax.plot(part["라운드"], part["val"], color="#1c7ed6", label="검증")
        ax.axhline(MLP_VAL, color="#2b8a3e", ls="--", lw=1,
                   label=f"MLP {MLP_VAL:.4f}")
        ax.axvline(400, color="#e03131", ls=":", lw=1, label="step06 상한 400")
        ax.axvline(row["최적 라운드"], color="#f08c00", lw=1,
                   label=f"최적 {row['최적 라운드']}")
        ax.set_xlabel("라운드")
        ax.set_ylabel("multiclass log loss")
        ax.set_title(row["설정"], fontsize=10)
        ax.set_ylim(min(part["train"].min(), MLP_VAL) - 0.004,
                    part["val"].iloc[min(60, len(part) - 1)] + 0.004)
        ax.legend(fontsize=8)
    fig.suptitle("step06b — 조기 종료가 걸릴 때까지 돌린 XGBoost", fontsize=11)
    fig.savefig(config.FIGURE_DIR / "step06b_xgb_long.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
