"""step06c — step06b 결과로 순위가 바뀌었으면 XGBoost 를 채택 후보로 올린다.

읽기  results/step06b_xgb_long.csv, results/step06_best.csv
쓰기  (순위가 바뀐 경우에만)
      data/model/xgboost_best.json          ← 길게 돌린 모델로 교체
      data/model/xgboost_best_span.json
      data/model/val_probs_xgboost.npy
      results/step06_best.csv               xgboost 행만 갱신
      results/step06c_promotion.json        무엇을 왜 바꿨는지
      (교체 전 400라운드 파일은 *_round400 으로 백업한다)

무엇을 판단하는가
-----------------
step06b 가 라운드 상한을 올려 다시 돌린 XGBoost 의 **검증 log loss** 가
step06 에서 채택된 MLP 보다 낮은지 본다. 낮으면 **선택 규칙(검증 log loss 최저)**
에 따라 채택이 XGBoost 로 바뀐다.

규칙을 바꾸는 것이 아니다 — **같은 규칙에 더 큰 탐색 예산을 준 것**이다.
판단은 검증 구간으로만 한다. 평가 구간(2026)은 이 스텝에서 열지 않는다.

바꾸지 않는 것
--------------
step06 의 탐색 기록(`step06_search.csv`)과 step06b 의 결과는 그대로 남긴다.
400라운드 모델도 `*_round400` 으로 백업한다. 보고서에 "400라운드로 끊었을 때"와
"끝까지 돌렸을 때"를 나란히 적을 수 있어야 한다.

실행  python src/step06c_promote.py
      → 교체했으면 종료 코드 0, 순위가 그대로면 10 (이어지는 단계를 건너뛰라는 신호)
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

import config

MLP_VAL = 1.351091          # step06 에서 채택된 MLP 의 검증 log loss


def main() -> int:
    long_table = pd.read_csv(config.RESULTS_DIR / "step06b_xgb_long.csv",
                             encoding="utf-8-sig")
    row = long_table.loc[long_table["val_log_loss"].idxmin()]
    xgb_val = float(row["val_log_loss"])

    print(f"MLP 검증 log loss        {MLP_VAL:.6f}")
    print(f"XGBoost(길게) 검증        {xgb_val:.6f}  [{row['설정']}, "
          f"최적 {int(row['최적 라운드'])} 라운드]")

    decision = {
        "rule": "검증 구간 multiclass log loss 가 가장 낮은 계열",
        "mlp_val_log_loss": MLP_VAL,
        "xgboost_long_val_log_loss": xgb_val,
        "xgboost_long_setting": str(row["설정"]),
        "xgboost_long_best_round": int(row["최적 라운드"]),
        "xgboost_stopped_itself": bool(row["스스로 멈췄는가"]),
        "note": "평가 구간은 이 스텝에서 열지 않았다. 판단은 검증 구간으로만 했다.",
    }

    if xgb_val >= MLP_VAL:
        decision["promoted"] = False
        decision["conclusion"] = (
            f"순위가 그대로다. MLP 가 여전히 {xgb_val - MLP_VAL:.6f} 낮다. "
            "채택을 바꾸지 않았고 보정·평가도 다시 돌리지 않는다.")
        (config.RESULTS_DIR / "step06c_promotion.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {decision['conclusion']}")
        return 10

    # ----- 순위가 바뀐 경우: 400라운드 산출물을 백업하고 교체한다 -----
    pairs = [
        ("xgboost_best.json", "xgboost_round400.json"),
        ("xgboost_best_span.json", "xgboost_round400_span.json"),
        ("val_probs_xgboost.npy", "val_probs_xgboost_round400.npy"),
    ]
    for original, backup in pairs:
        src = config.MODEL_DIR / original
        if src.exists():
            shutil.copy2(src, config.MODEL_DIR / backup)
    shutil.copy2(config.RESULTS_DIR / "step06_best.csv",
                 config.RESULTS_DIR / "step06_best_round400.csv")
    print("\n400라운드 산출물을 *_round400 으로 백업했다.")

    shutil.copy2(config.MODEL_DIR / "xgboost_long.json",
                 config.MODEL_DIR / "xgboost_best.json")
    shutil.copy2(config.MODEL_DIR / "xgboost_long_span.json",
                 config.MODEL_DIR / "xgboost_best_span.json")
    shutil.copy2(config.MODEL_DIR / "val_probs_xgboost_long.npy",
                 config.MODEL_DIR / "val_probs_xgboost.npy")

    best = pd.read_csv(config.RESULTS_DIR / "step06_best.csv",
                       encoding="utf-8-sig")
    mask = best["family"] == "xgboost"
    label = f"{row['설정']} · {int(row['최적 라운드'])}라운드"
    best.loc[mask, "설정"] = label
    best.loc[mask, "train_log_loss"] = float(row["train_log_loss"])
    best.loc[mask, "val_log_loss"] = xgb_val
    best.loc[mask, "best_iteration"] = int(row["최적 라운드"])
    best.loc[mask, "fit_seconds"] = float(row["fit_seconds"])
    best.to_csv(config.RESULTS_DIR / "step06_best.csv",
                index=False, encoding="utf-8-sig")

    decision["promoted"] = True
    decision["new_label"] = label
    decision["conclusion"] = (
        f"순위가 바뀌었다. XGBoost 가 MLP 보다 {MLP_VAL - xgb_val:.6f} 낮다. "
        "선택 규칙에 따라 채택을 XGBoost 로 바꾸고, 보정(step08)과 "
        "평가(step09)를 다시 돌린다.")
    (config.RESULTS_DIR / "step06c_promotion.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {decision['conclusion']}")
    print(f"   채택 설정 표기: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
