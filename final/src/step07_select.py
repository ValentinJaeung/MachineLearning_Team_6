"""step07 — 선언한 규칙 하나로 모델을 고른다.

읽기  data/model/val_probs_{family}.npy, results/step06_best.csv
쓰기  results/step07_selection.csv     계열별 지표
      results/step07_decision.json     ★ 채택된 계열 (step08·09 가 읽는다)
      figures/step07_selection.png

규칙 (숫자를 보기 전에 정했다)
------------------------------
**검증 구간의 multiclass log loss 가 가장 낮은 계열을 고른다.** 끝이다.

왜 log loss 하나인가: 세 계열 모두 이 값을 낮추도록 학습된다(lbfgs 의 수렴
조건, XGBoost 의 조기 종료, 신경망의 에폭 선택이 전부 log loss 를 읽는다).
적합 절차가 최적화하지 않는 다른 양으로 고르면, 각 계열을 그 양에 맞춰
튜닝하지 않은 상태로 그 양으로 줄을 세우는 셈이 된다.

CDF 보정오차는 **참고로만 적는다.** 확률의 정직성은 step08 의 보정이 담당하고,
그것이 실제로 지켜졌는지는 step09 의 이행률이 재므로 선택 단계에서 또 쓰지
않는다. 선택 지표를 둘로 두면 "어느 쪽으로 읽을지" 를 사후에 고를 수 있게 된다.

실행  python src/step07_select.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

FAMILIES = {"xgboost": "XGBoost", "logistic": "로지스틱", "mlp": "MLP"}


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    y_val = common.load_y("val")
    delay_val = common.load_meta("val")["arr_delay_min"].to_numpy(dtype="float64")

    best = pd.read_csv(config.RESULTS_DIR / "step06_best.csv")
    label_to_key = {FAMILIES[k]: k for k in FAMILIES}
    chosen_setting = {}
    for _, row in best.iterrows():
        key = label_to_key.get(row["family"], row["family"])
        chosen_setting[key] = row["설정"]

    rows = []
    for key, label in FAMILIES.items():
        path = config.MODEL_DIR / f"val_probs_{key}.npy"
        if not path.exists():
            print(f"건너뜀 (예측 없음): {label}")
            continue
        probs = np.load(path).astype("float64")
        curve = common.cdf(probs)
        rows.append({
            "family": label,
            "채택 설정": chosen_setting.get(key, ""),
            "val_log_loss": round(common.log_loss(probs, y_val), 6),
            "cdf_보정오차(참고)": round(
                common.cdf_calibration_error(curve, y_val, delay_val), 6),
            **{f"{k}(참고)": v for k, v in
               common.promise_table(curve, y_val, delay_val).items()
               if k in ("이행률", "평균보장지연", "보장불가비율")},
        })

    table = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    table["채택"] = [i == 0 for i in range(len(table))]
    table.to_csv(config.RESULTS_DIR / "step07_selection.csv",
                 index=False, encoding="utf-8-sig")

    winner_label = table.loc[0, "family"]
    winner_key = label_to_key[winner_label]
    gap = (float(table.loc[1, "val_log_loss"] - table.loc[0, "val_log_loss"])
           if len(table) > 1 else float("nan"))

    (config.RESULTS_DIR / "step07_decision.json").write_text(json.dumps({
        "rule": "검증 구간 multiclass log loss 가 가장 낮은 계열",
        "chosen_family": winner_key,
        "chosen_label": winner_label,
        "chosen_setting": chosen_setting.get(winner_key, ""),
        "val_log_loss": float(table.loc[0, "val_log_loss"]),
        "runner_up": str(table.loc[1, "family"]) if len(table) > 1 else None,
        "gap_to_runner_up": round(gap, 6),
        "note": "CDF 보정오차는 참고용이며 선택에 쓰지 않았다.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 계열별 비교 (검증 2025) ===")
    print(table.to_string(index=False))
    print(f"\n채택: {winner_label} ({chosen_setting.get(winner_key, '')}) — "
          f"2위와의 차이 {gap:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, column, title in (
            (axes[0], "val_log_loss", "검증 log loss (낮을수록 좋음) ← 선택 지표"),
            (axes[1], "cdf_보정오차(참고)", "CDF 보정오차 (참고용)")):
        colours = ["#2b8a3e" if c else "#adb5bd" for c in table["채택"]]
        ax.bar(table["family"], table[column], color=colours)
        for i, v in enumerate(table[column]):
            ax.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(table[column]) * 1.18)
    fig.suptitle("step07 — 모델 선택", fontsize=11)
    fig.savefig(config.FIGURE_DIR / "step07_selection.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
