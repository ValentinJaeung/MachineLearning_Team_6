"""step08 — 선정된 모델의 확률을 보정한다.

읽기  results/step07_decision.json, data/model/val_probs_{family}.npy
쓰기  data/model/calibrator.joblib        ★ step09 가 테스트에 적용한다
      results/step08_reliability.csv      경계별 '말한 확률 vs 실제 비율'
      figures/step08_reliability.png

무엇을 고치는가
---------------
모델이 "80% 안에 도착한다" 고 말한 편들을 모아 보면, 실제로 그 안에 도착한
비율이 80% 가 아닐 수 있다. 분포의 모양이 아니라 **눈금이 틀린 것**이다.
모델을 버리지 않고 **읽는 값만** 고친다 — 온도계를 바꾸는 대신 환산표를 만든다.

어떻게
------
경계 8개(0·15·30·60·90·120·180·300분) 각각에서 "모델이 말한 누적확률 → 실제로
그 안에 도착한 비율" 의 대응을 **isotonic regression**(단조증가 함수)으로
학습한다. 단조증가로 제한하는 이유는 "모델이 더 높은 확률을 준 편이 실제로도
더 자주 그 안에 도착한다" 는 순서만은 지켜야 하기 때문이다.

어디서 적합하는가 — 이것이 중요하다
-----------------------------------
**검증 구간(2025)에서 적합하고 평가 구간(2026)에 적용한다.** 평가 구간에서
적합하면 정답을 보고 확률을 고친 것이 되어 누출이다. 검증 구간은 모델이 학습
하지 않은 해이므로, 거기서 만든 환산표는 "본 적 없는 해에서 눈금을 맞춘 것"
이 된다.

보정은 성능 개선이 아니다
-------------------------
정직성 작업이다. 보정 후 log loss 는 오히려 나빠질 수 있고, 보장 시각은 대체로
**뒤로 밀린다.** 그 대가로 "80% 라고 말한 것이 정말 80%" 가 된다. 이 절의
결과를 "성능이 좋아졌다" 로 쓰면 안 된다.

이 단계의 표는 적합 구간(검증)에서 잰 것이므로 진단용이다. 보정의 효과는
step09 가 평가 구간에서 보정 전후로 재서 보고한다.

실행  python src/step08_calibrate.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    decision = common.winner()
    family, label = decision["chosen_family"], decision["chosen_label"]
    print(f"채택된 모델: {label} ({decision['chosen_setting']})")

    y_val = common.load_y("val")
    delay_val = common.load_meta("val")["arr_delay_min"].to_numpy(dtype="float64")
    probs = np.load(config.MODEL_DIR / f"val_probs_{family}.npy").astype("float64")

    calibrator = common.CdfCalibrator().fit(probs, y_val, delay_val)
    joblib.dump(calibrator, config.MODEL_DIR / "calibrator.joblib")
    print(f"보정기를 검증 2025 에서 적합 → "
          f"{config.MODEL_DIR / 'calibrator.joblib'}")

    before = common.cdf(probs)
    after = calibrator.transform(probs)
    arrived = y_val < config.NOARRIVE

    rows = []
    for i, edge in enumerate(config.EDGES):
        actual = float((arrived & (delay_val <= edge)).mean())
        rows.append({
            "경계(분)": edge,
            "보정 전 평균 확률": round(float(before[:, i].mean()), 6),
            "보정 후 평균 확률": round(float(after[:, i].mean()), 6),
            "실제 비율": round(actual, 6),
            "보정 전 오차": round(float(before[:, i].mean()) - actual, 6),
            "보정 후 오차": round(float(after[:, i].mean()) - actual, 6),
        })
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step08_reliability.csv",
                 index=False, encoding="utf-8-sig")

    print("\n=== 경계별 눈금 (검증 2025 = 적합 구간이므로 진단용) ===")
    print(table.to_string(index=False))
    print(f"\nCDF 보정오차  전 "
          f"{common.cdf_calibration_error(before, y_val, delay_val):.6f}"
          f"  →  후 "
          f"{common.cdf_calibration_error(after, y_val, delay_val):.6f}")

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="완벽한 눈금")
    ax.plot(table["보정 전 평균 확률"], table["실제 비율"], "o-",
            color="#adb5bd", label="보정 전")
    ax.plot(table["보정 후 평균 확률"], table["실제 비율"], "s-",
            color="#2b8a3e", label="보정 후")
    for _, row in table.iterrows():
        ax.annotate(f"{int(row['경계(분)'])}분",
                    (row["보정 후 평균 확률"], row["실제 비율"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("모델이 말한 평균 확률")
    ax.set_ylabel("실제로 그 안에 도착한 비율")
    ax.set_title("step08 — 눈금이 맞는가 (검증 2025, 적합 구간)")
    ax.legend()
    fig.savefig(config.FIGURE_DIR / "step08_reliability.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
