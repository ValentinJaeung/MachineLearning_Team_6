"""6.2 보강: (1) 검증 2025 보정 전후 표, (2) 평가 2026 경계별 보정 전 평균 예측확률 vs 실제 비율.

step09_service.py 와 같은 함수(common.log_loss, cdf_calibration_error, promise_table)를
같은 보정기(calibrator.joblib)로 2025 에 적용한다.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
sys.path.insert(0, str(OUT.parents[1] / "final" / "src"))
import common  # noqa: E402
import config  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

cal = joblib.load(config.MODEL_DIR / "calibrator.joblib")


def load(split):
    y = common.load_y(split)
    delay = common.load_meta(split)["arr_delay_min"].to_numpy("float64")
    raw = common.predict_split("mlp", split)
    return y, delay, raw


# 1) 검증 2025 보정 전후
y, delay, raw = load("val")
rows = []
for stage, probs, curve in (("보정 전", raw, common.cdf(raw)),
                            ("보정 후", cal.transform_probs(raw), cal.transform(raw))):
    rows.append({"단계": stage,
                 "log_loss": round(common.log_loss(probs, y), 6),
                 "cdf_보정오차": round(common.cdf_calibration_error(curve, y, delay), 6),
                 **common.promise_table(curve, y, delay)})
pd.DataFrame(rows).to_csv(OUT / "val_calibration.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(rows).to_string())

# 2) 평가 2026 경계별 보정 전 평균 예측확률 vs 실제 비율
y, delay, raw = load("test")
curve = common.cdf(raw)
arrived = y < config.NOARRIVE
edges = pd.DataFrame({
    "경계(분)": config.EDGES,
    "보정 전 평균 확률": [curve[:, i].mean() for i in range(len(config.EDGES))],
    "실제 비율": [(arrived & (delay <= e)).mean() for e in config.EDGES],
})
edges["보정 전 오차"] = edges["보정 전 평균 확률"] - edges["실제 비율"]
edges = edges.round(6)
edges.to_csv(OUT / "test_edges_before.csv", index=False, encoding="utf-8-sig")
print(edges.to_string())
print("검산 CDF 보정오차 =", round(edges["보정 전 오차"].abs().mean(), 6))

fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
x = np.arange(len(edges))
w = 0.38
b1 = ax.bar(x - w / 2, edges["보정 전 평균 확률"], w, color="#2a6fdb", label="평균 예측확률 (보정 전)", zorder=3)
b2 = ax.bar(x + w / 2, edges["실제 비율"], w, color="#b8c2cf", label="실제 비율", zorder=3)
for b in list(b1) + list(b2):
    v = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x, edges["경계(분)"].astype(str))
ax.set_ylim(0.5, 1.0)
ax.set_xlabel("경계 (분)")
ax.set_ylabel("누적 확률 (해당 경계 안에 도착)")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig6_2_test_edges_before.png", bbox_inches="tight")
plt.close(fig)
