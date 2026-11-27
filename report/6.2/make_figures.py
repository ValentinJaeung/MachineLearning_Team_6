from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent
RES = OUT.parents[1] / "final" / "results"

d = pd.read_csv(RES / "step09_calibration_test.csv", encoding="utf-8-sig").set_index("단계")
b, a = d.loc["보정 전"], d.loc["보정 후"]

spec = [
    ("Log Loss", "log_loss", "{:.6f}", "낮을수록 좋음"),
    ("CDF 보정오차", "cdf_보정오차", "{:.6f}", "낮을수록 좋음"),
    ("약속 이행률", "이행률", "{:.4f}", "0.80 이상"),
    ("평균 보장 지연 (분)", "평균보장지연", "{:.2f}", "작을수록 좋음"),
    ("보장 불가 비율", "보장불가비율", "{:.4f}", "작을수록 좋음"),
]
rows = []
for name, col, fmt, goal in spec:
    diff = a[col] - b[col]
    rows.append([name, fmt.format(b[col]), fmt.format(a[col]),
                 ("+" if diff >= 0 else "-") + fmt.format(abs(diff)), goal])
cols = ["지표", "보정 전", "보정 후", "변화 (후 - 전)", "기준"]

fig, ax = plt.subplots(figsize=(9, 2.6), dpi=200)
ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(10.5)
t.scale(1, 1.8)
t.auto_set_column_width(list(range(len(cols))))
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#c9d1dc")
    if r == 0:
        cell.set_facecolor("#2f3b4c")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 2:
        cell.set_facecolor("#e6efff")
fig.savefig(OUT / "table6_2_calibration.png", bbox_inches="tight")
plt.close(fig)

# 검증 2025, 보정 전 경계별 오차 (평균 예측확률 - 실제 비율)
r = pd.read_csv(RES / "step08_reliability.csv", encoding="utf-8-sig")
err = r["보정 전 오차"]
fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
bars = ax.bar(r["경계(분)"].astype(str), err, color="#2a6fdb", width=0.6, zorder=3)
ax.axhline(0, color="black", lw=0.8, zorder=4)
for b, v in zip(bars, err):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.0002 if v >= 0 else -0.0002),
            f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.set_ylim(-0.003, 0.0105)
ax.set_xlabel("경계 (분)")
ax.set_ylabel("평균 예측확률 - 실제 비율")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig6_2_calibration_error_by_edge.png", bbox_inches="tight")
plt.close(fig)

# 검증 2025, 보정 후 경계별 오차 (같은 형식·같은 y축으로 보정 전 그림과 나란히 비교)
err = r["보정 후 오차"]
fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
bars = ax.bar(r["경계(분)"].astype(str), err, color="#2a6fdb", width=0.6, zorder=3)
ax.axhline(0, color="black", lw=0.8, zorder=4)
for b, v in zip(bars, err):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0002,
            f"{v:+.4f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(-0.003, 0.0105)
ax.set_xlabel("경계 (분)")
ax.set_ylabel("평균 예측확률 - 실제 비율")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig6_2_calibration_error_by_edge_after.png", bbox_inches="tight")
plt.close(fig)

# 검증 2025, 보정 전후 경계별 오차를 한 그림에
import numpy as np
x = np.arange(len(r))
w = 0.38
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
b1 = ax.bar(x - w / 2, r["보정 전 오차"], w, color="#b8c2cf", label="보정 전", zorder=3)
b2 = ax.bar(x + w / 2, r["보정 후 오차"], w, color="#2a6fdb", label="보정 후", zorder=3)
ax.axhline(0, color="black", lw=0.8, zorder=4)
for b, v in zip(b1, r["보정 전 오차"]):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.0002 if v >= 0 else -0.0002),
            f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
for b, v in zip(b2, r["보정 후 오차"]):
    ax.text(b.get_x() + b.get_width() / 2, 0.0002, f"{abs(v):.0f}",
            ha="center", va="bottom", fontsize=8, color="#2a6fdb")
ax.set_xticks(x, r["경계(분)"].astype(str))
ax.set_ylim(-0.003, 0.0105)
ax.set_xlabel("경계 (분)")
ax.set_ylabel("평균 예측확률 - 실제 비율")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig6_2_calibration_error_by_edge_before_after.png", bbox_inches="tight")
plt.close(fig)
