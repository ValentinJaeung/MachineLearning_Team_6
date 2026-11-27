"""6.4 예시 편의 실제 조합(공역혼잡일×정상일) 보정 누적확률. 출처: example_true_curve.csv"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent

c = pd.read_csv(OUT / "example_true_curve.csv", encoding="utf-8-sig")
p = c["보정 누적확률"]
first = int((p >= 0.80).idxmax())                     # 0.80 에 처음 닿는 경계 = 보장 지연
colors = ["#2a6fdb" if i == first else "#b8c2cf" for i in range(len(c))]

fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
bars = ax.bar(c["경계(분)"].astype(str), p, color=colors, width=0.6, zorder=3)
ax.axhline(0.80, color="#d62728", ls="--", lw=1, zorder=4)
ax.text(-0.45, 0.803, "0.80", color="#d62728", ha="left", va="bottom", fontsize=9)
for b, v in zip(bars, p):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
            ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0.5, 1.0)
ax.set_xlabel("경계 (분)")
ax.set_ylabel("누적확률 (해당 경계 안에 도착)")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig6_4_example_cdf.png", bbox_inches="tight")
plt.close(fig)
