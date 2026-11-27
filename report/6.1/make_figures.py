from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).parent
RES = OUT.parents[1] / "final" / "results"

best = pd.read_csv(RES / "step06_best.csv", encoding="utf-8-sig")
name = {"로지스틱": "로지스틱 회귀", "xgboost": "XGBoost", "MLP": "MLP"}
setting = {
    "로지스틱": "C = 0.1",
    "xgboost": "depth 6, eta 0.05",
    "MLP": "(512, 128, 32), lr 3e-4",
}
best["label"] = best["family"].map(name)
best["setting"] = best["family"].map(setting)
best = best.sort_values("val_log_loss").reset_index(drop=True)

CHOSEN = "#2a6fdb"
OTHER = "#b8c2cf"
colors = [CHOSEN if f == "MLP" else OTHER for f in best["family"]]
xt = [f"{l}\n{s}" for l, s in zip(best["label"], best["setting"])]

# 1) 검증 log loss, y축 확대
mlp = best["val_log_loss"].iloc[0]
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
bars = ax.bar(xt, best["val_log_loss"], color=CHOSEN, width=0.5, zorder=3)
ax.set_ylim(1.350, 1.375)
for b, v in zip(bars, best["val_log_loss"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0003, f"{v:.6f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Validation Log Loss")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig6_1_val_logloss.png", bbox_inches="tight")
plt.close(fig)

# 2) 학습 vs 검증 log loss, y축 확대
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
w = 0.36
x = range(len(best))
b1 = ax.bar([i - w / 2 for i in x], best["train_log_loss"], w, label="학습 (2022–2024)",
            color="#e3e8ef", edgecolor="#8894a6", zorder=3)
b2 = ax.bar([i + w / 2 for i in x], best["val_log_loss"], w, label="검증 (2025)",
            color=CHOSEN, zorder=3)
for b in list(b1) + list(b2):
    v = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0006, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(list(x), xt)
lo2, hi2 = 1.29, 1.38
ax.set_ylim(lo2, hi2)
ax.set_ylabel("Multiclass Log Loss (낮을수록 좋음)")
ax.set_title("6.1 학습 vs 검증 Log Loss", fontsize=13)
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.99, -0.2, f"※ y축을 {lo2}~{hi2} 로 확대함 (0에서 시작하지 않음)",
        transform=ax.transAxes, ha="right", fontsize=8.5, color="#555")
fig.tight_layout()
fig.savefig(OUT / "fig6_1_train_vs_val.png", bbox_inches="tight")
plt.close(fig)

# 3) 표 이미지
rows = [[r.label, r.setting, f"{r.train_log_loss:.6f}", f"{r.val_log_loss:.6f}",
         "—" if r.family == "MLP" else f"+{r.val_log_loss - mlp:.6f}",
         "채택" if r.family == "MLP" else ""] for r in best.itertuples()]
cols = ["모델", "최종 하이퍼파라미터", "학습 log loss", "검증 log loss", "MLP 대비 차이", "선택"]
fig, ax = plt.subplots(figsize=(10, 1.9), dpi=200)
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
    elif rows[r - 1][-1] == "채택":
        cell.set_facecolor("#e6efff")
        cell.set_text_props(fontweight="bold")
fig.savefig(OUT / "table6_1_model_comparison.png", bbox_inches="tight")
plt.close(fig)
