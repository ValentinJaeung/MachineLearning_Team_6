"""6.1 과적합 확인 그래프 3장 (그래프 안 설명 없음, 점선 = 최종 모델).
MLP: final/logs/06_mlp.log '넓게 (512, 128, 32)' 블록 (epoch 별)        -> 점선 epoch 7  (step06_best best_epoch)
XGBoost: final/results/step06b_xgb_curve.csv '기준 depth6 eta0.05'      -> 점선 400라운드 (step06_best best_iteration)
로지스틱: final/results/step06_search.csv C=0.1/1/10 (검증 곡선)          -> 점선 C=0.1   (step06_best C)
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
OUT = Path(__file__).resolve().parent
RES = OUT.parents[1] / "final" / "results"
LOGS = OUT.parents[1] / "final" / "logs"
TRAIN, VAL = "#8894a6", "#2a6fdb"
LT, LV = "학습 (2022–2024)", "검증 (2025)"


def finish(ax, path, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Log Loss")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / path, bbox_inches="tight")
    plt.close(ax.figure)


# MLP
log = (LOGS / "06_mlp.log").read_text(encoding="utf-8")
block = log.split("MLP [넓게 (512, 128, 32)] 적합")[1].split("MLP [")[0]
mlp = pd.DataFrame([(int(e), float(t), float(v)) for e, t, v in
                    re.findall(r"epoch (\d+): train ([\d.]+)\s+val ([\d.]+)", block)],
                   columns=["epoch", "train", "val"])
mlp.to_csv(OUT / "mlp_curve.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6, 4.4), dpi=200)
ax.plot(mlp["epoch"], mlp["train"], "-o", color=TRAIN, ms=4, label=LT)
ax.plot(mlp["epoch"], mlp["val"], "-o", color=VAL, ms=4, label=LV)
ax.axvline(7, color="black", ls="--", lw=1)
ax.set_xticks(mlp["epoch"])
ax.set_ylim(1.28, 1.48)
finish(ax, "fig6_1_curve_mlp.png", "Epoch")

# XGBoost
xgb = pd.read_csv(RES / "step06b_xgb_curve.csv", encoding="utf-8-sig")
xgb = xgb[xgb["설정"] == "기준 depth6 eta0.05"]
fig, ax = plt.subplots(figsize=(6, 4.4), dpi=200)
ax.plot(xgb["라운드"], xgb["train"], color=TRAIN, lw=1.4, label=LT)
ax.plot(xgb["라운드"], xgb["val"], color=VAL, lw=1.4, label=LV)
ax.axvline(400, color="black", ls="--", lw=1)
ax.set_ylim(1.28, 1.48)
finish(ax, "fig6_1_curve_xgboost.png", "Boosting Round")

# 로지스틱 (검증 곡선: C 별)
lg = pd.read_csv(RES / "step06_search.csv", encoding="utf-8-sig")
lg = lg[lg["family"] == "로지스틱"].sort_values("C")
fig, ax = plt.subplots(figsize=(6, 4.4), dpi=200)
ax.plot(lg["C"], lg["train_log_loss"], "-o", color=TRAIN, ms=4, label=LT)
ax.plot(lg["C"], lg["val_log_loss"], "-o", color=VAL, ms=4, label=LV)
ax.axvline(0.1, color="black", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xticks(lg["C"], [f"{c:g}" for c in lg["C"]])
ax.set_ylim(1.34, 1.39)
finish(ax, "fig6_1_curve_logistic.png", "C (규제 강도의 역수, log 축)")
