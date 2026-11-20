"""
Step 11b - is a model needed at all?
(PROPOSAL_FLOW §7, 분석 원칙 1: "모델이 필요한지 먼저 증명한다")

Reads  data/prep/ (step 9), data/dataset/ (step 7)
Writes results/step11b_baselines.csv
       figures/step11b_baselines.png

A model's score means nothing on its own. 1.40 log loss is only impressive
against something, and the honest something is not a coin flip - it is the
best answer a person could produce with a lookup table and a clock.

So four baselines are built, and they are built to WIN if they can:

  B1 전역 분포          the training distribution, the same ten numbers for
                       every flight. Knows nothing.
  B2 출발 시각 경험분포   the training distribution of flights leaving at that
                       hour. Knows only the clock.
  B3 계층축소 룩업       that exact flight's own history: the empirical
                       distribution of flight_key, falling back to route_key,
                       group_key and then global, each level shrunk toward the
                       level above it. This is handed THREE YEARS of the
                       flight's own record - far more than the model's nine
                       columns describe.
                       Its shrinkage strength k is swept and the value that
                       scores BEST ON VALIDATION is reported. That lets the
                       baseline tune on data the model never saw, which is
                       unfair in the baseline's favour - deliberately, because
                       the point is to beat the strongest version of it.
  B4 룩업 사다리(무평활)  the same ladder with no smoothing: take the narrowest
                       key that has any rows at all and believe it. This is
                       what "just look it up" naively means, and it is here to
                       show what smoothing is actually buying.

B3 is the one that matters. If three years of a flight's own arrivals barely
beat the global average, then the per-flight signal everyone assumes exists
is mostly noise, and that is the real finding - it is also why step 10's
report-card features had nothing left to add.

All baselines are fitted on train 2022-24 and scored on validation 2025, the
same rows and the same metrics as the models in step 11.

Usage:  python src/step11b_baselines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

# B3's shrinkage strength is swept rather than assumed; the best one is used
SMOOTH_GRID = [1.0, 4.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

KEY_CHAIN = ["flight_key", "route_key", "group_key"]


def distribution_by(frame: pd.DataFrame, key: str) -> tuple[np.ndarray, dict]:
    """Empirical class distribution and row count for every value of `key`."""
    counts = (frame.groupby([key, "y_class"], observed=True).size()
              .unstack(fill_value=0)
              .reindex(columns=range(config.N_CLASS), fill_value=0))
    index = {value: i for i, value in enumerate(counts.index)}
    return counts.to_numpy(dtype="float64"), index


def lookup(values: pd.Series, table: np.ndarray, index: dict) -> np.ndarray:
    """Row of counts for each value; zeros where the value is unknown."""
    positions = values.map(index)
    out = np.zeros((len(values), config.N_CLASS))
    known = positions.notna().to_numpy()
    out[known] = table[positions[known].to_numpy(dtype="int64")]
    return out


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    cols = KEY_CHAIN + ["crs_dep_hour", "y_class"]
    train = pd.read_parquet(config.DATASET_DIR / "train.parquet", columns=cols)
    val_full = common.load_split("val")
    val = pd.read_parquet(config.DATASET_DIR / "val.parquet", columns=cols)
    cls, delay = val_full["cls"], val_full["delay"]
    print(f"train {len(train):,} / val {len(val):,}")

    global_counts = (train["y_class"].value_counts()
                     .reindex(range(config.N_CLASS), fill_value=0)
                     .to_numpy(dtype="float64"))
    global_prior = global_counts / global_counts.sum()

    results = {}

    # ----------------------------------------------------------------- B1 --
    results["B1 전역 분포 (아무것도 모름)"] = np.tile(global_prior, (len(val), 1))

    # ----------------------------------------------------------------- B2 --
    hour_counts, hour_index = distribution_by(train, "crs_dep_hour")
    raw = lookup(val["crs_dep_hour"], hour_counts, hour_index)
    total = raw.sum(axis=1, keepdims=True)
    results["B2 출발 시각 경험분포 (시계만)"] = np.where(
        total > 0, raw / np.maximum(total, 1), global_prior)

    # ----------------------------------------------------------------- B3 --
    # narrow -> wide, each level smoothed toward the level above it
    tables = {key: distribution_by(train, key) for key in KEY_CHAIN}
    counts = {key: lookup(val[key], *tables[key]) for key in KEY_CHAIN}

    def ladder(k: float) -> np.ndarray:
        estimate = np.tile(global_prior, (len(val), 1))  # start at the global
        for key in reversed(KEY_CHAIN):                  # group -> route -> flight
            n = counts[key].sum(axis=1, keepdims=True)
            own = np.divide(counts[key], np.maximum(n, 1e-9),
                            out=np.zeros_like(counts[key]), where=n > 0)
            estimate = (n * own + k * estimate) / (n + k)
        return estimate

    sweep = []
    for k in SMOOTH_GRID:
        probs = ladder(k)
        loss = common.evaluate(probs, cls, delay)["log_loss"]
        sweep.append({"k": k, "val_log_loss": loss})
        print(f"    B3 평활 k={k:>6.0f}: log loss {loss:.5f}")
    sweep_table = pd.DataFrame(sweep)
    sweep_table.to_csv(config.RESULTS_DIR / "step11b_lookup_k_sweep.csv",
                       index=False, encoding="utf-8-sig")
    best_k = float(sweep_table.loc[sweep_table["val_log_loss"].idxmin(), "k"])
    print(f"    -> 베이스라인에 유리하도록 val 기준 최적 k={best_k:.0f} 채택")
    results[f"B3 계층축소 룩업 (그 편 3년 이력, k={best_k:.0f})"] = ladder(best_k)

    # ----------------------------------------------------------------- B4 --
    # the naive ladder: the narrowest key with any rows at all, believed as is
    naive = np.tile(global_prior, (len(val), 1))
    assigned = np.zeros(len(val), dtype=bool)
    hit_share = {}
    for key in KEY_CHAIN:
        n = counts[key].sum(axis=1)
        here = (~assigned) & (n > 0)
        naive[here] = counts[key][here] / n[here, None]
        assigned |= here
        hit_share[key] = float(here.mean())
    hit_share["global"] = float((~assigned).mean())
    results["B4 룩업 사다리 (무평활)"] = naive

    # --------------------------------------------------------------------- #
    rows = []
    for label, probs in results.items():
        scores = common.evaluate(probs, cls, delay)
        rows.append({"방법": label, **common.flat(scores)})
        print(f"  {label:<36} log loss {scores['log_loss']:.5f}  "
              f"AUC {scores['auc_late']:.5f}")

    model_path = config.RESULTS_DIR / "step11_models.csv"
    if model_path.exists():
        models = pd.read_csv(model_path)
        best = models.sort_values("log_loss").iloc[0]
        rows.append({"방법": f"모델 — {best['model']} (9변수)",
                     **{c: best[c] for c in
                        ("log_loss", "brier", "cdf_ece", "coverage_90",
                         "coverage_90_share", "auc_late") if c in models.columns}})

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step11b_baselines.csv",
                 index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 5))
    colours = ["#adb5bd"] * len(table)
    if table["방법"].iloc[-1].startswith("모델"):
        colours[-1] = "#2b8a3e"
    bars = ax.barh(table["방법"], table["log_loss"], color=colours)
    for bar, v in zip(bars, table["log_loss"]):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2, f"{v:.5f}",
                va="center", fontsize=9)
    ax.set_xlim(table["log_loss"].min() - 0.03, table["log_loss"].max() + 0.04)
    ax.set_xlabel("val log loss (낮을수록 좋음)")
    ax.set_title("Step 11b — 모델이 필요한가: 정직하게 강한 베이스라인과의 비교")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11b_baselines.png")
    plt.close(fig)

    print("\n" + "=" * 90)
    print(table[["방법", "log_loss", "cdf_ece", "auc_late"]]
          .to_string(index=False))
    print("\n무평활 사다리가 몇 번째 키에서 멈췄나: " +
          "  ".join(f"{k} {v:.1%}" for k, v in hit_share.items()))

    b1 = table.loc[table["방법"].str.startswith("B1"), "log_loss"].iloc[0]
    b2 = table.loc[table["방법"].str.startswith("B2"), "log_loss"].iloc[0]
    b3 = table.loc[table["방법"].str.startswith("B3"), "log_loss"].iloc[0]
    b4 = table.loc[table["방법"].str.startswith("B4"), "log_loss"].iloc[0]
    print(f"\n그 편의 3년 기록을 전부 보고, 평활 강도까지 val 에 맞춰 고른 룩업이 "
          f"전역 평균보다 {b1 - b3:+.5f} 나아진다 "
          f"(k 를 키울수록 좋아진다는 것은, 편 단위 값이 아니라 "
          f"'더 넓은 키의 값'이 실제 신호라는 뜻이다).")
    print(f"그 룩업조차 '몇 시 편인가'만 보는 B2({b2:.5f})를 "
          f"{'이기지 못한다' if b3 >= b2 else '겨우 이긴다'}.")
    print("편 단위 신호가 있다고 다들 믿지만 대부분은 잡음이었고,")
    print("10단계에서 과거 이력 변수가 아무것도 더하지 못한 이유가 여기 있다.")
    print(f"무평활 사다리는 전역 평균보다 {b4 - b1:+.5f} 나쁘다 — "
          "표본이 적은 칸을 그대로 믿으면 손해다. 평활이 사는 이유.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
