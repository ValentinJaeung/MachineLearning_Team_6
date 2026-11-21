"""
Step 8 - EDA: fix the settings the later steps need, and describe the structure.

Reads  data/dataset/ (step 7), data/clean/ (step 2)
Writes results/step08_settings.json      <- the four settings, with their evidence
       results/step08_*.csv
       figures/step08_*.png              (nine figures)

Two jobs, two viewing ranges
----------------------------
1. SETTINGS - looked at on the TRAINING years only (2022-24, and the 2021-23
   source years behind them). Validation and test are not opened. Merely
   looking at the test year's shape influences a choice ("the test year is
   August-heavy, so let's treat Month like this"), and that is cheating.

2. STRUCTURE - described over the whole period, because it is descriptive
   statistics of post-hoc fields and feeds no setting and no feature choice.
   Its role stops at "what is worth checking".

The setting fixed here
----------------------
  one-hot airport count  smallest N whose cumulative share of training
                         departures reaches 95%

Usage:  python src/step08_eda.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import plotting
from plotting import plt

TRAIN_YEARS = config.SPLIT_YEARS["train"]
ONEHOT_COVERAGE = 0.95


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def class_cdf(counts: np.ndarray) -> np.ndarray:
    """Cumulative probability over the nine arrival bins (class 9 excluded)."""
    total = counts.sum()
    if total == 0:
        return np.zeros(len(config.EDGES))
    return np.cumsum(counts[:len(config.EDGES)]) / total


def guaranteed_delay(counts: np.ndarray, level: float = 0.90) -> float:
    """Minutes of delay by which `level` of these flights had arrived.

    Read straight off the empirical bin CDF, exactly the way the service reads
    it off a predicted distribution. NaN when the group never reaches `level`
    even at the last edge - which is the honest answer, not a made-up number.
    """
    cdf = class_cdf(counts)
    if cdf[-1] < level:
        return float("nan")
    return float(np.interp(level, cdf, config.EDGES))


def load_train(columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(config.DATASET_DIR / "train.parquet", columns=columns)


# --------------------------------------------------------------------------- #
# 1. the ten-bin distribution  (the service's opening sentence)
# --------------------------------------------------------------------------- #


def figure_class_distribution(train: pd.DataFrame) -> dict:
    counts = train["y_class"].value_counts().reindex(
        range(config.N_CLASS), fill_value=0).to_numpy()
    share = counts / counts.sum()
    cdf = class_cdf(counts)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(range(config.N_CLASS), share, color="#4c6ef5")
    axes[0].set_xticks(range(config.N_CLASS), config.CLASS_NAMES,
                       rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(share):
        axes[0].text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=8)
    axes[0].set_ylim(0, share.max() * 1.18)
    axes[0].set_ylabel("비중")
    axes[0].set_title("10구간 도착 결과 분포 (train 2022–24)")

    axes[1].plot(config.EDGES, cdf, "o-", color="#e8590c")
    for level in (0.80, 0.90, 0.95):
        axes[1].axhline(level, color="grey", ls=":", lw=1)
        if cdf[-1] >= level:
            x = np.interp(level, cdf, config.EDGES)
            axes[1].plot([x], [level], "k*", ms=11)
            axes[1].annotate(f"{level:.0%} → +{x:.0f}분", (x, level),
                             textcoords="offset points", xytext=(8, -12),
                             fontsize=9)
    axes[1].set_xlabel("예정 도착 대비 지연(분)")
    axes[1].set_ylabel("그 시각까지 도착할 확률")
    axes[1].set_title("누적 분포 — '보장 시각'이 읽히는 자리")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_class_distribution.png")
    plt.close(fig)

    print("\n=== 10구간 분포 (train) ===")
    for i, (name, s) in enumerate(zip(config.CLASS_NAMES, share)):
        print(f"  {i} {name:<16} {s:7.2%}")
    print(f"  정시·조기 {share[0]:.1%} — 항공사가 적어 준 도착 시각은 "
          f"실제로는 {share[0]:.0%}짜리 약속이다.")
    print(f"  90% 보장 지연: +{guaranteed_delay(counts):.0f}분  "
          f"| 95%: +{guaranteed_delay(counts, 0.95):.0f}분")

    pd.DataFrame({"class": range(config.N_CLASS), "name": config.CLASS_NAMES,
                  "count": counts, "share": share}).to_csv(
        config.RESULTS_DIR / "step08_class_distribution.csv",
        index=False, encoding="utf-8-sig")
    return {"c0_share": float(share[0]),
            "guaranteed_90_overall_min": guaranteed_delay(counts),
            "guaranteed_95_overall_min": guaranteed_delay(counts, 0.95)}


# --------------------------------------------------------------------------- #
# 2. departure hour  (is the shape a straight line?)
# --------------------------------------------------------------------------- #


def figure_hour_profile(train: pd.DataFrame) -> dict:
    table = (train.groupby("crs_dep_hour")["y_class"]
             .value_counts().unstack(fill_value=0)
             .reindex(columns=range(config.N_CLASS), fill_value=0))
    rows = []
    for hour, counts in table.iterrows():
        c = counts.to_numpy()
        rows.append({
            "crs_dep_hour": int(hour), "n": int(c.sum()),
            "ontime_rate": float(c[0] / c.sum()),
            "fail_rate": float(c[2:].sum() / c.sum()),
            "noarrive_rate": float(c[config.NOARRIVE] / c.sum()),
            "guaranteed_90_min": guaranteed_delay(c),
        })
    profile = pd.DataFrame(rows)
    profile = profile[profile["n"] >= 1000]
    profile.to_csv(config.RESULTS_DIR / "step08_hour_profile.csv",
                   index=False, encoding="utf-8-sig")

    fig, ax1 = plt.subplots(figsize=(9.5, 5))
    ax1.plot(profile["crs_dep_hour"], profile["ontime_rate"], "o-",
             color="#4c6ef5", label="정시·조기 도착 비율")
    ax1.plot(profile["crs_dep_hour"], profile["noarrive_rate"], "^-",
             color="#c92a2a", label="미도착 비율")
    ax1.set_xlabel("예정 출발 시(時)")
    ax1.set_ylabel("비율")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.bar(profile["crs_dep_hour"], profile["guaranteed_90_min"],
            alpha=0.22, color="#e8590c", label="90% 보장 지연(분)")
    ax2.set_ylabel("90% 보장 지연(분)", color="#e8590c")
    ax2.grid(False)
    lines = ax1.get_lines() + [ax2.containers[0]]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)
    ax1.set_title("출발 시각별 결과 — 직선이 아니라 꺾인 모양 (train 2022–24)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_hour_profile.png")
    plt.close(fig)

    best = profile.loc[profile["ontime_rate"].idxmax()]
    worst = profile.loc[profile["ontime_rate"].idxmin()]
    print("\n=== 출발 시각별 ===")
    print(f"  가장 좋은 시각 {int(best['crs_dep_hour']):02d}시: 정시 "
          f"{best['ontime_rate']:.1%}, 90% 보장 +{best['guaranteed_90_min']:.0f}분")
    print(f"  가장 나쁜 시각 {int(worst['crs_dep_hour']):02d}시: 정시 "
          f"{worst['ontime_rate']:.1%}, 90% 보장 +{worst['guaranteed_90_min']:.0f}분")
    print("  모양이 단조롭지 않다 → 선형 모델에는 시각을 원-핫으로 주어야 한다(9단계).")
    return {"best_hour": int(best["crs_dep_hour"]),
            "worst_hour": int(worst["crs_dep_hour"]),
            "guaranteed_90_best_min": float(best["guaranteed_90_min"]),
            "guaranteed_90_worst_min": float(worst["guaranteed_90_min"])}
# --------------------------------------------------------------------------- #
# 5. how many airports to one-hot
# --------------------------------------------------------------------------- #


def figure_onehot_cutoff(train: pd.DataFrame) -> dict:
    counts = train["Origin"].value_counts()
    cumulative = counts.cumsum() / counts.sum()
    n_airports = int((cumulative < ONEHOT_COVERAGE).sum() + 1)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(range(1, len(cumulative) + 1), cumulative.to_numpy(), color="#4c6ef5")
    ax.axhline(ONEHOT_COVERAGE, color="grey", ls=":")
    ax.axvline(n_airports, color="#c92a2a", ls="--",
               label=f"상위 {n_airports}개에서 {ONEHOT_COVERAGE:.0%} 도달")
    ax.set_xlabel("공항 순위 (운항량 기준)")
    ax.set_ylabel("누적 운항 비중")
    ax.set_title(f"원-핫할 공항 개수 — 전체 {len(counts)}개 중 상위 {n_airports}개")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_onehot_cutoff.png")
    plt.close(fig)

    tail = counts.iloc[n_airports:]
    print("\n=== 원-핫할 공항 개수 ===")
    print(f"  train 공항 {len(counts)}개 중 상위 {n_airports}개가 운항의 "
          f"{ONEHOT_COVERAGE:.0%}를 차지한다.")
    print(f"  나머지 {len(tail)}개는 공항당 평균 {tail.mean():,.0f}편 "
          f"(전체의 {tail.sum() / counts.sum():.1%}) → '기타'로 묶는다. "
          "하루 한두 편짜리 공항은 계수를 추정할 데이터가 없고 열만 늘린다.")
    return {"n_airports_total": int(len(counts)), "n_onehot": n_airports,
            "coverage": ONEHOT_COVERAGE,
            "other_airports": int(len(tail)),
            "other_share": float(tail.sum() / counts.sum())}


# --------------------------------------------------------------------------- #
# 6. correlation among numeric inputs
# --------------------------------------------------------------------------- #


def figure_correlation(train: pd.DataFrame) -> dict:
    numeric = (["crs_dep_hour", "crs_arr_hour", "Distance", "CRSElapsedTime"]
               + config.SCHEDULE_FEATURES)
    corr = train[numeric].corr()

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric)), numeric, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(numeric)), numeric, fontsize=8)
    for i in range(len(numeric)):
        for j in range(len(numeric)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title("수치형 입력 간 상관")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_correlation.png")
    plt.close(fig)
    corr.to_csv(config.RESULTS_DIR / "step08_correlation.csv", encoding="utf-8-sig")

    pairs = (corr.where(~np.eye(len(numeric), dtype=bool)).abs().stack()
             .sort_values(ascending=False))
    top = pairs.index[0]
    print("\n=== 변수 간 상관 ===")
    print(f"  가장 강한 쌍: {top[0]} ↔ {top[1]}  r = {corr.loc[top]:.3f}")
    print("  거리와 예정 소요시간은 사실상 같은 정보다. 둘 다 넣어도 "
          "트리는 손해를 보지 않지만, 선형 모델에서는 계수 해석이 불안정해진다.")
    return {"max_pair": [top[0], top[1]], "max_corr": float(corr.loc[top])}


# --------------------------------------------------------------------------- #
# 7. do the report-card features separate the target at all?
# --------------------------------------------------------------------------- #


def figure_feature_vs_target(train: pd.DataFrame) -> dict:
    features = config.SCHEDULE_FEATURES + ["Distance", "CRSElapsedTime",
                                           "crs_dep_hour"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    rows = []
    for ax, name in zip(axes.ravel(), features):
        bins = pd.qcut(train[name], 10, duplicates="drop", labels=False)
        grouped = train.assign(_b=bins).groupby("_b")
        ontime = grouped["y_class"].apply(lambda s: float((s == 0).mean()))
        centre = grouped[name].mean()
        ax.plot(centre.to_numpy(), ontime.to_numpy(), "o-", color="#4c6ef5")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("십분위 중앙값", fontsize=8)
        ax.set_ylabel("정시·조기 비율", fontsize=8)
        spread = float(ontime.max() - ontime.min())
        ax.annotate(f"폭 {spread:.3f}", (0.03, 0.06), xycoords="axes fraction",
                    fontsize=9)
        rows.append({"feature": name, "ontime_spread_over_deciles": spread})
    fig.suptitle("변수별 십분위와 정시 비율 — 혼자서 얼마나 갈라 보이는가 (train)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_feature_vs_target.png")
    plt.close(fig)

    table = pd.DataFrame(rows).sort_values("ontime_spread_over_deciles",
                                           ascending=False)
    table.to_csv(config.RESULTS_DIR / "step08_feature_spread.csv",
                 index=False, encoding="utf-8-sig")
    print("\n=== 변수와 정답의 관계 (단독) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  ⚠ 이것은 상관이지 기여가 아니다. 단독으로 잘 가르는 변수여도 "
          "이미 있는 변수가 같은 정보를 담고 있으면 보탬이 없다.")
    return {"spread": table.set_index("feature")[
        "ontime_spread_over_deciles"].round(4).to_dict()}


# --------------------------------------------------------------------------- #
# 8. why flights fail  (C group; description only)
# --------------------------------------------------------------------------- #


def figure_cause_mix() -> dict:
    causes = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
              "LateAircraftDelay"]
    labels = ["항공사", "날씨", "공역혼잡(NAS)", "보안", "연결편 지연"]

    parts = []
    for year in config.YEARS:
        for month in config.MONTHS:
            parts.append(pd.read_parquet(
                config.CLEAN_DIR / f"{year}_{month:02d}.parquet",
                columns=["Year", "crs_dep_hour"] + causes))
    frame = pd.concat(parts, ignore_index=True)
    for c in causes:
        frame[c] = frame[c].fillna(0.0)

    overall = frame[causes].sum()
    overall_share = overall / overall.sum()

    by_hour = frame.groupby("crs_dep_hour")[causes].sum()
    by_hour = by_hour.div(by_hour.sum(axis=1), axis=0)
    by_hour = by_hour[by_hour.index <= 23]

    by_year = frame.groupby("Year")[causes].sum()
    by_year = by_year.div(by_year.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].pie(overall_share, labels=labels, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
    axes[0].set_title("전체 지연의 원인 구성 (2021–2026)")

    bottom = np.zeros(len(by_hour))
    for c, label in zip(causes, labels):
        axes[1].bar(by_hour.index, by_hour[c], bottom=bottom, label=label)
        bottom += by_hour[c].to_numpy()
    axes[1].set_xlabel("예정 출발 시(時)")
    axes[1].set_ylabel("원인별 분(分) 비중")
    axes[1].set_title("시각이 바뀌면 '실패의 종류'도 바뀐다")
    axes[1].legend(fontsize=7, ncol=2)

    bottom = np.zeros(len(by_year))
    for c, label in zip(causes, labels):
        axes[2].bar(by_year.index.astype(str), by_year[c], bottom=bottom, label=label)
        bottom += by_year[c].to_numpy()
    axes[2].set_ylabel("원인별 분(分) 비중")
    axes[2].set_title("연도별 추세")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_cause_mix.png")
    plt.close(fig)

    by_hour.to_csv(config.RESULTS_DIR / "step08_cause_by_hour.csv",
                   encoding="utf-8-sig")
    print("\n=== 지연 원인 구성 (전 기간, 기술 통계) ===")
    for label, v in zip(labels, overall_share):
        print(f"  {label:<14} {v:6.1%}")
    print(f"  '연결편 지연'이 {overall_share['LateAircraftDelay']:.0%}를 차지한다 — "
          "③ 연결편 변수를 만든 근거가 여기 있다.")
    print("  ⚠ 전 기간을 본 기술이므로, 여기서 나온 관찰을 그대로 설계 근거로 "
          "쓰지 않는다. 입력 변수는 예약 시점 가용성과 기존 변수로 재구성되지 "
          "않는지를 기준으로 고른다.")
    return {"overall_share": overall_share.round(4).to_dict()}


# --------------------------------------------------------------------------- #
# 9. how many candidate flights a passenger actually has
# --------------------------------------------------------------------------- #


def figure_candidates(train: pd.DataFrame) -> dict:
    per_day = train.groupby(["Origin", "Dest", "FlightDate"]).size()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.hist(per_day.clip(upper=30), bins=np.arange(0, 32) - 0.5, color="#4c6ef5")
    ax.axvline(per_day.median(), color="#c92a2a", ls="--",
               label=f"중앙값 {per_day.median():.0f}편")
    ax.set_xlabel("노선·날짜당 운항 편수 (30에서 절단)")
    ax.set_ylabel("노선-날짜 수")
    ax.set_title("손님에게 제시할 선택지는 몇 개인가 (train)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step08_candidates_per_day.png")
    plt.close(fig)

    share_1 = float((per_day == 1).mean())
    print("\n=== 하루 후보 편수 ===")
    print(f"  노선-날짜당 중앙값 {per_day.median():.0f}편, 평균 {per_day.mean():.1f}편.")
    print(f"  단 한 편뿐인 노선-날짜가 {share_1:.1%} — 그런 노선에서는 "
          "'가장 늦은 편을 고른다'는 서비스가 성립하지 않는다. 한계로 적는다.")
    return {"median_candidates": float(per_day.median()),
            "mean_candidates": float(per_day.mean()),
            "single_flight_share": share_1}


# --------------------------------------------------------------------------- #


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    train = load_train([
        "y_class", "crs_dep_hour", "crs_arr_hour", "Distance", "CRSElapsedTime",
        "Origin", "Dest", "FlightDate"]
        + config.SCHEDULE_FEATURES)

    print(f"train rows: {len(train):,}  (years {TRAIN_YEARS})")
    print("설정값은 학습 연도만 보고 정한다. val 2025 · test 2026 은 열지 않는다.")

    settings = {
        "viewing_range": {
            "settings_from": TRAIN_YEARS,
            "structure_from": "2021-2026 (기술 통계 전용, 설계 근거로 쓰지 않음)",
        },
        "target_distribution": figure_class_distribution(train),
        "departure_hour": figure_hour_profile(train),
        "one_hot_airports": figure_onehot_cutoff(train),
        "correlation": figure_correlation(train),
        "single_feature_spread": figure_feature_vs_target(train),
        "candidates_per_day": figure_candidates(train),
        "cause_mix": figure_cause_mix(),
    }

    config.EDA_SETTINGS_PATH.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== 8단계 확정 설정값 ===")
    print(f"  원-핫 공항     상위 {settings['one_hot_airports']['n_onehot']}개 "
          f"(학습 연도 누적 운항 95% 지점)")
    print("  데이터를 봐서 정했다. 임의로 고른 숫자는 없다.")
    print(f"\nsettings: {config.EDA_SETTINGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
