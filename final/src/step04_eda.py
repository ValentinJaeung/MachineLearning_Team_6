"""step04 — EDA. 그리고 설정값 하나를 여기서 정한다(공항을 몇 개로 끊을지).

읽기  data/scen/{year}.parquet
쓰기  results/step04_class_distribution.csv   10구간 분포 (분할별)
      results/step04_cause_share.csv          원인별 지연 분 비중
      results/step04_hour_ontime.csv          시각별 정시 도착률
      results/step04_cell_counts.csv          25칸 편수 (분할별)
      results/step04_airport_coverage.csv     공항 누적 운항 비율
      results/step04_airport_list.json        ★ 원-핫할 공항 목록 (step05 가 읽는다)
      figures/step04_*.png                    그림 5장

설정값은 학습 연도만 보고 정한다
--------------------------------
공항을 몇 개까지 개별 열로 둘지는 **학습 연도(2022~2024)의 운항만** 보고
정한다. 검증·평가 연도를 보고 정하면 그 연도에 대한 정보가 설계에 새어 든다.

왜 공항을 끊는가: 공항 수가 수백 개라서 전부 원-핫하면 선형 모델과 MLP 의 열이
불필요하게 넓어지고, 뒤쪽 공항은 학습 편수가 너무 적어 열 하나가 잡음이 된다.
누적 운항이 AIRPORT_COVERAGE 를 덮는 지점에서 끊고 나머지는 '기타' 로 묶는다.

실행  python src/step04_eda.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
import plotting
from plotting import plt

COLUMNS = ["Year", "Origin", "Dest", "crs_dep_hour", "y_class",
           "scen_origin", "scen_dest"] + [f"min_{c}" for c in config.CAUSES]


def split_of(year: int) -> str:
    for name, years in config.SPLIT_YEARS.items():
        if year in years:
            return name
    raise ValueError(year)


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    frames = []
    for year in config.YEARS:
        part = pd.read_parquet(config.SCEN_DIR / f"{year}.parquet", columns=COLUMNS)
        part["split"] = split_of(year)
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    del frames
    train = data[data["split"] == "train"]
    print(f"전체 {len(data):,} 편 / 학습 {len(train):,} 편\n")

    # ------------------------------------------------------------------ #
    # 1. 10구간 분포 — 평균 하나로 답할 수 없다는 근거
    # ------------------------------------------------------------------ #
    dist = (data.groupby(["split", "y_class"], observed=True).size()
            .unstack("split", fill_value=0))
    dist = dist.div(dist.sum(axis=0), axis=1).reindex(range(config.N_CLASS))
    dist.index = config.CLASS_NAMES
    dist = dist[list(config.SPLITS)].round(6)
    dist.to_csv(config.RESULTS_DIR / "step04_class_distribution.csv",
                encoding="utf-8-sig")
    print("=== 10구간 분포 ===")
    print(dist.to_string())

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(config.N_CLASS)
    for i, split in enumerate(config.SPLITS):
        ax.bar(x + (i - 1) * 0.27, dist[split], 0.27, label=split)
    ax.set_xticks(x, config.CLASS_NAMES, rotation=40, ha="right")
    ax.set_ylabel("비율")
    ax.set_title("step04 — 10구간 정답 분포 (소수의 큰 지연이 긴 꼬리를 만든다)")
    ax.legend()
    fig.savefig(config.FIGURE_DIR / "step04_class_distribution.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. 원인별 지연 분 비중 — 유형 변수를 만드는 근거
    # ------------------------------------------------------------------ #
    minutes = {c: float(data[f"min_{c}"].sum()) for c in config.CAUSES}
    total = sum(minutes.values())
    cause = pd.DataFrame({
        "원인": list(minutes),
        "지연 분(합)": [round(v) for v in minutes.values()],
        "비중": [round(v / total, 6) for v in minutes.values()],
    }).sort_values("비중", ascending=False)
    cause.to_csv(config.RESULTS_DIR / "step04_cause_share.csv",
                 index=False, encoding="utf-8-sig")
    print("\n=== 원인별 지연 분 비중 ===")
    print(cause.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(cause["원인"], cause["비중"], color="#4c6ef5")
    for i, v in enumerate(cause["비중"]):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("지연 분 기준 비중")
    ax.set_title("step04 — 지연은 어디서 오는가")
    fig.savefig(config.FIGURE_DIR / "step04_cause_share.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. 시각별 정시 도착률 — 시각을 숫자 하나로 넣지 않는 근거
    # ------------------------------------------------------------------ #
    hour = (data.assign(ontime=data["y_class"] <= 1)
            .groupby("crs_dep_hour", observed=True)
            .agg(편수=("ontime", "size"), 정시율=("ontime", "mean"))
            .reset_index())
    hour["정시율"] = hour["정시율"].round(6)
    hour.to_csv(config.RESULTS_DIR / "step04_hour_ontime.csv",
                index=False, encoding="utf-8-sig")
    print("\n=== 시각별 15분 이내 도착률 (일부) ===")
    print(hour[hour["crs_dep_hour"].isin([5, 8, 11, 14, 17, 20, 23])]
          .to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(hour["crs_dep_hour"], hour["정시율"], "o-", color="#2b8a3e")
    ax.set_xlabel("예정 출발 시각 (시)")
    ax.set_ylabel("15분 이내 도착률")
    ax.set_title("step04 — 하루 동안 정시율이 어떻게 변하는가 (직선이 아니다)")
    ax.set_xticks(range(0, 24, 2))
    fig.savefig(config.FIGURE_DIR / "step04_hour_ontime.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. 25칸 편수 — 얇은 칸의 신뢰도
    # ------------------------------------------------------------------ #
    cells = []
    for split in config.SPLITS:
        part = data[data["split"] == split]
        table = (part.groupby(["scen_origin", "scen_dest"], observed=True)
                 .size().unstack(fill_value=0)
                 .reindex(index=range(5), columns=range(5), fill_value=0))
        table.index = config.SCENARIO_NAMES
        table.columns = config.SCENARIO_NAMES
        table.insert(0, "split", split)
        cells.append(table.rename_axis("출발지 유형"))
    cell_table = pd.concat(cells)
    cell_table.to_csv(config.RESULTS_DIR / "step04_cell_counts.csv",
                      encoding="utf-8-sig")
    test_cells = cell_table[cell_table["split"] == "test"].drop(columns="split")
    thin = int((test_cells.to_numpy() < config.MIN_CELL_FLIGHTS).sum())
    print(f"\n=== 테스트 25칸 편수 (판단 불가 기준 {config.MIN_CELL_FLIGHTS:,}편 "
          f"미만 = {thin}칸) ===")
    print(test_cells.to_string())

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    values = test_cells.to_numpy(dtype="float64")
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(5), config.SCENARIO_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), config.SCENARIO_NAMES)
    ax.set_xlabel("도착지 유형")
    ax.set_ylabel("출발지 유형")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(values[i, j]):,}", ha="center", va="center",
                    fontsize=8,
                    color="white" if values[i, j] > values.max() / 2 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("step04 — 테스트 연도 25칸 편수")
    fig.savefig(config.FIGURE_DIR / "step04_cell_counts.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5. 공항을 몇 개로 끊을지 — 학습 연도만 보고 정한다
    # ------------------------------------------------------------------ #
    appearance = pd.concat([train["Origin"], train["Dest"]]).value_counts()
    cumulative = appearance.cumsum() / appearance.sum()

    # 최소 운항 편수로 끊는다. 누적 95% 지점은 비교용으로만 기록한다 —
    # 이 데이터에서는 운항 2만 편짜리 공항까지 묶어 버려 기준으로 쓰지 않는다.
    keep_mask = appearance >= config.MIN_AIRPORT_FLIGHTS
    keep_n = int(keep_mask.sum())
    keep = sorted(appearance.index[keep_mask])
    coverage_n = int((cumulative < config.AIRPORT_COVERAGE).sum() + 1)

    coverage = pd.DataFrame({
        "순위": range(1, len(appearance) + 1),
        "공항": appearance.index,
        "운항수": appearance.to_numpy(),
        "누적비율": cumulative.to_numpy().round(6),
    })
    coverage.to_csv(config.RESULTS_DIR / "step04_airport_coverage.csv",
                    index=False, encoding="utf-8-sig")
    (config.RESULTS_DIR / "step04_airport_list.json").write_text(
        json.dumps({
            "rule": f"학습 연도 운항 {config.MIN_AIRPORT_FLIGHTS}편 이상",
            "min_flights": config.MIN_AIRPORT_FLIGHTS,
            "n_airports_total": int(len(appearance)),
            "n_airports_kept": keep_n,
            "coverage_reached": float(cumulative.iloc[keep_n - 1]),
            "n_if_coverage_rule": coverage_n,
            "coverage_rule_note": (
                f"누적 {config.AIRPORT_COVERAGE:.0%} 규칙을 쓰면 {coverage_n}개만 "
                f"남아 운항 {int(appearance.iloc[coverage_n - 1]):,}편짜리 공항까지 "
                f"'기타'로 묶인다. 그래서 쓰지 않았다."),
            "decided_on": "train years only",
            "airports": keep,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 공항 ===")
    print(f"학습 연도에 나타난 공항 {len(appearance)}개")
    print(f"  채택 규칙: 운항 {config.MIN_AIRPORT_FLIGHTS:,}편 이상 "
          f"→ {keep_n}개 (운항의 {cumulative.iloc[keep_n - 1]:.2%}), "
          f"나머지 {len(appearance) - keep_n}개는 '{config.OTHER_LABEL}'")
    print(f"  (비교) 누적 {config.AIRPORT_COVERAGE:.0%} 규칙 → {coverage_n}개, "
          f"운항 {int(appearance.iloc[coverage_n - 1]):,}편짜리 공항까지 묶인다")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(coverage["순위"], coverage["누적비율"], color="#4c6ef5")
    ax.axhline(config.AIRPORT_COVERAGE, color="#adb5bd", ls="--", lw=1.2,
               label=f"누적 {config.AIRPORT_COVERAGE:.0%} (쓰지 않음)")
    ax.axvline(keep_n, color="red", ls=":", lw=1.4,
               label=f"채택: 운항 {config.MIN_AIRPORT_FLIGHTS:,}편 이상 = {keep_n}개")
    ax.set_xlabel("공항 순위 (운항 많은 순)")
    ax.set_ylabel("누적 운항 비율")
    ax.set_title("step04 — 공항을 몇 개로 끊을지 (학습 연도만 보고 결정)")
    ax.legend()
    fig.savefig(config.FIGURE_DIR / "step04_airport_coverage.png")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
