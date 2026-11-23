"""step03 — 그날의 유형 2열을 만들고, 라벨할 수 없는 행을 버린다.

읽기  data/feature/{year}_{mm}.parquet
쓰기  data/scen/{year}.parquet              step02 의 열 + scen_origin, scen_dest
      results/step03_scenario_summary.csv   연도별 유형 분포와 버린 행
      results/step03_dropped.csv            왜 버렸는지

이 2열은 예약 시점에 알 수 없는 값이다
--------------------------------------
재료가 전부 비행이 끝난 뒤에 생기는 값(실제 지연, 원인별 지연 분)이다. 그래서
모델이 추측하지 않고 손님이 조건으로 고른다. 저장도 따로 하지 않고 같은 파일에
두지만, 이후 단계에서 '예약 시점 변수'와 구분해 다룬다.

라벨을 붙이는 규칙 (출발지·도착지 각각 같은 방식)
-------------------------------------------------
1. 라벨은 **(공항, 날짜)** 에 붙인다. 항공편 하나에 "이 편은 날씨 때문에 늦었다"
   를 붙이지 않는다. BTS 는 **이미 늦은 편에만** 원인을 기록하므로, 정시에 온
   편과 비교할 기준이 없어 결과로 결과를 설명하게 된다.

2. '실패' 는 15분을 넘는 지연(= class 2 이상) 또는 미도착.

3. 그날이 **나쁜 날**인지: 그 (공항, 날짜) 의 실패율이 **그 공항의 평소 수준**
   보다 높은가. 평소 수준은 그 공항의 **그해 여름 날짜별 실패율의 평균**이다.
   해마다 따로 재므로 연도 사이에 값이 섞이지 않는다.

4. 나쁜 날이면 **원인을 고른다**: 원인별 지연 분(分)의 비중을 **그 공항 안에서
   표준화**한 뒤 가장 큰 것. 비중을 그대로 쓰면 산악 지역 공항은 늘 '날씨일'이
   된다 — 그 공항 기준으로 이상한 쪽을 골라야 한다.

5. 나쁜 날이 아니면 **정상일**.

6. **자기 편은 집계에서 뺀다(leave-one-out).** 그날의 라벨을 그날의 편들로
   만드는데 라벨을 받는 편도 그중 하나다. 500편 공항에서는 무의미하지만 작은
   공항에서는 그 편이 자기 답을 읽게 된다. 평소 수준은 한 시즌 전체(수천 편)
   평균이라 한 편의 영향이 없어 빼지 않는다.

7. 그 (공항, 날짜) 에 20편 미만이면 라벨하지 않고 **그 행을 버린다.** 몇 편으로
   그날을 규정할 수 없다. 출발지와 도착지 **둘 다** 라벨된 편만 남긴다.

실행  python src/step03_scenario.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

CAUSES = config.CAUSES
MIN_FLIGHTS = config.SCENARIO_MIN_FLIGHTS


def label_side(frame: pd.DataFrame, airport: str) -> tuple[np.ndarray, np.ndarray, int]:
    """한 쪽(출발지 또는 도착지) 기준의 유형과 라벨 가능 여부.

    반환: (유형 0~4, 라벨 가능 여부, 나쁜 날인데 원인 분이 없어 정상일로 둔 편수)
    """
    group = frame.groupby([airport, "FlightDate"], observed=True)
    own_fail = frame["failed"].to_numpy(dtype="float64")

    # --- 그날의 실패율 (자기 편 제외) ------------------------------------- #
    n = group["failed"].transform("size").to_numpy(dtype="float64")
    fail_sum = group["failed"].transform("sum").to_numpy(dtype="float64")
    loo_fail = (fail_sum - own_fail) / np.maximum(n - 1.0, 1.0)

    # --- 그날의 원인별 분 비중 (자기 편 제외) ----------------------------- #
    loo_min = {}
    for cause in CAUSES:
        total = group[f"min_{cause}"].transform("sum").to_numpy(dtype="float64")
        loo_min[cause] = total - frame[f"min_{cause}"].to_numpy(dtype="float64")
    loo_total = np.sum([loo_min[c] for c in CAUSES], axis=0)
    shares = np.stack(
        [np.divide(loo_min[c], loo_total, out=np.zeros_like(loo_total),
                   where=loo_total > 0) for c in CAUSES], axis=1)

    # --- 그 공항의 평소 수준 (그해 여름, 날짜별 실패율의 평균) ------------ #
    day_fail = group["failed"].mean().rename("day_fail").reset_index()
    usual_map = day_fail.groupby(airport, observed=True)["day_fail"].mean()
    usual = frame[airport].map(usual_map).to_numpy(dtype="float64")

    # --- 그 공항 안에서 비중을 표준화 ------------------------------------- #
    share_frame = pd.DataFrame(shares, columns=CAUSES, index=frame.index)
    share_frame["_airport"] = frame[airport].to_numpy()
    grouped = share_frame.groupby("_airport", observed=True)[CAUSES]
    stats = (grouped.mean().add_suffix("_mean")
             .join(grouped.std().add_suffix("_std")).reset_index())
    merged = (frame[[airport]].rename(columns={airport: "_airport"})
              .merge(stats, on="_airport", how="left"))
    z = np.stack([
        (shares[:, i] - merged[f"{c}_mean"].to_numpy(dtype="float64"))
        / np.maximum(merged[f"{c}_std"].to_numpy(dtype="float64"), 1e-9)
        for i, c in enumerate(CAUSES)], axis=1)

    # --- 판정 ------------------------------------------------------------- #
    labelled = (n >= MIN_FLIGHTS) & np.isfinite(usual)
    bad = labelled & (loo_fail > usual)
    attributable = bad & (loo_total > 0)

    kind = np.zeros(len(frame), dtype="int8")
    kind[attributable] = (np.nan_to_num(z[attributable], nan=-np.inf)
                          .argmax(axis=1) + 1).astype("int8")

    # 나쁜 날인데 원인 분이 전혀 없는 편(결항만 몰린 날 등)은 정상일로 둔다.
    unattributed = int((bad & ~attributable).sum())
    return kind, labelled, unattributed


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()

    rows, dropped = [], []
    for year in config.YEARS:
        parts = [pd.read_parquet(config.FEATURE_DIR / f"{year}_{m:02d}.parquet")
                 for m in config.MONTHS]
        frame = pd.concat(parts, ignore_index=True)
        del parts
        n_start = len(frame)

        frame["failed"] = (frame["y_class"] >= config.FAIL_FROM_CLASS).astype("float64")

        kinds, keeps, unattr = {}, {}, {}
        for side, column in (("origin", "Origin"), ("dest", "Dest")):
            kinds[side], keeps[side], unattr[side] = label_side(frame, column)

        keep = keeps["origin"] & keeps["dest"]
        frame["scen_origin"] = kinds["origin"]
        frame["scen_dest"] = kinds["dest"]
        out = frame.loc[keep].drop(columns=["failed"]).reset_index(drop=True)

        out.to_parquet(config.SCEN_DIR / f"{year}.parquet", index=False)

        share = out["scen_origin"].value_counts(normalize=True)
        rows.append({
            "year": year, "편수": len(out),
            **{name: round(float(share.get(i, 0.0)), 4)
               for i, name in enumerate(config.SCENARIO_NAMES)},
        })
        dropped.append({
            "year": year, "원본": n_start, "남은편수": len(out),
            "버린편수": n_start - len(out),
            "버린비율": round((n_start - len(out)) / n_start, 4),
            "출발지_미라벨": int((~keeps["origin"]).sum()),
            "도착지_미라벨": int((~keeps["dest"]).sum()),
            "나쁜날_원인없음_출발지": unattr["origin"],
            "나쁜날_원인없음_도착지": unattr["dest"],
        })
        print(f"  {year}  {len(out):>9,} 편  "
              f"(버림 {n_start - len(out):,} = "
              f"{(n_start - len(out)) / n_start:.1%})", flush=True)
        del frame, out

    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "step03_scenario_summary.csv",
                   index=False, encoding="utf-8-sig")
    drop_table = pd.DataFrame(dropped)
    drop_table.to_csv(config.RESULTS_DIR / "step03_dropped.csv",
                      index=False, encoding="utf-8-sig")

    print("\n=== 출발지 유형 분포 ===")
    print(summary.to_string(index=False))
    print("\n=== 버린 행 ===")
    print(drop_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
