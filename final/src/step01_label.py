"""step01 — 원자료를 정제하고 10구간 정답을 만든다.

읽기  {RAW_DIR}/{year}_{mm}.parquet          BTS On-Time Performance (이미 받아 둔 것)
쓰기  data/label/{year}_{mm}.parquet         정제된 편 + y_class
      results/step01_rows.csv                연도별 편수와 제외된 행
      results/step01_class_distribution.csv  10구간 분포

정답을 만드는 규칙
------------------
도착 지연은 ArrDelay(예정 도착 시각 대비 분)로 잰다.

  결항                                   → 미도착(class 9)
  회항했고 목적지에 결국 도착            → DivArrDelay 를 지연으로 써서 구간에 넣는다
  회항했고 목적지에 끝내 닿지 못함       → 미도착(class 9)
  그 외                                  → ArrDelay 로 구간에 넣는다

회항 편을 따로 두는 이유: 회항 중 상당수는 급유나 기상 대기 후 결국 목적지에
도착하고, BTS 는 그 최종 지연을 DivArrDelay 에 따로 기록한다(일반 편의 ArrDelay
자리는 비어 있다). 손님에게는 "많이 늦게 도착했다" 와 다를 바 없으므로 지연
구간에 넣는다. 미도착 칸은 결항과, 회항 후 끝내 못 간 편뿐이다.

제외하는 행
-----------
결항·회항이 아닌데 ArrDelay 가 비어 있는 행(결과를 알 수 없다), 그리고 예정
시각·공항·거리 같은 예약 시점 정보가 비어 있는 행. 지연이 큰 행은 제외하지
않는다 — 긴 꼬리가 바로 우리가 맞혀야 하는 대상이다.

실행  python src/step01_label.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config

READ = [
    "Year", "Month", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "Distance", "CRSElapsedTime",
    "ArrDelay", "Cancelled", "Diverted", "DivReachedDest", "DivArrDelay",
] + list(config.CAUSE_COLUMN.values())

# 예약 시점에 반드시 있어야 하는 값 — 하나라도 비면 그 편은 못 쓴다.
REQUIRED = ["FlightDate", "Reporting_Airline", "Origin", "Dest",
            "CRSDepTime", "CRSArrTime", "Distance", "CRSElapsedTime"]

OUT = [
    "FlightDate", "Year", "Month", "DayOfWeek",
    "Reporting_Airline", "Origin", "Dest",
    "crs_dep_hour", "crs_arr_hour", "Distance", "CRSElapsedTime",
    "arr_delay_min", "y_class",
] + [f"min_{c}" for c in config.CAUSES]


def to_hour(series: pd.Series) -> pd.Series:
    """1435 → 14. 2400 은 0 시로 본다."""
    return (series.astype("float64") // 100 % 24).astype("int8")


def label(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """한 달치를 정제하고 y_class 를 붙인다."""
    n_start = len(frame)

    cancelled = frame["Cancelled"].fillna(0) == 1
    diverted = frame["Diverted"].fillna(0) == 1
    reached = frame["DivReachedDest"].fillna(0) == 1

    # --- 도착 지연 분(分) ------------------------------------------------- #
    delay = frame["ArrDelay"].astype("float64")
    # 회항했지만 목적지에 도착한 편은 DivArrDelay 를 쓴다
    div_ok = diverted & reached
    delay = delay.where(~div_ok, frame["DivArrDelay"].astype("float64"))

    arrived = ~cancelled & ~(diverted & ~reached) & delay.notna()

    # 결과를 알 수 없는 행: 결항·회항이 아닌데 지연이 비어 있다
    unknown = ~cancelled & ~diverted & delay.isna()
    # 회항했는데 도착 여부 자체가 비어 있는 행도 판단할 수 없다
    unknown |= diverted & frame["DivReachedDest"].isna()
    # 회항 후 목적지에 도착했는데 최종 지연이 비어 있으면 구간을 정할 수 없다.
    # 이 편을 미도착으로 넣으면 도착한 편을 못 왔다고 세는 것이 된다.
    unknown |= div_ok & delay.isna()

    missing = frame[REQUIRED].isna().any(axis=1)
    drop = unknown | missing

    # --- 10구간 ----------------------------------------------------------- #
    y = np.full(len(frame), config.NOARRIVE, dtype="int8")   # 기본값 = 미도착
    d = delay.to_numpy(dtype="float64")
    ok = arrived.to_numpy()
    # np.searchsorted: d <= 0 → 0, 0 < d <= 15 → 1, ... d > 300 → 8
    y[ok] = np.searchsorted(config.EDGES, d[ok], side="left").astype("int8")

    out = frame.loc[~drop].copy()
    out["arr_delay_min"] = np.where(arrived, d, np.nan)[~drop.to_numpy()]
    out["y_class"] = y[~drop.to_numpy()]
    out["crs_dep_hour"] = to_hour(out["CRSDepTime"])
    out["crs_arr_hour"] = to_hour(out["CRSArrTime"])
    for cause, column in config.CAUSE_COLUMN.items():
        out[f"min_{cause}"] = out[column].fillna(0.0).astype("float32")

    stats = {
        "원본": n_start,
        "제외_결과불명": int(unknown.sum()),
        "제외_예약정보결측": int((missing & ~unknown).sum()),
        "남은편수": len(out),
        "미도착": int((out["y_class"] == config.NOARRIVE).sum()),
    }
    return out[OUT], stats


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()

    if not config.RAW_DIR.exists():
        print(f"원자료 폴더가 없습니다: {config.RAW_DIR}", file=sys.stderr)
        print("환경변수 FINAL_RAW_DIR 로 경로를 지정하세요.", file=sys.stderr)
        return 1

    rows, counts = [], np.zeros(config.N_CLASS, dtype="int64")
    for year in config.YEARS:
        for month in config.MONTHS:
            path = config.RAW_DIR / f"{year}_{month:02d}.parquet"
            if not path.exists():
                print(f"없음: {path}", file=sys.stderr)
                return 1
            frame = pd.read_parquet(path, columns=READ)
            out, stats = label(frame)
            out.to_parquet(config.LABEL_DIR / f"{year}_{month:02d}.parquet",
                           index=False)
            counts += np.bincount(out["y_class"], minlength=config.N_CLASS)
            rows.append({"year": year, "month": month, **stats})
            print(f"  {year}-{month:02d}  {stats['남은편수']:>8,} 편  "
                  f"(제외 {stats['제외_결과불명'] + stats['제외_예약정보결측']:,})",
                  flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "step01_rows.csv",
                 index=False, encoding="utf-8-sig")

    dist = pd.DataFrame({
        "class": range(config.N_CLASS),
        "이름": config.CLASS_NAMES,
        "편수": counts,
        "비율": (counts / counts.sum()).round(6),
    })
    dist.to_csv(config.RESULTS_DIR / "step01_class_distribution.csv",
                index=False, encoding="utf-8-sig")

    print(f"\n총 {table['남은편수'].sum():,} 편 "
          f"(원본 {table['원본'].sum():,}, "
          f"제외 {table['원본'].sum() - table['남은편수'].sum():,})")
    print("\n=== 10구간 분포 ===")
    print(dist.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
