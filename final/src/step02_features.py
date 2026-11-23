"""step02 — 파생 변수 2개를 만든다.

읽기  data/label/{year}_{mm}.parquet
쓰기  data/feature/{year}_{mm}.parquet      step01 의 열 + 파생 2열
      results/step02_holidays.csv           쓴 공휴일 목록
      results/step02_feature_summary.csv    두 변수의 요약

두 변수 모두 예약 시점에 알 수 있는 값이다
-------------------------------------------
1. dest_arr_load — 그 날짜·그 시각에 그 도착지로 들어오도록 **예정된** 편 수.
   올해 공개 시간표만으로 세므로 예약하는 날에 계산할 수 있다.

   왜 필요한가: 모델은 Dest 와 예정 도착 시각을 이미 받으므로 "ORD 18시는
   평균적으로 붐빈다" 는 배울 수 있다. 하지만 **이 날짜의 그 시각**에 70편이
   들어오는지 40편이 들어오는지는 볼 수 없다. 그 차이를 넣는다.

   왜 출발지 쪽은 넣지 않는가: 출발 혼잡은 예정 출발 시각이 이미 상당 부분
   담고 있고(시각별 정시율의 모양), 우리가 맞히는 것은 도착 지연이다.

2. days_to_nearest_holiday — 가장 가까운 연방 공휴일까지의 부호 있는 일수.
   음수는 공휴일 전, 양수는 후. ±21 일에서 자른다. 한 달 넘게 떨어진 날은
   공휴일과 무관하므로 더 멀어도 같은 값으로 둔다.

   공휴일은 날짜 규칙으로 계산한다(메모리얼데이 = 5월 마지막 월요일, 노동절 =
   9월 첫 월요일, 준틴스 6/19, 독립기념일 7/4). 대체공휴일은 쓰지 않고 실제
   날짜를 쓴다.

실행  python src/step02_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# 여름(6~8월) 기준 ±21일 안에 들어올 수 있는 연방 공휴일만 있으면 되지만,
# 범위를 넓게 잡아도 결과는 같으므로 5~9월을 모두 계산한다.
FIXED = [(6, 19), (7, 4)]        # 준틴스, 독립기념일


def nth_weekday(year: int, month: int, weekday: int, nth: int) -> pd.Timestamp:
    """그 달의 n번째 특정 요일. nth=-1 이면 마지막."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[days.month == month]
    matches = days[days.weekday == weekday]
    return matches[nth if nth >= 0 else len(matches) + nth]


def holidays(years) -> pd.DatetimeIndex:
    """연방 공휴일 — 여름 범위에 영향을 주는 것들."""
    out = []
    for year in years:
        out.append(nth_weekday(year, 5, 0, -1))      # 메모리얼데이: 5월 마지막 월요일
        out.append(nth_weekday(year, 9, 0, 0))       # 노동절: 9월 첫 월요일
        for month, day in FIXED:
            out.append(pd.Timestamp(year=year, month=month, day=day))
    return pd.DatetimeIndex(sorted(out))


def days_to_holiday(dates: pd.Series, table: pd.DatetimeIndex) -> np.ndarray:
    """가장 가까운 공휴일까지의 부호 있는 일수 (음수 = 공휴일 전)."""
    day = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    ref = table.to_numpy(dtype="datetime64[D]").astype("int64")
    diff = day[:, None] - ref[None, :]                     # 음수면 공휴일 전
    nearest = diff[np.arange(len(day)), np.abs(diff).argmin(axis=1)]
    return np.clip(nearest, -config.HOLIDAY_CLIP_DAYS,
                   config.HOLIDAY_CLIP_DAYS).astype("int16")


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()

    table = holidays(range(min(config.YEARS) - 1, max(config.YEARS) + 2))
    pd.DataFrame({"date": table.strftime("%Y-%m-%d"),
                  "요일": table.day_name()}).to_csv(
        config.RESULTS_DIR / "step02_holidays.csv",
        index=False, encoding="utf-8-sig")

    rows = []
    for year in config.YEARS:
        for month in config.MONTHS:
            path = config.LABEL_DIR / f"{year}_{month:02d}.parquet"
            frame = pd.read_parquet(path)

            # --- 1. 도착지 시간대 혼잡 (올해 시간표만으로) ----------------- #
            frame["dest_arr_load"] = (
                frame.groupby(["Dest", "FlightDate", "crs_arr_hour"],
                              observed=True)["Dest"]
                .transform("size").astype("int16"))

            # --- 2. 공휴일까지 남은 일수 ---------------------------------- #
            frame["days_to_nearest_holiday"] = days_to_holiday(
                frame["FlightDate"], table)

            frame.to_parquet(
                config.FEATURE_DIR / f"{year}_{month:02d}.parquet", index=False)

            rows.append({
                "year": year, "month": month, "편수": len(frame),
                "load_평균": round(float(frame["dest_arr_load"].mean()), 2),
                "load_최대": int(frame["dest_arr_load"].max()),
                "holiday_최소": int(frame["days_to_nearest_holiday"].min()),
                "holiday_최대": int(frame["days_to_nearest_holiday"].max()),
            })
            print(f"  {year}-{month:02d}  완료", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "step02_feature_summary.csv",
                   index=False, encoding="utf-8-sig")
    print()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
