"""
Step 3 - three more features a passenger could look up before booking.

Reads  data/clean/{year}_{mm}.parquet   (step 2)
Writes data/schedule/{year}_{mm}.parquet   one row per flight, same order
       results/step03_schedule_summary.csv

The nine schedule features say what the ticket says. These three say things
the ticket does NOT say but that anyone could work out at booking time, from
the published timetable and from last year's public record.

Each one is here because it carries information the other nine cannot
reconstruct. That test matters: a feature that is a function of columns the
model already has adds nothing, however well it correlates with the target.

--------------------------------------------------------------------------
1. sched_pad   = CRSElapsedTime − (last year's median ACTUAL block time
                                   on this route)

   How much slack the airline wrote into the timetable, in minutes.

   Why it cannot be reconstructed: the model knows the scheduled block time
   and the route, but not how long that route actually takes. That gap is
   the new information.

   Why the mechanism is direct rather than incidental: our target is
   measured AGAINST the scheduled arrival, and

       scheduled arrival = scheduled departure + scheduled block time

   so the padding moves the very baseline the delay is measured from. A
   generously padded flight can fly exactly as long as a tight one and still
   land "on time".

   ⚠️ The sign is not decidable in advance. Airlines pad the routes that are
   already unreliable, so a large pad may mean "lots of slack" or "a bad
   route". The report takes the sign from the measurement, not from the
   story.

   ⚠️ ActualElapsedTime is a post-hoc field, so it is used from year Y−1
   ONLY and only as a route-level median — never the flight's own value.
   Cancelled flights have no actual time and are excluded.
   Routes absent from last year fall back: route → airport pair → global,
   recorded in pad_level.

--------------------------------------------------------------------------
2. dest_arr_load = how many flights are scheduled to ARRIVE at this
                   destination, in this hour, on this date

   Why it cannot be reconstructed: the model has Dest and the scheduled
   arrival hour, so it can learn "ORD at 18:00 is busy on average". What it
   cannot see is that THIS particular date has 70 arrivals in that hour
   while the same hour a week earlier had 40.

   Why not the origin side as well: departure congestion is already strongly
   carried by the departure hour (step 8 shows the shape over the day). The
   destination is the side the passenger's own departure time cannot speak
   for - a flight can leave a quiet airport into a jammed one.

   Known at booking: the timetable is published in advance. No outcome
   field is touched.

--------------------------------------------------------------------------
3. days_to_nearest_holiday = signed days to the nearest US federal holiday,
                             clipped to ±21

   Why it cannot be reconstructed: the model has Month and DayOfWeek. The
   4th of July falls on a different weekday every year, so no combination of
   month and weekday can locate it.

   Why signed and continuous rather than a holiday flag: only two federal
   holidays fall inside June–August (Juneteenth, Independence Day), which
   over three training years is six events. A binary flag would be a coin
   tossed six times, and would mostly learn what the weather happened to be
   on those particular days. A signed distance takes a value on every day of
   the season, so it is estimated from the whole sample, and it can express
   "the days before a holiday differ from the days after".

   Memorial Day and Labor Day sit just outside the window and are included
   as anchors, because the start of June and the end of August are measured
   against them.

Usage:  python src/step03_schedule_features.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config

YEARS = list(range(2022, 2027))      # target years; sched_pad comes from Y−1
CLIP_DAYS = 21

READ = ["Year", "Month", "FlightDate", "Reporting_Airline", "Origin", "Dest",
        "crs_arr_hour", "CRSElapsedTime", "route_key"]
READ_PREV = ["Reporting_Airline", "Origin", "Dest", "route_key",
             "CRSElapsedTime", "ActualElapsedTime"]

OUT_COLUMNS = ["sched_pad", "pad_level", "dest_arr_load",
               "days_to_nearest_holiday"]


class Stop(Exception):
    """Fatal data condition: report and stop."""


# --------------------------------------------------------------------------- #
# holidays
# --------------------------------------------------------------------------- #


def federal_holidays(year: int) -> list[pd.Timestamp]:
    """The four US federal holidays that bracket or fall inside the summer."""
    may = pd.date_range(f"{year}-05-01", f"{year}-05-31")
    memorial = may[may.dayofweek == 0][-1]            # last Monday of May
    september = pd.date_range(f"{year}-09-01", f"{year}-09-30")
    labor = september[september.dayofweek == 0][0]    # first Monday of September
    return [memorial,
            pd.Timestamp(f"{year}-06-19"),            # Juneteenth
            pd.Timestamp(f"{year}-07-04"),            # Independence Day
            labor]


def days_to_holiday(dates: pd.Series, year: int) -> np.ndarray:
    """Signed days to the nearest holiday (negative = before), clipped."""
    anchors = np.array([d.to_datetime64() for d in federal_holidays(year)])
    values = dates.to_numpy(dtype="datetime64[D]").astype("datetime64[D]")
    diff = (values[:, None] - anchors[None, :].astype("datetime64[D]")
            ).astype("timedelta64[D]").astype("int32")
    nearest = diff[np.arange(len(diff)), np.abs(diff).argmin(axis=1)]
    return np.clip(nearest, -CLIP_DAYS, CLIP_DAYS).astype("int16")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_year(year: int, columns: list[str]) -> pd.DataFrame:
    parts = []
    for month in config.MONTHS:
        path = config.CLEAN_DIR / f"{year}_{month:02d}.parquet"
        if not path.exists():
            raise Stop(f"Missing step 2 output: {path}")
        parts.append(pd.read_parquet(path, columns=columns))
    return pd.concat(parts, ignore_index=True)


def block_time_table(year: int) -> tuple[pd.Series, pd.Series, float]:
    """Median ACTUAL block time last year, by route and by airport pair."""
    prev = load_year(year, READ_PREV)
    flown = prev[prev["ActualElapsedTime"].notna()].copy()
    if flown.empty:
        raise Stop(f"{year}: no completed flights to measure block time")
    flown["pair"] = flown["Origin"].astype(str) + "_" + flown["Dest"].astype(str)

    by_route = flown.groupby("route_key", observed=True)["ActualElapsedTime"].median()
    by_pair = flown.groupby("pair", observed=True)["ActualElapsedTime"].median()
    global_pad = float((flown["CRSElapsedTime"] - flown["ActualElapsedTime"]).median())
    return by_route, by_pair, global_pad


# --------------------------------------------------------------------------- #


def build_year(year: int) -> dict:
    frame = load_year(year, READ)
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"])

    # --- 1. schedule padding, from year Y−1 only --------------------------- #
    by_route, by_pair, global_pad = block_time_table(year - 1)
    pair = frame["Origin"].astype(str) + "_" + frame["Dest"].astype(str)

    actual = frame["route_key"].map(by_route)
    level = pd.Series(0, index=frame.index, dtype="int8").where(
        actual.notna(), other=1)
    fill_pair = pair.map(by_pair)
    actual = actual.fillna(fill_pair)
    level = level.where(actual.notna(), other=2)

    crs = frame["CRSElapsedTime"].astype("float64")
    pad = (crs - actual.astype("float64"))
    pad = pad.fillna(global_pad)                      # level 2: global median pad
    frame["sched_pad"] = pad.astype("float32")
    frame["pad_level"] = level.astype("int8")

    # --- 2. destination arrival load, from this year's own timetable ------- #
    load = (frame.groupby(["Dest", "FlightDate", "crs_arr_hour"], observed=True)
            ["Dest"].transform("size"))
    frame["dest_arr_load"] = load.astype("int16")

    # --- 3. signed distance to the nearest federal holiday ----------------- #
    frame["days_to_nearest_holiday"] = days_to_holiday(frame["FlightDate"], year)

    # --- write, month by month, in the same row order as data/clean/ ------- #
    config.SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    out = frame[OUT_COLUMNS]
    start = 0
    for month in config.MONTHS:
        length = int((frame["FlightDate"].dt.month == month).sum())
        out.iloc[start:start + length].to_parquet(
            config.SCHEDULE_DIR / f"{year}_{month:02d}.parquet", index=False)
        start += length
    if start != len(out):
        raise Stop(f"{year}: month split did not cover every row")

    summary = {
        "year": year, "rows": len(frame),
        "pad_mean": round(float(frame["sched_pad"].mean()), 2),
        "pad_sd": round(float(frame["sched_pad"].std()), 2),
        "pad_level0_route": round(float((frame["pad_level"] == 0).mean()), 4),
        "pad_level1_pair": round(float((frame["pad_level"] == 1).mean()), 4),
        "pad_level2_global": round(float((frame["pad_level"] == 2).mean()), 4),
        "load_mean": round(float(frame["dest_arr_load"].mean()), 2),
        "load_p95": int(frame["dest_arr_load"].quantile(0.95)),
        "load_max": int(frame["dest_arr_load"].max()),
        "holiday_abs_mean": round(float(
            frame["days_to_nearest_holiday"].abs().mean()), 2),
        "at_holiday_share": round(float(
            (frame["days_to_nearest_holiday"] == 0).mean()), 4),
        "nulls": int(frame[OUT_COLUMNS].isna().sum().sum()),
    }
    print(f"[{year}] pad 평균 {summary['pad_mean']:+.1f}분 (sd {summary['pad_sd']:.1f}, "
          f"노선일치 {summary['pad_level0_route']:.1%})  "
          f"도착혼잡 평균 {summary['load_mean']:.1f}편/시 (최대 {summary['load_max']})  "
          f"공휴일 당일 {summary['at_holiday_share']:.2%}  nulls={summary['nulls']}")
    return summary


def main() -> int:
    config.ensure_dirs()
    try:
        summaries = [build_year(y) for y in YEARS]
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    table = pd.DataFrame(summaries)
    out = config.RESULTS_DIR / "step03_schedule_summary.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n=== Step 3 summary ===")
    print(table.to_string(index=False))
    print("\nsched_pad 은 Y−1 년의 노선 중앙값만 쓴다 — 그 편 자신의 실제 소요시간은")
    print("들어가지 않는다. dest_arr_load 와 days_to_nearest_holiday 는 결과 컬럼을")
    print("전혀 건드리지 않는다(시간표와 달력뿐).")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
