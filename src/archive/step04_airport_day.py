"""
Step 4 - the state of each origin airport on each day.
(PROPOSAL_FLOW §4, 4단계)

Reads  data/clean/{year}_{mm}.parquet   (2021-2026)
Writes data/airport_day/airport_day.parquet   one row per (Origin, FlightDate)
       results/step04_airport_day_summary.csv

Step 3 captured one aircraft's troubles. Some days the whole airport goes
down - a storm, a ground stop - and then EVERY flight is late. That is a
different thing and it needs its own measurement.

Per airport-day
---------------
  n_flights        flights scheduled out of that airport that day
  mean_dep_delay   mean DepDelay over NON-CANCELLED flights
                   (a flight that never left has no departure delay)
  fail_rate        share of flights that failed, over ALL flights
                   (the target counts a cancellation as a failure, so the
                   denominator has to include them)
                   failure := y_class >= 2, i.e. arrived 15+ minutes late or
                   never arrived - the same 15-minute line the industry uses
                   and the same one the bin edges already draw.
  bad_day          1 if mean_dep_delay is in the top 10% of THAT airport's
                   days in THAT year

Why "bad day" is relative, not an absolute number of minutes
------------------------------------------------------------
A 30-minute mean departure delay is the worst day of the year at a quiet
airport and an ordinary Tuesday at a busy one. One fixed cut-off would make
the variable measure "is this a busy airport", which the airport name already
says. The cut is taken per year as well, because the overall delay level moves
from year to year and a pooled cut would pile every bad day into one year.

Note that mean_dep_delay and fail_rate are built from C-group and B-group
fields. They are NOT model inputs: step 5 turns them into a last-year summary,
and only that summary is offered to a model.

Usage:  python src/step04_airport_day.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CLEAN_DIR = config.CLEAN_DIR
OUT_DIR = config.AIRPORT_DAY_DIR
OUT_PATH = OUT_DIR / "airport_day.parquet"
SUMMARY_PATH = config.RESULTS_DIR / "step04_airport_day_summary.csv"

YEARS = config.YEARS
MONTHS = config.MONTHS

BAD_DAY_QUANTILE = config.BAD_DAY_QUANTILE   # top 10% of days, per airport-year

# a flight "failed" if it arrived 15+ minutes late or never arrived
FAIL_FROM_CLASS = 2

NEEDED_COLUMNS = ["Origin", "FlightDate", "Month", "Year", "DepDelay",
                  "Cancelled", "y_class"]


class Stop(Exception):
    """Fatal condition: stop the whole run and report."""


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def load_clean() -> pd.DataFrame:
    """Load the columns needed from every clean month."""
    parts = []
    for year in YEARS:
        for month in MONTHS:
            path = CLEAN_DIR / "{}_{:02d}.parquet".format(year, month)
            if not path.exists():
                raise Stop("Missing step 2 output: {}".format(path))
            parts.append(pd.read_parquet(path, columns=NEEDED_COLUMNS))
    frame = pd.concat(parts, ignore_index=True)
    print("loaded {:,} rows from {} clean files".format(len(frame), len(parts)))
    return frame


def aggregate_airport_day(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (Origin, FlightDate)."""
    cancelled = pd.to_numeric(frame["Cancelled"], errors="coerce").fillna(0) == 1

    # mean DepDelay over non-cancelled flights only
    frame = frame.assign(
        dep_delay_noncx=frame["DepDelay"].where(~cancelled),
        is_cancelled=cancelled.astype("int8"),
        failed=(frame["y_class"] >= FAIL_FROM_CLASS).astype("int8"),
    )

    grouped = frame.groupby(["Origin", "FlightDate"], observed=True)
    table = grouped.agg(
        year=("Year", "first"),
        Month=("Month", "first"),
        n_flights=("failed", "size"),
        n_cancelled=("is_cancelled", "sum"),
        n_dep_delay=("dep_delay_noncx", "count"),
        mean_dep_delay=("dep_delay_noncx", "mean"),
        fail_rate=("failed", "mean"),
    ).reset_index()

    table["year"] = table["year"].astype("int16")
    return table


def mark_bad_days(table: pd.DataFrame) -> pd.DataFrame:
    """Per airport per year, mark the top 10% of days by mean_dep_delay."""
    keys = ["Origin", "year"]
    table["threshold_p90"] = table.groupby(keys, observed=True)[
        "mean_dep_delay"
    ].transform(lambda s: s.quantile(BAD_DAY_QUANTILE))
    table["days_in_year"] = table.groupby(keys, observed=True)[
        "mean_dep_delay"
    ].transform("size")

    table["bad_day"] = (
        (table["mean_dep_delay"] >= table["threshold_p90"])
        & table["mean_dep_delay"].notna()
        & table["threshold_p90"].notna()
    ).astype("int8")
    return table


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    config.ensure_dirs()

    try:
        frame = load_clean()
        table = aggregate_airport_day(frame)
        table = mark_bad_days(table)
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    no_delay_data = int(table["mean_dep_delay"].isna().sum())

    table = table[
        [
            "Origin",
            "FlightDate",
            "year",
            "Month",
            "n_flights",
            "n_cancelled",
            "n_dep_delay",
            "mean_dep_delay",
            "fail_rate",
            "threshold_p90",
            "days_in_year",
            "bad_day",
        ]
    ].sort_values(["Origin", "FlightDate"]).reset_index(drop=True)

    table.to_parquet(OUT_PATH, engine="pyarrow", index=False)

    # ----------------------------------------------------------------- #
    # Summary
    # ----------------------------------------------------------------- #
    by_year = table.groupby("year").apply(
        lambda g: pd.Series(
            {
                "airports": g["Origin"].nunique(),
                "airport_days": len(g),
                "bad_days": int(g["bad_day"].sum()),
                "bad_day_share": g["bad_day"].mean(),
                "mean_dep_delay": g["mean_dep_delay"].mean(),
                "bad_day_mean_delay": g.loc[g["bad_day"] == 1, "mean_dep_delay"].mean(),
                "normal_day_mean_delay": g.loc[
                    g["bad_day"] == 0, "mean_dep_delay"
                ].mean(),
                "bad_day_fail_rate": g.loc[g["bad_day"] == 1, "fail_rate"].mean(),
                "normal_day_fail_rate": g.loc[g["bad_day"] == 0, "fail_rate"].mean(),
            }
        ),
        include_groups=False,
    )
    by_year.to_csv(SUMMARY_PATH, encoding="utf-8")

    print("\n=== airport_day ===")
    print("rows: {:,}   airports: {}   airport-days without DepDelay data: {}".format(
        len(table), table["Origin"].nunique(), no_delay_data
    ))

    print("\nBy year:")
    print(by_year.to_string())

    print("\nBad days per airport-year (distribution):")
    per_airport = (
        table.groupby(["Origin", "year"], observed=True)["bad_day"].sum().astype(int)
    )
    print(per_airport.describe().to_string())

    print("\nDays per airport-year (distribution):")
    days = table.groupby(["Origin", "year"], observed=True).size()
    print(days.describe().to_string())
    print("airport-years with fewer than 30 days: {} of {}".format(
        int((days < 30).sum()), len(days)
    ))

    print("\nbad_day threshold (mean DepDelay, minutes) at selected airports:")
    sample = ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "EWR", "SFO"]
    show = (
        table[table["Origin"].isin(sample)]
        .groupby(["Origin", "year"], observed=True)["threshold_p90"]
        .first()
        .unstack()
    )
    print(show.round(1).to_string())

    print("\noutput: {}".format(OUT_PATH))
    print("summary: {}".format(SUMMARY_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
