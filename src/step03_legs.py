"""
Step 3 - linking each flight to the previous leg of the same aircraft.
(PROPOSAL_FLOW §4, 3단계)

Reads  data/clean/{year}_{mm}.parquet   (2021-2025; 2026 never needs legs
       because 2026's report card is built from 2025)
Writes data/legs/{year}_{mm}.parquet
       results/step03_legs_summary.csv

An aircraft flies several times a day, and a late inbound leg makes the next
departure late before it has even begun. The passenger's screen shows only
"08:25 departure" and none of that, so the chain has to be rebuilt.

A link is accepted only when ALL FOUR hold:
  1. same Tail_Number
  2. same FlightDate      (an overnight gap means the aircraft rested)
  3. previous destination == this origin   (otherwise a leg is missing)
  4. the previous leg was not cancelled    (a cancelled leg has no arrival)

Produced per flight
-------------------
  day_seq          this flight's order within the aircraft's day (1 = first)
  is_first_flight  1 if day_seq == 1
  turnaround_min   SCHEDULED departure - SCHEDULED previous arrival (0-300)
                   Scheduled, not actual: at booking time the timetable is all
                   anyone has, and what we want to measure is the slack the
                   airline designed in.
  prev_delayed     previous leg arrived >= 15 minutes late
  absorbed         1 if the previous leg was late and this one still left on
                   time; NA when the previous leg was NOT late.
                   NA, not 0: with no delay to absorb there is nothing to
                   judge, and filling 0 would count "did not sit the exam" as
                   "failed the exam".

Tail_Number is used ONLY here, and only on past years, where the assignment is
already history. It is never a model input - at booking time nobody knows
which airframe will operate the flight.

Usage:  python src/step03_legs.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CLEAN_DIR = config.CLEAN_DIR
LEGS_DIR = config.LEGS_DIR
SUMMARY_PATH = config.RESULTS_DIR / "step03_legs_summary.csv"

YEARS = range(2021, 2026)          # 2021-2025; 2026 needs no legs
MONTHS = config.MONTHS

DELAY_THRESHOLD = 15               # minutes, same threshold as the target
TURNAROUND_MIN = 0                 # accepted scheduled turnaround range
TURNAROUND_MAX = 300

NEEDED_COLUMNS = [
    "Month",
    "FlightDate",
    "Tail_Number",
    "Origin",
    "Dest",
    "crs_dep_min",
    "crs_arr_min",
    "ArrDelay",
    "DepDelay",
    "Cancelled",
    "Year",
    "flight_key",
    "route_key",
    "group_key",
]

OUT_COLUMNS = [
    "Year",
    "Month",
    "FlightDate",
    "flight_key",
    "route_key",
    "group_key",
    "Tail_Number",
    "Origin",
    "Dest",
    "crs_dep_min",
    "crs_arr_min",
    "day_seq",
    "is_first_flight",
    "linked",
    "turnaround_min",
    "prev_arr_delay",
    "dep_delay",
    "prev_delayed",
    "absorbed",
]


class Stop(Exception):
    """Fatal condition: stop the whole run and report."""


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


def build_legs(year: int, month: int) -> dict:
    """Build the legs table for one month and write it. Returns a summary dict."""
    tag = "{}-{:02d}".format(year, month)
    clean_path = CLEAN_DIR / "{}_{:02d}.parquet".format(year, month)
    out_path = LEGS_DIR / "{}_{:02d}.parquet".format(year, month)

    if not clean_path.exists():
        raise Stop("Missing step 2 output: {}".format(clean_path))

    frame = pd.read_parquet(clean_path, columns=NEEDED_COLUMNS)
    rows_in = len(frame)

    # 1. drop rows without a Tail_Number -------------------------------------
    tail = frame["Tail_Number"].astype("string").str.strip()
    frame = frame.assign(Tail_Number=tail)
    frame = frame[frame["Tail_Number"].notna() & (frame["Tail_Number"] != "")]
    rows_tail = len(frame)

    # 2. sort by aircraft, date, scheduled departure, then look one row back --
    # crs_dep_min is used instead of raw hhmm so that the 2400 -> 0000
    # conversion from step 2 is respected.
    frame = frame.sort_values(
        ["Tail_Number", "FlightDate", "crs_dep_min"], kind="mergesort"
    ).reset_index(drop=True)

    prev_tail = frame["Tail_Number"].shift(1)
    prev_date = frame["FlightDate"].shift(1)
    prev_dest = frame["Dest"].shift(1)
    prev_arr_min = frame["crs_arr_min"].shift(1)
    prev_arr_delay = frame["ArrDelay"].shift(1)
    prev_cancelled = frame["Cancelled"].shift(1)

    # 3. a link counts only if same aircraft, same day, airports chain, and
    #    the previous leg was not cancelled
    same_tail = prev_tail == frame["Tail_Number"]
    same_day = prev_date == frame["FlightDate"]
    chains = prev_dest == frame["Origin"]
    prev_ok = prev_cancelled.fillna(1) != 1

    linked = same_tail & same_day & chains & prev_ok

    # 4. scheduled turnaround, accepted only within range ---------------------
    turnaround = (frame["crs_dep_min"] - prev_arr_min).astype("Float64")
    in_range = (turnaround >= TURNAROUND_MIN) & (turnaround <= TURNAROUND_MAX)
    turnaround_ok = linked & in_range.fillna(False)
    frame["turnaround_min"] = turnaround.where(turnaround_ok)

    # 5. same-day flight sequence -------------------------------------------
    frame["day_seq"] = (
        frame.groupby(["Tail_Number", "FlightDate"]).cumcount() + 1
    ).astype("int16")
    frame["is_first_flight"] = (frame["day_seq"] == 1).astype("int8")

    # 6. absorption: judged only when the previous leg arrived >= 15 late ----
    frame["linked"] = linked.astype("int8")
    frame["prev_arr_delay"] = prev_arr_delay.astype("float64").where(linked)
    frame["dep_delay"] = frame["DepDelay"].astype("float64")

    prev_delayed = linked & (prev_arr_delay >= DELAY_THRESHOLD).fillna(False)
    frame["prev_delayed"] = prev_delayed.astype("int8")

    judgeable = prev_delayed & frame["dep_delay"].notna()
    frame["absorbed"] = (
        pd.Series(frame["dep_delay"] < DELAY_THRESHOLD, index=frame.index)
        .astype("Int8")
        .where(judgeable)
    )

    # --- write -------------------------------------------------------------- #
    out = frame[OUT_COLUMNS]
    LEGS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, engine="pyarrow", index=False)

    n_linked = int(out["linked"].sum())
    n_turn = int(out["turnaround_min"].notna().sum())
    n_prev_delayed = int(out["prev_delayed"].sum())
    n_judgeable = int(out["absorbed"].notna().sum())
    n_absorbed = int((out["absorbed"] == 1).sum())

    summary = {
        "year": year,
        "month": month,
        "rows_clean": rows_in,
        "rows_with_tail": rows_tail,
        "tail_missing": rows_in - rows_tail,
        "linked": n_linked,
        "link_rate": round(n_linked / rows_tail, 6),
        "turnaround_valid": n_turn,
        "turnaround_rate_of_linked": round(n_turn / n_linked, 6) if n_linked else 0.0,
        "turnaround_mean": round(float(out["turnaround_min"].mean()), 2),
        "turnaround_median": float(out["turnaround_min"].median()),
        "first_flight_rate": round(float(out["is_first_flight"].mean()), 6),
        "day_seq_mean": round(float(out["day_seq"].mean()), 3),
        "day_seq_max": int(out["day_seq"].max()),
        "prev_delayed": n_prev_delayed,
        "absorb_judgeable": n_judgeable,
        "absorbed": n_absorbed,
        "absorb_rate": round(n_absorbed / n_judgeable, 6) if n_judgeable else 0.0,
        "flight_keys": int(out["flight_key"].nunique()),
    }

    print(
        "[{}] tail-rows={:,} ({} missing)  linked={:,} ({:.1%})  "
        "turnaround ok={:,}  prev_delayed={:,}  absorbed={:,}/{:,} ({:.1%})".format(
            tag,
            rows_tail,
            summary["tail_missing"],
            n_linked,
            summary["link_rate"],
            n_turn,
            n_prev_delayed,
            n_absorbed,
            n_judgeable,
            summary["absorb_rate"],
        )
    )
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    config.ensure_dirs()
    summaries = []

    try:
        for year in YEARS:
            for month in MONTHS:
                summaries.append(build_legs(year, month))
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    table = pd.DataFrame(summaries)
    table.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")

    print("\n=== Summary ===")
    cols = [
        "year",
        "month",
        "rows_with_tail",
        "link_rate",
        "turnaround_median",
        "first_flight_rate",
        "prev_delayed",
        "absorb_rate",
    ]
    print(table[cols].to_string(index=False))

    print("\nBy year:")
    by_year = table.groupby("year").apply(
        lambda g: pd.Series(
            {
                "rows_with_tail": int(g["rows_with_tail"].sum()),
                "tail_missing": int(g["tail_missing"].sum()),
                "link_rate": g["linked"].sum() / g["rows_with_tail"].sum(),
                "prev_delayed": int(g["prev_delayed"].sum()),
                "absorb_rate": g["absorbed"].sum() / g["absorb_judgeable"].sum(),
            }
        ),
        include_groups=False,
    )
    print(by_year.to_string())
    print("\nsummary: {}".format(SUMMARY_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
