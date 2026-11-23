"""
Step 2 - cleaning, clock conversion, keys, and the ten-class target.

Reads  data/raw/{year}_{mm}.parquet        (step 1)
Writes data/clean/{year}_{mm}.parquet
       results/step02_clean_summary.csv

What this step makes
--------------------
clock -> number   crs_dep_min, crs_arr_min, crs_dep_hour, crs_arr_hour
                  1435 is not a number: 1359 and 1400 are one minute apart but
                  41 apart as integers. Minutes-from-midnight fixes that.

three keys        flight_key = airline + flight no. + origin + dest  (narrow)
                  route_key  = airline + origin + dest               (middle)
                  group_key  = airline + origin + scheduled dep hour (wide)
                  Airlines reassign flight numbers to different routes
                  between years, so the narrow key alone does not identify a
                  route. route_key is what step 3 measures block time on.

the target        y_class, 0..9 over the bins in config.EDGES, plus class 9
                  "never arrived" for a cancellation or a diversion that did
                  not reach the destination.

                  A flight that was diverted but DID reach its destination is
                  not a different kind of event to a passenger - it is a very
                  late arrival - so DivArrDelay puts it in a delay bin.

Outliers are NOT removed
------------------------
The extreme tail (300+ minutes, never arrived) is the product. A "90%
guaranteed arrival time" is read off the tail of the distribution, so trimming
the tail would make every guarantee optimistic and break the promise.

Usage:  python src/step02_clean.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config


class Stop(Exception):
    """Fatal data condition: report and stop rather than guess."""


# columns carried through from the raw file
CARRY = (
    ["Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate"]
    + ["Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline"]
    + ["Origin", "OriginCityName", "OriginState", "Dest", "DestCityName", "DestState"]
    + ["CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance"]
    + config.B_OUTCOME
    + config.C_POSTHOC
)

DERIVED = [
    "crs_dep_min", "crs_arr_min", "crs_dep_hour", "crs_arr_hour",
    "flight_key", "route_key", "group_key",
    "arr_delay_min", "y_class",
]


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """hhmm (1435) -> minutes from midnight (875). 2400 means 0000."""
    values = pd.to_numeric(series, errors="coerce").astype("Int32")
    values = values.where(values != 2400, 0)
    hours, minutes = values // 100, values % 100
    bad = ((hours > 23) | (minutes > 59)) & values.notna()
    if bool(bad.any()):
        raise Stop(
            f"{int(bad.sum())} scheduled times outside hhmm range, "
            f"e.g. {series[bad].head(5).tolist()}"
        )
    return (hours * 60 + minutes).astype("Int32")


def build_target(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Ten-class arrival outcome and the delay in minutes (NaN if never arrived)."""
    cancelled = frame["Cancelled"].fillna(0).to_numpy() == 1
    diverted = frame["Diverted"].fillna(0).to_numpy() == 1
    reached = frame["DivReachedDest"].fillna(0).to_numpy() == 1

    delay = frame["ArrDelay"].to_numpy(dtype="float64").copy()
    div_delay = frame["DivArrDelay"].to_numpy(dtype="float64")

    # diverted but arrived -> a very late arrival, with its own delay figure
    div_ok = diverted & reached
    delay[div_ok] = div_delay[div_ok]

    never = cancelled | (diverted & ~reached)
    delay[never] = np.nan

    y = np.digitize(delay, config.EDGES, right=True).astype("int16")
    y[never] = config.NOARRIVE
    return y, delay


def clean_month(year: int, month: int) -> dict:
    tag = f"{year}-{month:02d}"
    raw_path = config.RAW_DIR / f"{year}_{month:02d}.parquet"
    out_path = config.CLEAN_DIR / f"{year}_{month:02d}.parquet"
    if not raw_path.exists():
        raise Stop(f"Missing step 1 output: {raw_path}")

    frame = pd.read_parquet(raw_path)
    missing = [c for c in CARRY if c not in frame.columns]
    if missing:
        raise Stop(f"{raw_path.name} is missing columns: {missing}")
    frame = frame[CARRY].copy()
    rows_in = len(frame)

    # --- clock -> number --------------------------------------------------- #
    dep_2400 = int((pd.to_numeric(frame["CRSDepTime"], errors="coerce") == 2400).sum())
    arr_2400 = int((pd.to_numeric(frame["CRSArrTime"], errors="coerce") == 2400).sum())
    frame["crs_dep_min"] = hhmm_to_minutes(frame["CRSDepTime"])
    frame["crs_arr_min"] = hhmm_to_minutes(frame["CRSArrTime"])
    frame["crs_dep_hour"] = (frame["crs_dep_min"] // 60).astype("Int32")
    frame["crs_arr_hour"] = (frame["crs_arr_min"] // 60).astype("Int32")

    # --- keys -------------------------------------------------------------- #
    airline = frame["Reporting_Airline"].astype("string").str.strip()
    flight_no = frame["Flight_Number_Reporting_Airline"].astype("string").str.strip()
    origin = frame["Origin"].astype("string").str.strip()
    dest = frame["Dest"].astype("string").str.strip()
    dep_hour = frame["crs_dep_hour"].astype("string").str.zfill(2)

    frame["flight_key"] = (airline + "_" + flight_no + "_" + origin + "_" + dest)
    frame["route_key"] = (airline + "_" + origin + "_" + dest)
    frame["group_key"] = (airline + "_" + origin + "_" + dep_hour)
    key_null = int(sum(frame[k].isna().sum() for k in
                       ("flight_key", "route_key", "group_key")))

    # --- target ------------------------------------------------------------ #
    y, delay = build_target(frame)
    frame["y_class"] = y
    frame["arr_delay_min"] = delay

    # --- checks, reported rather than silently repaired --------------------- #
    cancelled = int((frame["Cancelled"].fillna(0) == 1).sum())
    diverted = int((frame["Diverted"].fillna(0) == 1).sum())
    div_reached = int(((frame["Diverted"].fillna(0) == 1)
                       & (frame["DivReachedDest"].fillna(0) == 1)).sum())
    # ArrDelay should be null only for a cancelled or diverted flight
    unexplained = int((frame["ArrDelay"].isna()
                       & (frame["Cancelled"].fillna(0) == 0)
                       & (frame["Diverted"].fillna(0) == 0)).sum())
    if (frame["y_class"] < 0).any() or (frame["y_class"] >= config.N_CLASS).any():
        raise Stop(f"{tag}: y_class outside 0..{config.N_CLASS - 1}")

    frame = frame[CARRY + DERIVED]
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, engine="pyarrow", index=False)

    counts = frame["y_class"].value_counts().reindex(range(config.N_CLASS), fill_value=0)
    summary = {
        "year": year, "month": month, "rows": rows_in,
        "cancelled": cancelled, "diverted": diverted, "div_reached": div_reached,
        "arrdelay_null_unexplained": unexplained,
        "crs_dep_2400": dep_2400, "crs_arr_2400": arr_2400,
        "crs_dep_min_null": int(frame["crs_dep_min"].isna().sum()),
        "key_null": key_null,
        "flight_keys": int(frame["flight_key"].nunique()),
        "route_keys": int(frame["route_key"].nunique()),
        "group_keys": int(frame["group_key"].nunique()),
    }
    for i in range(config.N_CLASS):
        summary[f"c{i}"] = int(counts.iloc[i])

    print(
        f"[{tag}] rows={rows_in:,}  on-time(c0)={counts.iloc[0] / rows_in:.1%}  "
        f"never-arrived(c9)={counts.iloc[9] / rows_in:.2%}  "
        f"cancelled={cancelled:,}  diverted={diverted:,} (reached {div_reached:,})  "
        f"unexplained ArrDelay nulls={unexplained}"
    )
    return summary


def main() -> int:
    config.ensure_dirs()
    summaries = []
    try:
        for year in config.YEARS:
            for month in config.MONTHS:
                summaries.append(clean_month(year, month))
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    table = pd.DataFrame(summaries)
    out = config.RESULTS_DIR / "step02_clean_summary.csv"
    table.to_csv(out, index=False, encoding="utf-8")

    total = int(table["rows"].sum())
    print(f"\n=== Step 2 summary ===\nTotal rows: {total:,}")

    share = pd.DataFrame({
        "class": range(config.N_CLASS),
        "name": config.CLASS_NAMES,
        "count": [int(table[f"c{i}"].sum()) for i in range(config.N_CLASS)],
    })
    share["share"] = share["count"] / total
    print(share.to_string(index=False, formatters={"share": "{:.4f}".format}))

    print("\nBy year (share of class 0, 'on time or early'):")
    by_year = table.groupby("year").apply(
        lambda g: pd.Series({"rows": int(g["rows"].sum()),
                             "c0_share": g["c0"].sum() / g["rows"].sum(),
                             "c9_share": g["c9"].sum() / g["rows"].sum()}),
        include_groups=False)
    print(by_year.to_string())

    print(f"\nUnexplained ArrDelay nulls: {int(table['arrdelay_null_unexplained'].sum()):,}")
    print(f"Null keys: {int(table['key_null'].sum()):,}")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
