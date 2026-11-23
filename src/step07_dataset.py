"""
Step 7 - one table per split, cut by year.

Reads  data/clean/ (step 2), data/schedule/ (step 3), data/scenario/ (step 6)
Writes data/dataset/{train,val,test}.parquet
       results/step07_dataset_summary.csv

    train  2022 2023 2024
    val    2025
    test   2026

The split is by YEAR, not at random
-----------------------------------
The real task is "learn from the past, answer about the future". Shuffling at
random would put two flights from the same airport on the same day into
training and test at once, and that day's weather would be in both. The score
would then be measuring memory of a specific afternoon.

The three sources are already row-aligned: step 2, step 3 and step 6 each
write one row per flight, in the same order, month by month. This step checks
that alignment rather than assuming it, concatenates the columns, and cuts the
years into splits.

Usage:  python src/step07_dataset.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config

TARGET = "y_class"
FEATURE_COLUMNS = config.MODEL_FEATURES          # the twelve model inputs

SCENARIO_COLUMNS = ["scen_origin", "scen_dest",
                    "scen_origin_applied", "scen_dest_applied"]
SCHEDULE_EXTRA = ["pad_level"]       # carried for auditing, not an input

# carried through, never offered to a model
META_COLUMNS = [
    "Year", "FlightDate", "flight_key", "route_key", "group_key",
    "Flight_Number_Reporting_Airline", "OriginCityName", "DestCityName",
    "crs_dep_min", "crs_arr_min", "arr_delay_min",
]

CLEAN_COLUMNS = sorted(set(
    config.BASE_FEATURES
    + ["FlightDate", "Year", "flight_key", "route_key", "group_key",
       "Flight_Number_Reporting_Airline", "OriginCityName", "DestCityName",
       "crs_dep_min", "crs_arr_min", "arr_delay_min", TARGET]
))



class Stop(Exception):
    """Fatal condition: stop and report."""


def load_year(year: int) -> pd.DataFrame:
    clean, sched, scen = [], [], []
    for month in config.MONTHS:
        cpath = config.CLEAN_DIR / f"{year}_{month:02d}.parquet"
        dpath = config.SCHEDULE_DIR / f"{year}_{month:02d}.parquet"
        spath = config.SCENARIO_DIR / f"{year}_{month:02d}.parquet"
        for path, step in ((cpath, 2), (dpath, 3), (spath, 6)):
            if not path.exists():
                raise Stop(f"Missing step {step} output: {path}")
        c = pd.read_parquet(cpath, columns=CLEAN_COLUMNS)
        d = pd.read_parquet(dpath)
        s = pd.read_parquet(spath)
        if not len(c) == len(d) == len(s):
            raise Stop(f"{year}-{month:02d}: clean {len(c):,} / schedule {len(d):,}"
                       f" / scenario {len(s):,} row counts differ")
        clean.append(c); sched.append(d); scen.append(s)
    return pd.concat([pd.concat(clean, ignore_index=True),
                      pd.concat(sched, ignore_index=True),
                      pd.concat(scen, ignore_index=True)], axis=1)


def build_year(year: int) -> tuple[pd.DataFrame, dict]:
    frame = load_year(year)
    rows = len(frame)

    for col in ["Month", "DayOfWeek"]:
        frame[col] = frame[col].astype("int8")
    for col in ["crs_dep_hour", "crs_arr_hour"]:
        if int(frame[col].isna().sum()):
            raise Stop(f"{col} has missing values in {year}")
        frame[col] = frame[col].astype("int8")

    frame = frame[FEATURE_COLUMNS + SCENARIO_COLUMNS + SCHEDULE_EXTRA
                  + [TARGET] + META_COLUMNS]

    info = {
        "year": year, "rows": rows,
        "c0_share": round(float((frame[TARGET] == 0).mean()), 6),
        "c9_share": round(float((frame[TARGET] == 9).mean()), 6),
    }
    print(f"[{year}] rows={rows:,}  정시={info['c0_share']:.2%}  "
          f"미도착={info['c9_share']:.2%}")
    return frame, info


def main() -> int:
    config.ensure_dirs()
    infos, frames = [], {}
    try:
        per_year = {}
        for years in config.SPLIT_YEARS.values():
            for year in years:
                frame, info = build_year(year)
                per_year[year] = frame
                infos.append(info)

        for split, years in config.SPLIT_YEARS.items():
            frames[split] = pd.concat([per_year[y] for y in years],
                                      ignore_index=True)
            path = config.DATASET_DIR / f"{split}.parquet"
            frames[split].to_parquet(path, engine="pyarrow", index=False)
            print(f"wrote {path.name}  rows={len(frames[split]):,}  years={years}")
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    table = pd.DataFrame(infos)
    out = config.RESULTS_DIR / "step07_dataset_summary.csv"
    table.to_csv(out, index=False, encoding="utf-8")

    print("\n=== Splits ===")
    for split in config.SPLITS:
        frame = frames[split]
        print(f"{split:<5} rows={len(frame):>9,}  정시={((frame[TARGET] == 0).mean()):.2%}"
              f"  미도착={((frame[TARGET] == 9).mean()):.2%}  "
              f"years={sorted(frame['Year'].unique().tolist())}")

    print(f"\n=== Columns ===\n모델 입력 ({len(config.MODEL_FEATURES)}): {config.MODEL_FEATURES}")
    print(f"시나리오(14단계 모델 B 전용): {SCENARIO_COLUMNS}")
    print(f"target: {TARGET}\nmeta (never a model input): {META_COLUMNS}")

    print("\n=== Nulls in feature columns ===")
    for split in config.SPLITS:
        nulls = frames[split][FEATURE_COLUMNS].isna().sum()
        nulls = nulls[nulls > 0]
        print(f"{split}: {'none' if not len(nulls) else nulls.to_dict()}")

    print("\n=== Year overlap (must all be empty) ===")
    years = {s: set(frames[s]["Year"].unique()) for s in config.SPLITS}
    print(f"train&val: {years['train'] & years['val']}  "
          f"train&test: {years['train'] & years['test']}  "
          f"val&test: {years['val'] & years['test']}")
    print(f"\nsummary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
