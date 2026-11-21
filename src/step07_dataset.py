"""
Step 7 - one table per split, joined by key fallback and cut by year.
(PROPOSAL_FLOW §4, 7단계)

Reads  data/clean/ (step 2), data/history/ (step 5), data/scenario/ (step 6)
Writes data/dataset/{train,val,test}.parquet
       results/step07_dataset_summary.csv

    train  2022 2023 2024    report cards from 2021 2022 2023
    val    2025               report card  from 2024
    test   2026               report card  from 2025

The split is by YEAR, not at random
-----------------------------------
The real task is "learn from the past, answer about the future". Shuffling at
random would put two flights from the same airport on the same day into
training and test at once, and that day's weather would be in both. The score
would then be measuring memory of a specific afternoon.

Train spans three years, but each year receives its own previous year's report
card. Pooling the three and building one report card would let a 2022 flight
see 2023.

The four-level fallback, recorded
---------------------------------
    0  the flight key itself appeared last summer
    1  it did not; the route key's value was used
    2  neither; the group key's value was used
    3  none of them; last summer's global mean was used

Level 3 uses the mean rather than a blank or a zero on purpose. New routes
appear every year and still need an answer. A blank teaches the model that
"blank = new route"; a zero claims the flight has never once been late. The
mean leans neither way. substitute_level is carried so that step 11 can check
whether the exactly-matched rows alone tell the same story.

Usage:  python src/step07_dataset.py
"""

from __future__ import annotations

import sys

import pandas as pd

import config

TARGET = "y_class"
FEATURE_COLUMNS = config.MODEL_FEATURES + config.HISTORY_FEATURES   # 12 + 5

SCENARIO_COLUMNS = ["scen_origin", "scen_dest",
                    "scen_origin_applied", "scen_dest_applied"]
SCHEDULE_EXTRA = ["pad_level"]       # carried for auditing, not an input

# carried through, never offered to a model
META_COLUMNS = [
    "Year", "FlightDate", "flight_key", "route_key", "group_key",
    "Flight_Number_Reporting_Airline", "OriginCityName", "DestCityName",
    "crs_dep_min", "crs_arr_min",
    "substitute_level", "is_substituted", "base_fail_rate", "arr_delay_min",
]

CLEAN_COLUMNS = sorted(set(
    config.BASE_FEATURES
    + ["FlightDate", "Year", "flight_key", "route_key", "group_key",
       "Flight_Number_Reporting_Airline", "OriginCityName", "DestCityName",
       "crs_dep_min", "crs_arr_min", "arr_delay_min", TARGET]
))

HISTORY_VALUES = config.HISTORY_FEATURES + ["base_fail_rate"]

CHAIN = [(0, "flight", "flight_key"),
         (1, "route", "route_key"),
         (2, "group", "group_key")]


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


def join_year(year: int) -> tuple[pd.DataFrame, dict]:
    frame = load_year(year)
    rows = len(frame)

    for col in HISTORY_VALUES:
        frame[col] = pd.Series(float("nan"), index=frame.index, dtype="float64")
    frame["substitute_level"] = pd.Series(-1, index=frame.index, dtype="int8")

    counts = {}
    for level, suffix, key in CHAIN:
        path = config.HISTORY_DIR / f"history_{year}_{suffix}.parquet"
        if not path.exists():
            raise Stop(f"Missing step 5 output: {path}")
        table = pd.read_parquet(path, columns=[key] + HISTORY_VALUES)
        if int(table[key].duplicated().sum()):
            raise Stop(f"{key} is not unique in {path.name}")

        todo = frame["substitute_level"] == -1
        aligned = table.set_index(key).reindex(frame.loc[todo, key])
        hit = pd.Series(aligned[HISTORY_VALUES[0]].notna().to_numpy(),
                        index=frame.index[todo])
        hit_index = frame.index[todo][hit.to_numpy()]

        for col in HISTORY_VALUES:
            values = pd.Series(aligned[col].to_numpy(), index=frame.index[todo])
            frame.loc[hit_index, col] = values.loc[hit_index].astype("float64")
        frame.loc[hit_index, "substitute_level"] = level
        counts[level] = len(hit_index)

    # level 3: last summer's global mean
    flight_path = config.HISTORY_DIR / f"history_{year}_flight.parquet"
    global_values = {
        col: float(pd.read_parquet(flight_path, columns=[col])[col]
                   .astype("float64").mean())
        for col in HISTORY_VALUES
    }
    todo = frame["substitute_level"] == -1
    for col in HISTORY_VALUES:
        frame.loc[todo, col] = global_values[col]
    frame.loc[todo, "substitute_level"] = 3
    counts[3] = int(todo.sum())

    frame["is_substituted"] = (frame["substitute_level"] > 0).astype("int8")

    for col in ["Month", "DayOfWeek"]:
        frame[col] = frame[col].astype("int8")
    for col in ["crs_dep_hour", "crs_arr_hour"]:
        if int(frame[col].isna().sum()):
            raise Stop(f"{col} has missing values in {year}")
        frame[col] = frame[col].astype("int8")
    if int((frame["substitute_level"] < 0).sum()):
        raise Stop(f"Some rows were left unassigned in {year}")

    frame = frame[FEATURE_COLUMNS + SCENARIO_COLUMNS + SCHEDULE_EXTRA
                  + [TARGET] + META_COLUMNS]

    info = {
        "year": year, "rows": rows,
        "level0_flight_key": counts[0], "level1_route_key": counts[1],
        "level2_group_key": counts[2], "level3_global": counts[3],
        "substituted_share": round(float(frame["is_substituted"].mean()), 6),
        "c0_share": round(float((frame[TARGET] == 0).mean()), 6),
        "c9_share": round(float((frame[TARGET] == 9).mean()), 6),
    }
    print(f"[{year}] rows={rows:,}  flight {counts[0] / rows:.1%}  "
          f"route {counts[1] / rows:.1%}  group {counts[2] / rows:.2%}  "
          f"global {counts[3] / rows:.2%}  정시={info['c0_share']:.2%}")
    return frame, info


def main() -> int:
    config.ensure_dirs()
    infos, frames = [], {}
    try:
        per_year = {}
        for years in config.SPLIT_YEARS.values():
            for year in years:
                frame, info = join_year(year)
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
              f"  대체={frame['is_substituted'].mean():.2%}  "
              f"years={sorted(frame['Year'].unique().tolist())}")

    print("\n=== substitute_level share by split ===")
    for split in config.SPLITS:
        share = (frames[split]["substitute_level"].value_counts(normalize=True)
                 .sort_index())
        print(f"{split:<5} " + "  ".join(f"L{i}={v:.1%}" for i, v in share.items()))

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
