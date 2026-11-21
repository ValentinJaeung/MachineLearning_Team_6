"""
Step 5 - last year's report card, per key.
(PROPOSAL_FLOW §4, 5단계 · §3-2 ⑤)

Reads  data/clean/, data/legs/ (step 2, 3), data/airport_day/ (step 4)
Writes data/history/history_{Y}_{flight,route,group}.parquet
       results/step05_history_summary.csv
       results/step05_smoothing_k.json

The rule, without exception
---------------------------
    year Y's report card is built from year Y-1 ONLY.
Not one row of year Y goes into it. That is exactly the information a
passenger could look up before booking.

Five features per key
---------------------
  A_absorb_rate         share of "previous leg was late" occasions on which
                        this flight still departed on time          (smoothed)
  B1_turnaround_mean    mean scheduled turnaround, minutes
  B2_day_seq_mean       mean position in the aircraft's day
  B3_first_flight_rate  share of days this was the first flight
  C_bad_day_rel_fail    on the airport's BAD days, (did this flight fail)
                        minus (that day's airport-wide failure rate)
                                                                    (smoothed)
  base_fail_rate        plain failure rate last summer - used by the step 10
                        look-up baseline, never offered as a model input

Why C subtracts the airport's own rate
--------------------------------------
A raw bad-day failure rate would mostly measure "is this flight at an airport
that has many bad days", which the airport name already says. Subtracting the
same day's airport-wide rate makes the other flights of that day the baseline,
so weather and ATC cancel out and only this flight's own robustness is left.
Negative means it held up better than the airport around it.

Three key widths, because flight numbers move
---------------------------------------------
Airlines reassign flight numbers to different routes between years, so the
narrow key alone would miss most rows. Step 7 walks flight -> route -> group
-> global and records how far it had to go.

Smoothing, with k estimated rather than chosen
----------------------------------------------
    smoothed = (n * own_value + k * prior_value) / (n + k)

k is estimated by empirical Bayes, separately for A and for C:

    spread between keys = real differences + sampling noise
    k = (within-key variance per observation) / (between-key true variance)

so a feature whose key-to-key spread is mostly noise gets a large k (trust the
wider key) and one with real separation gets a small k. Estimated on the
TRAINING source years (2021-2023) only, so the validation and test years are
never looked at. Applied to A and C only: B is a continuous quantity like
"61 minutes of turnaround", which does not swing to an extreme on one
observation the way a one-trial rate does.

Usage:  python src/step05_history.py [--k-a K] [--k-c K]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config

TARGET_YEARS = range(2022, 2027)        # Y; built from Y-1
TRAIN_SOURCE_YEARS = [2021, 2022, 2023]  # the source years behind train 2022-24
FAIL_FROM_CLASS = 2                      # same definition as step 4

FEATURES = config.HISTORY_FEATURES
VALUE_COLUMNS = FEATURES + ["base_fail_rate"]

SUMMARY_PATH = config.RESULTS_DIR / "step05_history_summary.csv"
K_PATH = config.RESULTS_DIR / "step05_smoothing_k.json"


class Stop(Exception):
    """Fatal condition: stop and report."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_year(directory, year: int, columns: list[str]) -> pd.DataFrame:
    parts = []
    for month in config.MONTHS:
        path = directory / f"{year}_{month:02d}.parquet"
        if not path.exists():
            raise Stop(f"Missing input: {path}")
        parts.append(pd.read_parquet(path, columns=columns))
    return pd.concat(parts, ignore_index=True)


def source_material(year: int, airport_day: pd.DataFrame) -> dict:
    """Everything year Y-1 contributes: legs, flights, and bad-day rows."""
    legs = load_year(config.LEGS_DIR, year, [
        "flight_key", "route_key", "group_key",
        "day_seq", "is_first_flight", "turnaround_min", "absorbed"])
    clean = load_year(config.CLEAN_DIR, year, [
        "flight_key", "route_key", "group_key", "Origin", "FlightDate", "y_class"])
    clean["failed"] = (clean["y_class"] >= FAIL_FROM_CLASS).astype("float64")

    day = airport_day.loc[airport_day["year"] == year,
                          ["Origin", "FlightDate", "fail_rate", "bad_day"]]
    merged = clean.merge(day, on=["Origin", "FlightDate"], how="left")
    unmatched = int(merged["bad_day"].isna().sum())
    bad_rows = merged[merged["bad_day"] == 1].copy()
    bad_rows["rel_fail"] = bad_rows["failed"] - bad_rows["fail_rate"]
    return {"legs": legs, "clean": clean, "bad_rows": bad_rows,
            "unmatched": unmatched}


# --------------------------------------------------------------------------- #
# Empirical Bayes for the smoothing strength
# --------------------------------------------------------------------------- #


def estimate_k(values: pd.Series, keys: pd.Series) -> dict:
    """Method-of-moments k for `smoothed = (n*own + k*prior)/(n+k)`.

    values : one row per observation (0/1 for A, a residual for C)
    keys   : the key each observation belongs to

    within  = mean per-observation variance inside a key
    between = variance of the key means, minus the part sampling noise alone
              would produce
    k       = within / between
    """
    frame = pd.DataFrame({"v": values.astype("float64").to_numpy(),
                          "key": keys.to_numpy()}).dropna()
    grouped = frame.groupby("key")["v"]
    n = grouped.size()
    mean = grouped.mean()
    var = grouped.var(ddof=1)

    usable = n >= 2
    if usable.sum() < 50:
        raise Stop("too few keys with 2+ observations to estimate k")

    # pooled within-key variance, weighted by degrees of freedom
    within = float(((n[usable] - 1) * var[usable]).sum() / (n[usable] - 1).sum())

    # observed spread of key means, weighted by how much each key is worth
    weights = n.astype("float64")
    grand = float((mean * weights).sum() / weights.sum())
    observed = float((weights * (mean - grand) ** 2).sum() / weights.sum())

    # the part of that spread that sampling noise alone explains
    noise = float((weights * (within / n)).sum() / weights.sum())
    between = observed - noise

    if between <= 0:
        # no real key-to-key difference survives the noise: shrink hard
        k = float("inf")
        between = 0.0
    else:
        k = within / between

    return {"within": within, "between_observed": observed,
            "noise": noise, "between_true": between,
            "k": k, "n_keys": int(len(n)), "n_obs": int(n.sum()),
            "grand_mean": grand}


def fit_smoothing_k(airport_day: pd.DataFrame) -> dict:
    """Estimate k for A and for C on the training source years only."""
    absorb_v, absorb_k, rel_v, rel_k = [], [], [], []
    for year in TRAIN_SOURCE_YEARS:
        material = source_material(year, airport_day)
        legs = material["legs"]
        judge = legs[legs["absorbed"].notna()]
        absorb_v.append(judge["absorbed"].astype("float64"))
        absorb_k.append(judge["flight_key"].astype(str) + f"|{year}")
        bad = material["bad_rows"]
        rel_v.append(bad["rel_fail"])
        rel_k.append(bad["flight_key"].astype(str) + f"|{year}")

    a = estimate_k(pd.concat(absorb_v, ignore_index=True),
                   pd.concat(absorb_k, ignore_index=True))
    c = estimate_k(pd.concat(rel_v, ignore_index=True),
                   pd.concat(rel_k, ignore_index=True))

    out = {
        "estimated_on": TRAIN_SOURCE_YEARS,
        "note": ("k = within-key variance per observation / true between-key "
                 "variance; estimated on the training source years only"),
        "A_absorb_rate": a,
        "C_bad_day_rel_fail": c,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    K_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Empirical-Bayes smoothing strength (train source years only) ===")
    for name in ("A_absorb_rate", "C_bad_day_rel_fail"):
        d = out[name]
        print(f"{name:22s} keys={d['n_keys']:>8,}  obs={d['n_obs']:>9,}  "
              f"within={d['within']:.5f}  between(true)={d['between_true']:.6f}  "
              f"k={d['k']:.2f}")
    print(f"written: {K_PATH}\n")
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate_level(key: str, legs, clean, bad_rows) -> pd.DataFrame:
    """Raw per-key aggregates for one key width. No smoothing yet."""
    base = clean.groupby(key, observed=True)["failed"].agg(
        base_fail_rate="mean", n_flights="size")

    judgeable = legs[legs["absorbed"].notna()]
    absorb = judgeable.groupby(key, observed=True)["absorbed"].agg(
        absorbed_sum="sum", n_absorb="size")

    legs_agg = legs.groupby(key, observed=True).agg(
        B1_turnaround_mean=("turnaround_min", "mean"),
        n_turnaround=("turnaround_min", "count"),
        B2_day_seq_mean=("day_seq", "mean"),
        B3_first_flight_rate=("is_first_flight", "mean"),
        n_legs=("day_seq", "size"))

    bad = bad_rows.groupby(key, observed=True)["rel_fail"].agg(
        C_sum="sum", n_bad="size")

    out = base.join(absorb, how="left").join(legs_agg, how="left").join(bad, how="left")
    out["A_raw"] = out["absorbed_sum"] / out["n_absorb"]
    out["C_raw"] = out["C_sum"] / out["n_bad"]
    out["n_absorb"] = out["n_absorb"].fillna(0)
    out["n_bad"] = out["n_bad"].fillna(0)
    return out


def map_key(clean: pd.DataFrame, from_key: str, to_key: str) -> pd.Series:
    """The to_key each from_key most often belongs to."""
    counts = (clean.groupby([from_key, to_key], observed=True).size()
              .reset_index(name="n")
              .sort_values([from_key, "n"], ascending=[True, False]))
    return (counts.drop_duplicates(from_key).set_index(from_key)[to_key]
            .rename(to_key + "_main"))


def smooth(raw, n, prior, k: float):
    """(n*raw + k*prior) / (n + k). Equals the prior when n == 0."""
    n = n.fillna(0).astype("float64")
    raw = raw.astype("float64").fillna(0.0)
    prior = prior.astype("float64")
    if not np.isfinite(k):
        return prior
    return (n * raw + k * prior) / (n + k)


B_COLUMNS = ["B1_turnaround_mean", "B2_day_seq_mean", "B3_first_flight_rate"]


def build_history(target_year: int, k_a: float, k_c: float, airport_day) -> dict:
    source_year = target_year - 1
    material = source_material(source_year, airport_day)
    legs, clean, bad_rows = material["legs"], material["clean"], material["bad_rows"]

    global_values = {
        "A_absorb_rate": float(legs.loc[legs["absorbed"].notna(), "absorbed"]
                               .astype("float64").mean()),
        "B1_turnaround_mean": float(legs["turnaround_min"].mean()),
        "B2_day_seq_mean": float(legs["day_seq"].mean()),
        "B3_first_flight_rate": float(legs["is_first_flight"].mean()),
        "C_bad_day_rel_fail": float(bad_rows["rel_fail"].mean()),
        "base_fail_rate": float(clean["failed"].mean()),
    }

    # --- widest key: smoothed toward the global mean ----------------------- #
    group = aggregate_level("group_key", legs, clean, bad_rows)
    group["A_absorb_rate"] = smooth(
        group["A_raw"], group["n_absorb"],
        pd.Series(global_values["A_absorb_rate"], index=group.index), k_a)
    group["C_bad_day_rel_fail"] = smooth(
        group["C_raw"], group["n_bad"],
        pd.Series(global_values["C_bad_day_rel_fail"], index=group.index), k_c)
    for col in B_COLUMNS:
        group[col] = group[col].fillna(global_values[col])

    # --- middle key: smoothed toward its group ----------------------------- #
    route = aggregate_level("route_key", legs, clean, bad_rows)
    route = route.join(map_key(clean, "route_key", "group_key"), how="left")
    group_for_route = group.reindex(route["group_key_main"])

    prior_a = pd.Series(group_for_route["A_absorb_rate"].to_numpy(), index=route.index
                        ).fillna(global_values["A_absorb_rate"])
    prior_c = pd.Series(group_for_route["C_bad_day_rel_fail"].to_numpy(),
                        index=route.index).fillna(global_values["C_bad_day_rel_fail"])
    route["A_absorb_rate"] = smooth(route["A_raw"], route["n_absorb"], prior_a, k_a)
    route["C_bad_day_rel_fail"] = smooth(route["C_raw"], route["n_bad"], prior_c, k_c)
    for col in B_COLUMNS:
        fill = pd.Series(group_for_route[col].to_numpy(), index=route.index
                         ).fillna(global_values[col])
        route[col] = route[col].fillna(fill)
    for col in VALUE_COLUMNS:
        route[col] = route[col].fillna(global_values[col])

    # --- narrowest key: smoothed toward its route -------------------------- #
    flight = aggregate_level("flight_key", legs, clean, bad_rows)
    flight = flight.join(map_key(clean, "flight_key", "route_key"), how="left")
    flight = flight.join(map_key(clean, "flight_key", "group_key"), how="left")
    prior_table = route.reindex(flight["route_key_main"])

    prior_a = pd.Series(prior_table["A_absorb_rate"].to_numpy(), index=flight.index
                        ).fillna(global_values["A_absorb_rate"])
    prior_c = pd.Series(prior_table["C_bad_day_rel_fail"].to_numpy(), index=flight.index
                        ).fillna(global_values["C_bad_day_rel_fail"])
    flight["A_absorb_rate"] = smooth(flight["A_raw"], flight["n_absorb"], prior_a, k_a)
    flight["C_bad_day_rel_fail"] = smooth(flight["C_raw"], flight["n_bad"], prior_c, k_c)
    for col in B_COLUMNS:
        fill = pd.Series(prior_table[col].to_numpy(), index=flight.index
                         ).fillna(global_values[col])
        flight[col] = flight[col].fillna(fill)

    # --- write -------------------------------------------------------------- #
    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    count_cols = ["n_flights", "n_legs", "n_turnaround", "n_absorb", "n_bad"]
    flight_out = flight.reset_index()[
        ["flight_key", "route_key_main", "group_key_main"] + VALUE_COLUMNS + count_cols]
    route_out = route.reset_index()[
        ["route_key", "group_key_main"] + VALUE_COLUMNS + count_cols]
    group_out = group.reset_index()[["group_key"] + VALUE_COLUMNS + count_cols]

    for name, table in (("flight", flight_out), ("route", route_out),
                        ("group", group_out)):
        table.to_parquet(config.HISTORY_DIR / f"history_{target_year}_{name}.parquet",
                         engine="pyarrow", index=False)

    summary = {
        "target_year": target_year, "source_year": source_year,
        "k_a": k_a, "k_c": k_c,
        "flight_keys": len(flight_out), "route_keys": len(route_out),
        "group_keys": len(group_out), "bad_day_rows": len(bad_rows),
        "airport_day_unmatched": material["unmatched"],
        "flight_keys_with_absorb": int((flight_out["n_absorb"] > 0).sum()),
        "flight_keys_with_bad": int((flight_out["n_bad"] > 0).sum()),
        "route_keys_with_absorb": int((route_out["n_absorb"] > 0).sum()),
        "route_keys_with_bad": int((route_out["n_bad"] > 0).sum()),
        "flight_median_n_absorb": float(flight_out["n_absorb"].median()),
        "flight_median_n_bad": float(flight_out["n_bad"].median()),
        "route_median_n_bad": float(route_out["n_bad"].median()),
        "A_mean_flight": round(float(flight_out["A_absorb_rate"].mean()), 4),
        "C_mean_flight": round(float(flight_out["C_bad_day_rel_fail"].mean()), 5),
        "C_std_flight": round(float(flight_out["C_bad_day_rel_fail"].std()), 5),
        "C_std_route": round(float(route_out["C_bad_day_rel_fail"].std()), 5),
        "nulls": int(flight_out[VALUE_COLUMNS].isna().sum().sum()
                     + route_out[VALUE_COLUMNS].isna().sum().sum()
                     + group_out[VALUE_COLUMNS].isna().sum().sum()),
    }
    print(f"[Y={target_year} from {source_year}] flight={summary['flight_keys']:,}  "
          f"route={summary['route_keys']:,}  group={summary['group_keys']:,}  "
          f"flight keys with A obs "
          f"{summary['flight_keys_with_absorb'] / summary['flight_keys']:.0%} / "
          f"C obs {summary['flight_keys_with_bad'] / summary['flight_keys']:.0%}  "
          f"nulls={summary['nulls']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-a", type=float, default=None,
                        help="override the estimated k for A_absorb_rate")
    parser.add_argument("--k-c", type=float, default=None,
                        help="override the estimated k for C_bad_day_rel_fail")
    args = parser.parse_args()

    config.ensure_dirs()
    try:
        path = config.AIRPORT_DAY_DIR / "airport_day.parquet"
        if not path.exists():
            raise Stop(f"Missing step 4 output: {path}")
        airport_day = pd.read_parquet(path)

        fitted = fit_smoothing_k(airport_day)
        k_a = args.k_a if args.k_a is not None else fitted["A_absorb_rate"]["k"]
        k_c = args.k_c if args.k_c is not None else fitted["C_bad_day_rel_fail"]["k"]

        summaries = [build_history(y, k_a, k_c, airport_day) for y in TARGET_YEARS]
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    table = pd.DataFrame(summaries)
    table.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    print(f"\n=== Step 5 summary (k_A = {k_a:.2f}, k_C = {k_c:.2f}) ===")
    print(table[["target_year", "source_year", "flight_keys", "route_keys",
                 "group_keys", "flight_median_n_bad", "route_median_n_bad"]]
          .to_string(index=False))
    print("\nSmoothed feature spread (narrow key vs middle key):")
    print(table[["target_year", "A_mean_flight", "C_mean_flight",
                 "C_std_flight", "C_std_route"]].to_string(index=False))
    print(f"\nTotal nulls: {int(table['nulls'].sum())}")
    print(f"summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
