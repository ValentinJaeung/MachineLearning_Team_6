"""
Leakage and consistency audit (run after step 9, re-run after any change).

Writes results/verify_report.csv and exits non-zero if any check fails.

The project's whole claim is "only what a passenger could know at booking
time". That claim is worth nothing if it is only asserted in prose, so it is
written down here as checks a machine runs. Every check prints PASS or FAIL
with the number it measured, not just a verdict.

Groups
------
  A  splits        the three years never overlap; row counts line up across
                   every file a split is stored in
  B  columns       no outcome column (B group), no post-hoc column (C group)
                   and no identifier (A-2) reaches a model matrix
  C  fit-on-train  the imputation median and the standardisation statistics
                   equal the values recomputed from train alone
  D  target        y_class is inside 0..9 and agrees with arr_delay_min
  E  report card   year Y's features came from year Y-1 and nothing else
  F  vocabulary    one-hot columns only use categories seen in train

Usage:  python src/verify_leakage.py
"""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

import config

RESULTS = []


def check(group: str, name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"group": group, "check": name,
                    "result": "PASS" if passed else "FAIL", "detail": detail})
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    print("=== Leakage and consistency audit ===")
    meta = {s: pd.read_parquet(config.PREP_DIR / f"{s}_meta.parquet")
            for s in config.SPLITS}
    xgb = {s: pd.read_parquet(config.PREP_DIR / f"{s}_xgb.parquet")
           for s in config.SPLITS}
    y = {s: np.load(config.PREP_DIR / f"{s}_y.npy") for s in config.SPLITS}
    scen = {s: pd.read_parquet(config.PREP_DIR / f"{s}_scen.parquet")
            for s in config.SPLITS}
    linear = {s: sparse.load_npz(config.PREP_DIR / f"{s}_linear.npz")
              for s in config.SPLITS}
    names = json.loads((config.PREP_DIR / "feature_names.json")
                       .read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- A --- #
    print("\nA. splits")
    years = {s: set(meta[s]["Year"].unique().tolist()) for s in config.SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = years[a] & years[b]
        check("A", f"{a} and {b} share no year", not overlap, f"overlap={overlap}")
    for s in config.SPLITS:
        check("A", f"{s}: expected years",
              years[s] == set(config.SPLIT_YEARS[s]),
              f"{sorted(years[s])}")
        n = len(meta[s])
        aligned = (len(xgb[s]) == n and len(y[s]) == n
                   and len(scen[s]) == n and linear[s].shape[0] == n)
        check("A", f"{s}: every file has {n:,} rows", aligned)

    # ---------------------------------------------------------------- B --- #
    print("\nB. columns a model can see")
    banned = set(config.B_OUTCOME) | set(config.C_POSTHOC) | set(config.A2_ID)
    banned |= {"arr_delay_min", "y_class", "ArrDel15", "DepDel15"}
    for s in config.SPLITS:
        leaked = sorted(set(xgb[s].columns) & banned)
        check("B", f"{s}: xgb matrix has no outcome/post-hoc/id column",
              not leaked, f"found={leaked}" if leaked else "14 features only")
    lower = [c.lower() for c in names["linear_columns"]]
    hits = sorted({b for b in banned if any(b.lower() in c for c in lower)})
    check("B", "linear matrix column names contain no banned field",
          not hits, f"found={hits}" if hits else f"{len(lower)} columns")
    check("B", "xgb feature list is exactly the 14 agreed features",
          list(xgb["train"].columns) == names["xgb_categorical"] + names["xgb_numeric"],
          f"{len(xgb['train'].columns)} columns")
    check("B", "scenario columns are stored apart from the feature matrices",
          all("scen_origin" not in xgb[s].columns for s in config.SPLITS))

    # ---------------------------------------------------------------- C --- #
    print("\nC. fitted on train only")
    pre = joblib.load(config.PREP_DIR / "preprocessor.joblib")
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    train_raw = pd.read_parquet(config.DATASET_DIR / "train.parquet",
                                columns=names["linear_numeric"])
    for i, col in enumerate(names["linear_numeric"]):
        median = float(train_raw[col].median())
        filled = train_raw[col].fillna(median)
        ok_median = np.isclose(imputer.statistics_[i], median, rtol=1e-9)
        ok_mean = np.isclose(scaler.mean_[i], filled.mean(), rtol=1e-6)
        ok_scale = np.isclose(scaler.scale_[i], filled.std(ddof=0), rtol=1e-6)
        check("C", f"{col}: median/mean/scale recomputed from train match",
              ok_median and ok_mean and ok_scale,
              f"median={median:.4f} mean={filled.mean():.4f}")

    # ---------------------------------------------------------------- D --- #
    print("\nD. target")
    for s in config.SPLITS:
        values = y[s]
        check("D", f"{s}: y_class within 0..{config.N_CLASS - 1}",
              bool(values.min() >= 0 and values.max() < config.N_CLASS),
              f"min={values.min()} max={values.max()}")
        delay = meta[s]["arr_delay_min"].to_numpy(dtype="float64")
        never = values == config.NOARRIVE
        check("D", f"{s}: class 9 means no arrival time, and only class 9 does",
              bool(np.isnan(delay[never]).all() and
                   (~np.isnan(delay[~never])).all()),
              f"class9={int(never.sum()):,}")
        rebuilt = np.digitize(delay[~never], config.EDGES, right=True)
        check("D", f"{s}: delay minutes reproduce the bin index",
              bool((rebuilt == values[~never]).all()))

    # ---------------------------------------------------------------- E --- #
    print("\nE. report card uses year Y-1 only")
    summary = pd.read_csv(config.RESULTS_DIR / "step05_history_summary.csv")
    check("E", "every target year was built from the year before it",
          bool((summary["target_year"] - summary["source_year"] == 1).all()),
          f"{len(summary)} years")
    for s in config.SPLITS:
        levels = meta[s]["substitute_level"]
        check("E", f"{s}: substitute_level fully assigned, 0..3",
              bool(levels.between(0, 3).all()),
              f"L0={float((levels == 0).mean()):.1%} "
              f"L3={float((levels == 3).mean()):.1%}")
    k = json.loads((config.RESULTS_DIR / "step05_smoothing_k.json")
                   .read_text(encoding="utf-8"))
    check("E", "smoothing k was estimated on the training source years only",
          k["estimated_on"] == [2021, 2022, 2023],
          f"k_A={k['A_absorb_rate']['k']:.2f} k_C={k['C_bad_day_rel_fail']['k']:.2f}")

    # ---------------------------------------------------------------- F --- #
    print("\nF. vocabulary")
    train_airlines = set(xgb["train"]["Reporting_Airline"].cat.categories)
    for s in ("val", "test"):
        used = set(xgb[s]["Reporting_Airline"].dropna().unique().tolist())
        check("F", f"{s}: no airline outside the train vocabulary",
              used <= train_airlines, f"{len(used)} airlines")
    other = names["other_label"]
    for s in ("val", "test"):
        cats = set(xgb[s]["Origin"].cat.categories)
        check("F", f"{s}: unseen airports fold into '{other}', not a new column",
              cats == set(xgb["train"]["Origin"].cat.categories),
              f"{len(cats)} categories")
    check("F", "linear matrix width is identical across splits",
          len({linear[s].shape[1] for s in config.SPLITS}) == 1,
          f"{linear['train'].shape[1]} columns")

    # --------------------------------------------------------------------- #
    table = pd.DataFrame(RESULTS)
    out = config.RESULTS_DIR / "verify_report.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    failed = int((table["result"] == "FAIL").sum())
    print(f"\n=== {len(table)} checks, {len(table) - failed} PASS, {failed} FAIL ===")
    print(f"report: {out}")
    if failed:
        print(table[table["result"] == "FAIL"].to_string(index=False),
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
