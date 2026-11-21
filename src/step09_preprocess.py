"""
Step 9 - two design matrices from the same fourteen features.

Reads  data/dataset/{split}.parquet (step 7), results/step08_settings.json
Writes data/prep/{split}_linear.npz      sparse matrix for the linear models
       data/prep/{split}_xgb.parquet     category-typed frame for the tree
       data/prep/{split}_y.npy           target
       data/prep/{split}_scen.parquet    scenario columns, kept apart
       data/prep/{split}_meta.parquet    identifiers and service fields
       data/prep/preprocessor.joblib, feature_names.json
       results/step09_preprocess_summary.csv, results/step09_numeric_scaling.csv

Why two forms of the same data
------------------------------
linear (logistic regression, MLP)
    categorical -> one-hot. A linear model cannot read "ATL"; numbering the
    airports 1, 2, 3 would invent a false order in which DFW sits between ATL
    and DEN.
    hours -> ALSO one-hot. Step 8 measured the shape over the day: 77.7% on
    time at 05h, 43.5% at 20h, and back up after midnight. A single scaled
    number can only draw one straight line through that.
    numeric -> median impute (train median), then standardise.

xgb (XGBoost)
    categorical stays categorical, hours stay numeric. A tree can split the
    same number repeatedly and build the bend itself, and one-hot would force
    it to spend a hundred splits on information it could use in one.

The point is to compare MODELS. Handing each model a form that suits it is
what makes the step 11 comparison a comparison of models rather than of
encodings.

Everything fitted on train only
-------------------------------
The median, the mean, the standard deviation, the airport vocabulary: all
from train. Validation and test are only transformed. Knowing the test year's
mean would be knowing the future distribution.

Usage:  python src/step09_preprocess.py [--top-airports N]
"""

from __future__ import annotations

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

OTHER = "기타"
TARGET = "y_class"

CATEGORICAL = ["Reporting_Airline", "Origin", "Dest", "DayOfWeek", "Month"]
HOURS = ["crs_dep_hour", "crs_arr_hour"]
NUMERIC = (HOURS + ["Distance", "CRSElapsedTime"]
           + config.SCHEDULE_FEATURES)
FEATURE_COLUMNS = CATEGORICAL + NUMERIC

LINEAR_CATEGORICAL = CATEGORICAL + HOURS
LINEAR_NUMERIC = [c for c in NUMERIC if c not in HOURS]

SCENARIO_COLUMNS = ["scen_origin", "scen_dest",
                    "scen_origin_applied", "scen_dest_applied"]

META_COLUMNS = [
    "Year", "FlightDate", "flight_key", "route_key", "group_key",
    "Origin", "Dest", "Reporting_Airline", "Flight_Number_Reporting_Airline",
    "OriginCityName", "DestCityName", "crs_dep_min", "crs_arr_min",
    "crs_dep_hour", "arr_delay_min",
]

AIRPORT_COLUMNS = ["Origin", "Dest"]


class Stop(Exception):
    """Fatal condition: stop and report."""


def top_airports(train: pd.DataFrame, n: int) -> list[str]:
    appearances = pd.concat([train[c] for c in AIRPORT_COLUMNS], ignore_index=True)
    return appearances.value_counts().head(n).index.tolist()


def fold_categories(frame: pd.DataFrame, keep: dict) -> tuple[pd.DataFrame, dict]:
    """Replace values outside the kept vocabulary with 기타; count how many."""
    out = frame.copy()
    folded = {}
    for col, allowed in keep.items():
        values = out[col].astype("object")
        outside = ~values.isin(allowed)
        folded[col] = int(outside.sum())
        out[col] = values.where(~outside, OTHER)
    return out, folded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-airports", type=int, default=None,
                        help="override the count step 8 decided")
    args = parser.parse_args()

    config.ensure_dirs()
    try:
        if args.top_airports is not None:
            n_airports = args.top_airports
            source = "command line"
        else:
            if not config.EDA_SETTINGS_PATH.exists():
                raise Stop("Run src/step08_eda.py first; it decides the airport count")
            settings = json.loads(config.EDA_SETTINGS_PATH.read_text(encoding="utf-8"))
            n_airports = int(settings["one_hot_airports"]["n_onehot"])
            source = "step 8 (95% of training departures)"
        print(f"one-hot airports: top {n_airports}  [{source}]")

        frames = {}
        for split in config.SPLITS:
            path = config.DATASET_DIR / f"{split}.parquet"
            if not path.exists():
                raise Stop(f"Missing step 7 output: {path}")
            frames[split] = pd.read_parquet(path)
            print(f"{split:<5} rows={len(frames[split]):,}")

        train = frames["train"]

        # --- vocabularies, all from train only ----------------------------- #
        keep_airports = top_airports(train, n_airports)
        all_train_airports = sorted(set(train["Origin"].dropna())
                                    | set(train["Dest"].dropna()))
        airlines = sorted(train["Reporting_Airline"].dropna().unique().tolist())
        coverage = float(pd.concat([train["Origin"], train["Dest"]],
                                   ignore_index=True).isin(keep_airports).mean())
        print(f"\nairports in train: {len(all_train_airports)}   "
              f"one-hot set covers {coverage:.2%} of train traffic   "
              f"airlines: {len(airlines)}")

        linear_keep = {"Origin": keep_airports, "Dest": keep_airports,
                       "Reporting_Airline": airlines}
        xgb_keep = {"Origin": all_train_airports, "Dest": all_train_airports,
                    "Reporting_Airline": airlines}

        categories = [
            np.array(airlines + [OTHER], dtype=object),
            np.array(keep_airports + [OTHER], dtype=object),
            np.array(keep_airports + [OTHER], dtype=object),
            np.array(sorted(train["DayOfWeek"].unique().tolist())),
            np.array(sorted(train["Month"].unique().tolist())),
            np.array(sorted(train["crs_dep_hour"].unique().tolist())),
            np.array(sorted(train["crs_arr_hour"].unique().tolist())),
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(categories=categories,
                                      handle_unknown="ignore",
                                      sparse_output=True, dtype=np.float32),
                 LINEAR_CATEGORICAL),
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]),
                 LINEAR_NUMERIC),
            ],
            sparse_threshold=0.3,
        )

        train_input, _ = fold_categories(train[FEATURE_COLUMNS], linear_keep)
        print("\nfitting the encoder and the scaler on train only ...")
        matrix = sparse.csr_matrix(preprocessor.fit_transform(train_input),
                                   dtype=np.float32)
        feature_names = preprocessor.get_feature_names_out().tolist()
        print(f"linear matrix: {matrix.shape[1]} columns")

        imputer = preprocessor.named_transformers_["num"].named_steps["impute"]
        scaler = preprocessor.named_transformers_["num"].named_steps["scale"]
        stats = pd.DataFrame({
            "feature": LINEAR_NUMERIC,
            "train_median_used_for_impute": imputer.statistics_,
            "train_mean": scaler.mean_,
            "train_scale": scaler.scale_,
        })
        print("\nnumeric scaling, fitted on train and applied unchanged elsewhere:")
        print(stats.round(4).to_string(index=False))

        # --- transform every split ----------------------------------------- #
        rows = []
        for split in config.SPLITS:
            frame = frames[split]
            linear_input, folded = fold_categories(frame[FEATURE_COLUMNS], linear_keep)
            data = matrix if split == "train" else sparse.csr_matrix(
                preprocessor.transform(linear_input), dtype=np.float32)
            sparse.save_npz(config.PREP_DIR / f"{split}_linear.npz", data)

            xgb_input, folded_xgb = fold_categories(frame[FEATURE_COLUMNS], xgb_keep)
            for col, allowed in xgb_keep.items():
                xgb_input[col] = pd.Categorical(xgb_input[col],
                                                categories=list(allowed) + [OTHER])
            # DayOfWeek and Month are categories for the tree too. Cast to str
            # first: pandas writes an integer-categorical to parquet as plain
            # int64, which XGBoost would then read as an ordinal number.
            for col in ["DayOfWeek", "Month"]:
                labels = [str(v) for v in sorted(train[col].unique().tolist())]
                xgb_input[col] = pd.Categorical(
                    xgb_input[col].astype(int).astype(str), categories=labels)
            xgb_input.to_parquet(config.PREP_DIR / f"{split}_xgb.parquet",
                                 engine="pyarrow", index=False)

            np.save(config.PREP_DIR / f"{split}_y.npy",
                    frame[TARGET].to_numpy(dtype=np.int8))
            frame[SCENARIO_COLUMNS].to_parquet(
                config.PREP_DIR / f"{split}_scen.parquet", index=False)
            frame[META_COLUMNS].to_parquet(
                config.PREP_DIR / f"{split}_meta.parquet",
                engine="pyarrow", index=False)

            rows.append({
                "split": split, "rows": data.shape[0],
                "linear_columns": data.shape[1], "nonzeros": int(data.nnz),
                "density": round(data.nnz / (data.shape[0] * data.shape[1]), 5),
                "c0_share": round(float((frame[TARGET] == 0).mean()), 6),
                "airports_folded_to_other": folded["Origin"] + folded["Dest"],
                "airport_fold_share": round(
                    (folded["Origin"] + folded["Dest"]) / (2 * len(frame)), 5),
                "airlines_unseen_in_train": folded["Reporting_Airline"],
                "airports_unseen_in_train": (folded_xgb["Origin"]
                                             + folded_xgb["Dest"]),
            })
            print(f"{split:<5} linear {data.shape[0]:,} x {data.shape[1]}  "
                  f"nnz={data.nnz:,}  + xgb / y / scen / meta")

        joblib.dump(preprocessor, config.PREP_DIR / "preprocessor.joblib")
        (config.PREP_DIR / "feature_names.json").write_text(json.dumps({
            "linear_columns": feature_names,
            "linear_categorical": LINEAR_CATEGORICAL,
            "linear_numeric": LINEAR_NUMERIC,
            "xgb_categorical": CATEGORICAL,
            "xgb_numeric": NUMERIC,
            "top_airports": keep_airports,
            "other_label": OTHER,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        stats.to_csv(config.RESULTS_DIR / "step09_numeric_scaling.csv",
                     index=False, encoding="utf-8-sig")
        table = pd.DataFrame(rows)
        table.to_csv(config.RESULTS_DIR / "step09_preprocess_summary.csv",
                     index=False, encoding="utf-8-sig")
    except Stop as exc:
        print("\n=== STOPPED ===", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    print("\n=== Summary ===")
    print(table.to_string(index=False))
    print(f"\noutputs: {config.PREP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
