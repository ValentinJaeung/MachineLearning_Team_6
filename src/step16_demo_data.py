"""
Step 16 - pre-computed data for the demo page.

Reads  data/prep/ (step 9), data/model/ (steps 11, 12, 14)
Writes data/model/demo_data.json
       results/step16_demo_summary.csv

Why pre-compute
---------------
The demo must open on any laptop in a lecture room with no server, no install
and no network. So every distribution the page can show is computed here and
written into one JSON file; the page only reads the CDF and does arithmetic.

What the page needs, and nothing more
-------------------------------------
  cdf    the CALIBRATED distribution of the deployable model (A), stored as
         eight cumulative probabilities in per-mille integers. The main
         answer - the latest flight that still meets the deadline - is read
         from this.
  na     P(never arrives). Shown separately as a risk badge, never multiplied
         into the CDF: the CDF is already "probability of arriving", so
         applying it twice would discount the flight twice.
  scen   the same flight under each of the five assumed day types, from model
         B. Model B is UNCALIBRATED and is handed an assumption, so these are
         only ever shown as a DIFFERENCE from 정상일, never as an absolute
         promise. The page must say so too.

Routes are chosen as the busiest in the test year, which is also where a
passenger has a real choice of flights to shift between.

Usage:  python src/step16_demo_data.py [--routes N] [--days N]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import common
import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=24)
    parser.add_argument("--days", type=int, default=0,
                        help="0 = every day of the test summer")
    args = parser.parse_args()

    config.set_seeds()
    config.ensure_dirs()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    test = common.load_split("test", drop=drop, with_scenario=True)
    meta = test["meta"]

    calibrated_path = config.MODEL_DIR / "test_pred_calibrated.npy"
    if not calibrated_path.exists():
        raise SystemExit("Run src/step15_service_eval.py first")
    calibrated = np.load(calibrated_path).astype("float64")

    route = meta["Origin"].astype(str) + "-" + meta["Dest"].astype(str)
    busiest = route.value_counts().head(args.routes).index.tolist()
    chosen = route.isin(busiest).to_numpy()
    index = np.flatnonzero(chosen)
    print(f"routes: {len(busiest)}   flights: {len(index):,}")

    # --- model B, the same flights under each assumed day type ------------ #
    model_b = xgb.Booster()
    model_b.load_model(config.MODEL_DIR / "xgboost_scenario.json")
    frame = test["X_xgb"].iloc[index].reset_index(drop=True)
    scen_cdf = np.empty((len(index), 5, len(config.EDGES)), dtype="float64")
    for s in range(5):
        trial = frame.copy()
        for column in ("scen_origin", "scen_dest"):
            trial[column] = pd.Categorical([str(s)] * len(trial),
                                           categories=[str(i) for i in range(5)])
        probs = model_b.predict(xgb.DMatrix(trial, enable_categorical=True)
                                ).astype("float64")
        scen_cdf[:, s, :] = probs.cumsum(axis=1)[:, :len(config.EDGES)]
        print(f"  scenario {config.SCENARIO_NAMES[s]} done")

    main_cdf = calibrated[index].cumsum(axis=1)[:, :len(config.EDGES)]
    na = calibrated[index, config.NOARRIVE]

    sub = meta.iloc[index].reset_index(drop=True)
    sub["route"] = route.to_numpy()[index]
    sub["date"] = pd.to_datetime(sub["FlightDate"]).dt.strftime("%Y-%m-%d")

    routes: dict = {}
    for name in busiest:
        rows = np.flatnonzero(sub["route"].to_numpy() == name)
        first = sub.iloc[rows[0]]
        days: dict = {}
        for date, part in sub.iloc[rows].groupby("date", sort=True):
            flights = []
            for position, row in zip(part.index, part.itertuples()):
                flights.append({
                    "al": str(row.Reporting_Airline),
                    "no": str(row.Flight_Number_Reporting_Airline),
                    "dep": int(row.crs_dep_min),
                    "arr": int(row.crs_arr_min),
                    "cdf": [int(round(v * 1000)) for v in main_cdf[position]],
                    "na": int(round(na[position] * 1000)),
                    "scen": [[int(round(v * 1000)) for v in scen_cdf[position, s]]
                             for s in range(5)],
                })
            flights.sort(key=lambda f: f["dep"])
            days[date] = flights
            if args.days and len(days) >= args.days:
                break
        routes[name] = {
            "origin": str(first["Origin"]), "dest": str(first["Dest"]),
            "originCity": str(first["OriginCityName"]),
            "destCity": str(first["DestCityName"]),
            "days": days,
        }

    payload = {
        "edges": config.EDGES,
        "classNames": config.CLASS_NAMES,
        "scenarios": config.SCENARIO_NAMES,
        "note": {
            "cdf": "보정 완료, 배포 모델 A — 메인 답에 사용",
            "scen": "보정 전, 모델 B — 정상일 대비 차이로만 사용할 것",
            "na": "미도착 확률. CDF 에 다시 곱하지 말 것(이중 차감)",
            "period": "미국 국내선 2026년 6–8월, 가격·실시간 정보 없음",
        },
    }
    payload["routes"] = routes

    out = config.MODEL_DIR / "demo_data.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")

    counts = [{"route": name,
               "days": len(info["days"]),
               "flights": sum(len(v) for v in info["days"].values())}
              for name, info in routes.items()]
    summary = pd.DataFrame(counts)
    summary.to_csv(config.RESULTS_DIR / "step16_demo_summary.csv",
                   index=False, encoding="utf-8-sig")

    size_mb = out.stat().st_size / 1e6
    print(f"\n{len(routes)} routes · {int(summary['flights'].sum()):,} flights "
          f"· {size_mb:.1f} MB")
    print(summary.head(10).to_string(index=False))
    print(f"\noutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
