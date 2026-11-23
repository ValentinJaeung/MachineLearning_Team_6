"""
Step 15 - did the promise hold? The one and only look at test 2026.

Reads  data/prep/ (step 9), data/model/ (steps 11, 12)
Writes data/model/test_pred_{raw,calibrated}.npy
       results/step15_metrics.csv, step15_promise.csv,
       results/step15_weakness_*.csv, step15_service_sim.csv
       figures/step15_promise.png, figures/step15_service.png

The test year opens once
------------------------
Looking and then changing the design would make 2026 no longer "a future we
have not seen". Any change of mind has to be rehearsed on validation first.

The service calculation
-----------------------
    slack           = deadline − scheduled arrival      (negative → out)
    P(meets)        = read off the distribution at `slack`
    guaranteed time = scheduled arrival + the minutes at which the CDF
                      reaches the requested confidence
    answer          = of the flights whose P(meets) ≥ requested confidence,
                      the one that departs LAST

Two things it refuses to do
---------------------------
1. It does NOT multiply by the cancellation probability again. The
   distribution is already "probability of arriving"; class 9 is excluded
   from the CDF, so multiplying would discount the flight twice. The
   cancellation risk is shown separately on the screen instead.
2. It does NOT invent a guaranteed time for a flight whose distribution never
   reaches the requested level. That flight is reported as "보장 불가".

The comparison that matters
---------------------------
"Take an early flight" is common sense and needs no model. What common sense
cannot do is find the BOUNDARY - the last flight that still keeps the
promise. So the service is scored against the first-flight-of-the-day rule on
the only axis where it can win: how much later the passenger may leave while
still keeping the deadline at the requested confidence.

Usage:  python src/step15_service_eval.py [--levels 0.8,0.9,0.95]
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

import common
import config
import plotting
from plotting import plt


def promise_table(probs, cls, delay, levels) -> pd.DataFrame:
    """For each confidence level: who got a promise, and who kept it."""
    cdf = probs.cumsum(axis=1)[:, :len(config.EDGES)]
    arrived = cls < config.NOARRIVE
    rows = []
    for level in levels:
        reach = cdf >= level
        has = reach.any(axis=1)
        first = np.where(has, reach.argmax(axis=1), len(config.EDGES) - 1)
        promised = np.asarray(config.EDGES)[first]
        kept = arrived & (delay <= promised)
        rows.append({
            "약속 확률": level,
            "약속을 줄 수 있었던 편": int(has.sum()),
            "약속 가능 비율": round(float(has.mean()), 4),
            "실제 이행률": round(float(kept[has].mean()), 4),
            "오차(%p)": round(float(kept[has].mean() - level) * 100, 2),
            "평균 보장 지연(분)": round(float(promised[has].mean()), 1),
        })
    return pd.DataFrame(rows)


def by_group(probs, cls, delay, group: pd.Series, level: float,
             min_rows: int = 5000) -> pd.DataFrame:
    cdf = probs.cumsum(axis=1)[:, :len(config.EDGES)]
    arrived = cls < config.NOARRIVE
    reach = cdf >= level
    has = reach.any(axis=1)
    first = np.where(has, reach.argmax(axis=1), len(config.EDGES) - 1)
    promised = np.asarray(config.EDGES)[first]
    kept = arrived & (delay <= promised)

    frame = pd.DataFrame({"group": group.to_numpy(), "has": has,
                          "kept": kept, "promised": promised})
    # how many flights could be given the promise at all - a group where the
    # service often answers "보장 불가" is weak in a different way than one
    # where it answers and then misses
    reachable = (frame.groupby("group", observed=True)["has"]
                 .agg(약속가능비율="mean", 전체편수="size").reset_index())
    out = (frame[frame["has"]].groupby("group", observed=True)
           .agg(n=("kept", "size"), 이행률=("kept", "mean"),
                평균보장지연=("promised", "mean")).reset_index())
    out = out.merge(reachable, on="group", how="left")
    out = out[out["전체편수"] >= min_rows]
    out["오차(%p)"] = ((out["이행률"] - level) * 100).round(2)
    return out.sort_values("이행률")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str, default="0.80,0.90,0.95")
    parser.add_argument("--sim-days", type=int, default=25_000,
                        help="how many route-days to simulate (sampled, seeded)")
    args = parser.parse_args()
    levels = [float(v) for v in args.levels.split(",")]

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    drop = json.loads((config.MODEL_DIR / "chosen.json")
                      .read_text(encoding="utf-8"))["dropped_features"]
    print("=== test 2026 를 여는 유일한 지점 ===")
    print("서비스가 답하는 방식 그대로 — 가정한 날 유형을 입력으로 넣는다.")
    test = common.load_split("test", drop=drop, with_scenario=True)
    booster = xgb.Booster()
    booster.load_model(config.MODEL_DIR / "xgboost_scenario.json")

    # the calibrator is fitted the same way the answer is given: on val 2025,
    # with the assumed type equal to that day's type.
    cal_path = config.MODEL_DIR / "conditional_calibrator.joblib"
    if cal_path.exists():
        calibrator = joblib.load(cal_path)
    else:
        print("  보정기를 val 2025 에서 적합 ...")
        val = common.load_split("val", drop=drop, with_scenario=True)
        raw_val = booster.predict(
            xgb.DMatrix(val["X_xgb"], enable_categorical=True)).astype("float64")
        calibrator = common.CdfCalibrator().fit(raw_val, val["cls"], val["delay"])
        joblib.dump(calibrator, cal_path)
        del val, raw_val

    raw = booster.predict(xgb.DMatrix(test["X_xgb"], enable_categorical=True)
                          ).astype("float64")
    calibrated = calibrator.transform(raw)
    np.save(config.MODEL_DIR / "test_pred_raw.npy", raw.astype("float32"))
    np.save(config.MODEL_DIR / "test_pred_calibrated.npy",
            calibrated.astype("float32"))

    cls, delay = test["cls"], test["delay"]
    metrics = pd.DataFrame([
        {"stage": "보정 전", **common.flat(common.evaluate(raw, cls, delay))},
        {"stage": "보정 후", **common.flat(common.evaluate(calibrated, cls, delay))},
    ])
    metrics.to_csv(config.RESULTS_DIR / "step15_metrics.csv",
                   index=False, encoding="utf-8-sig")
    print("\n=== test 2026 지표 ===")
    print(metrics.to_string(index=False))

    # ------------------------------------------------------------------ #
    # the headline: were the promises kept?
    # ------------------------------------------------------------------ #
    before = promise_table(raw, cls, delay, levels).assign(stage="보정 전")
    after = promise_table(calibrated, cls, delay, levels).assign(stage="보정 후")
    promise = pd.concat([before, after], ignore_index=True)
    promise.to_csv(config.RESULTS_DIR / "step15_promise.csv",
                   index=False, encoding="utf-8-sig")
    print("\n=== 약속 이행률 (핵심 지표) ===")
    print(promise.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.6, 5))
    width = 0.35
    x = np.arange(len(levels))
    for offset, stage, colour in ((-width / 2, "보정 전", "#adb5bd"),
                                  (width / 2, "보정 후", "#2b8a3e")):
        part = promise[promise["stage"] == stage]
        bars = ax.bar(x + offset, part["실제 이행률"], width, label=stage,
                      color=colour)
        for bar, v in zip(bars, part["실제 이행률"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", fontsize=9)
    for i, level in enumerate(levels):
        ax.plot([i - 0.45, i + 0.45], [level, level], "r--", lw=1.4)
    ax.set_xticks(x, [f"{l:.0%} 약속" for l in levels])
    ax.set_ylim(min(promise["실제 이행률"]) - 0.06, 1.0)
    ax.set_ylabel("실제 이행률")
    ax.set_title("Step 15 — 약속한 확률이 실제로 지켜졌는가 (test 2026, 빨강 = 약속선)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step15_promise.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # where it is weakest - reported, not hidden
    # ------------------------------------------------------------------ #
    meta = test["meta"]
    traffic = meta["Origin"].value_counts()
    size_band = meta["Origin"].map(
        lambda a: "대형" if traffic.get(a, 0) >= 20000
        else ("중형" if traffic.get(a, 0) >= 5000 else "소형"))
    groups = {
        "airline": meta["Reporting_Airline"],
        "airport_size": size_band,
        "month": pd.to_datetime(meta["FlightDate"]).dt.month.astype(str) + "월",
        "assumed_type": test["X_xgb"]["scen_origin"].astype(int)
        .map(dict(enumerate(config.SCENARIO_NAMES))),
    }
    print("\n=== 90% 약속의 약한 곳 ===")
    for name, series in groups.items():
        table = by_group(calibrated, cls, delay, series, 0.90)
        table.to_csv(config.RESULTS_DIR / f"step15_weakness_{name}.csv",
                     index=False, encoding="utf-8-sig")
        worst = table.iloc[0]
        print(f"  {name:<17} 최악 {worst['group']}: 이행률 "
              f"{worst['이행률']:.3f} ({worst['오차(%p)']:+.2f}%p, n={int(worst['n']):,})")

    # ------------------------------------------------------------------ #
    # the service simulation: how much later may the passenger leave?
    # ------------------------------------------------------------------ #
    print("\n=== 서비스 시뮬레이션 ===")
    frame = pd.DataFrame({
        "Origin": meta["Origin"].to_numpy(), "Dest": meta["Dest"].to_numpy(),
        "FlightDate": pd.to_datetime(meta["FlightDate"]).to_numpy(),
        "dep": meta["crs_dep_min"].to_numpy(dtype="float64"),
        "arr": meta["crs_arr_min"].to_numpy(dtype="float64"),
        "row": np.arange(len(meta)),
    })
    cdf = calibrated.cumsum(axis=1)[:, :len(config.EDGES)]
    edges = np.asarray(config.EDGES, dtype="float64")
    arrived = cls < config.NOARRIVE

    rng = np.random.default_rng(config.SEED)
    keys = frame.groupby(["Origin", "Dest", "FlightDate"], sort=False).indices
    multi = [v for v in keys.values() if len(v) >= 2]
    print(f"  후보가 2편 이상인 노선-날짜 {len(multi):,}개 중 "
          f"{min(args.sim_days, len(multi)):,}개 표본 추출")
    take = rng.choice(len(multi), size=min(args.sim_days, len(multi)),
                      replace=False)
    sampled = [multi[i] for i in take]

    dep_all = frame["dep"].to_numpy()
    arr_all = frame["arr"].to_numpy()

    sim_rows = []
    for level in levels:
        picked_service, picked_early, kept_service, kept_early = [], [], [], []
        for rows_i in sampled:
            arr = arr_all[rows_i]
            dep = dep_all[rows_i]
            # the deadline: one hour after the latest scheduled arrival that day
            deadline = float(arr.max()) + 60.0
            slack = deadline - arr
            prob = np.array([
                np.interp(s, edges, cdf[r], left=0.0, right=float(cdf[r, -1]))
                for s, r in zip(slack, rows_i)])
            ok = (slack >= 0) & (prob >= level)
            if not ok.any():
                continue
            choice = int(np.argmax(np.where(ok, dep, -np.inf)))   # the LAST that qualifies
            first = int(np.argmin(dep))                           # common sense
            picked_service.append(dep[choice])
            picked_early.append(dep[first])
            for slot, idx in ((kept_service, choice), (kept_early, first)):
                r = rows_i[idx]
                slot.append(bool(arrived[r] and arr[idx] + delay[r] <= deadline))

        later = float(np.mean(np.asarray(picked_service)
                              - np.asarray(picked_early)))
        sim_rows.append({
            "요구 확률": level,
            "시뮬레이션 노선-날짜": len(sampled),
            "답을 찾은 경우": len(picked_service),
            "서비스 이행률": round(float(np.mean(kept_service)), 4),
            "첫 편 이행률": round(float(np.mean(kept_early)), 4),
            "더 늦게 출발(분)": round(later, 1),
        })
        print(f"  {level:.0%}: 서비스 이행 {sim_rows[-1]['서비스 이행률']:.3f} vs "
              f"첫 편 {sim_rows[-1]['첫 편 이행률']:.3f}  "
              f"평균 {later:.0f}분 늦게 출발")

    sim = pd.DataFrame(sim_rows)
    sim.to_csv(config.RESULTS_DIR / "step15_service_sim.csv",
               index=False, encoding="utf-8-sig")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(levels))
    ax1.bar(x - 0.2, sim["서비스 이행률"], 0.4, label="우리 답(가장 늦은 편)",
            color="#2b8a3e")
    ax1.bar(x + 0.2, sim["첫 편 이행률"], 0.4, label="상식(첫 편)", color="#adb5bd")
    ax1.set_xticks(x, [f"{l:.0%}" for l in levels])
    ax1.set_ylim(0.8, 1.01)
    ax1.set_ylabel("마감 준수율")
    ax1.legend(loc="lower left", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(x, sim["더 늦게 출발(분)"], "o--", color="#e8590c")
    ax2.set_ylabel("첫 편보다 더 늦게 출발할 수 있는 시간(분)", color="#e8590c")
    ax2.grid(False)
    ax1.set_title("Step 15 — 같은 마감을 지키면서 몇 분 더 늦게 떠날 수 있나 (test 2026)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step15_service.png")
    plt.close(fig)

    best = promise[(promise["stage"] == "보정 후") & (promise["약속 확률"] == 0.90)]
    print(f"\n핵심: 90% 약속의 실제 이행률 {float(best['실제 이행률'].iloc[0]):.3f} "
          f"({float(best['오차(%p)'].iloc[0]):+.2f}%p)")
    print("첫 편을 타라는 상식도 마감은 지킨다. 다만 하루를 통째로 쓴다.")
    print("우리 답은 같은 확률을 지키면서 더 늦게 떠나게 해 준다 — "
          "상식이 계산하지 못하는 것은 이 경계뿐이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
