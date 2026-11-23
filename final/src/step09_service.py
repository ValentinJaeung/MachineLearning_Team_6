"""step09 — 평가 구간을 한 번 열어 서비스가 약속을 지켰는지 센다.

읽기  results/step07_decision.json, data/model/calibrator.joblib, data/prep/test_*
쓰기  results/step09_calibration_test.csv  보정 전후 (보고서 6.2)
      results/step09_overall.csv           표 1 — 전체
      results/step09_cell_rate.csv         표 2 — 조합별 이행률 (+ 편수)
      results/step09_cell_delay.csv        표 3 — 조합별 평균 보장 지연·보장 불가
      results/step09_example_flight.csv    한 편의 25칸 (보고서 4.4 예시 화면)
      figures/step09_*.png

평가 구간은 여기서 한 번만 열린다
---------------------------------
모델 선택(step07)과 보정(step08)은 전부 검증 구간에서 끝났다. 여기서 보고
설계를 바꾸면 2026 년은 더 이상 "아직 보지 못한 미래" 가 아니게 된다.

채점 방법
---------
각 편에 **그 편의 실제 조합**을 입력으로 넣는다. 손님에게 25칸을 모두 보여주는
서비스이므로 표 안에는 **항상 맞는 칸이 들어 있고**, 검증은 "그 맞는 칸이
약속을 지키는가" 를 확인하는 것이다.

  보장 지연   누적 곡선이 0.80 에 처음 닿는 구간 경계
  보장 불가   곡선이 0.80 에 끝까지 닿지 못하는 편 — 분모에서 빼고 비율을 보고
  지킴        도착했고 실제 지연 ≤ 보장 지연
  못 지킴     결항·미도착, 또는 보장 지연을 넘긴 편

조합별로는 **실제 조합이 그 칸인 편만** 모아서 센다. 그래야 그 칸이 자기
조건에서 정직한지를 잴 수 있다. 편수가 MIN_CELL_FLIGHTS 미만인 칸은
'판단 불가' 로 적는다 — 적은 표본의 비율을 그대로 믿으면 숫자에 속는다.

이행률이 0.80 보다 높게 나오는 구조적 이유
------------------------------------------
보장 지연을 **구간 경계에서** 읽기 때문이다. 실제로 0.80 을 넘는 지점이 15~30분
사이여도 경계가 8개뿐이라 보수적으로 30분을 쓴다. 그래서 이행률은 위로 밀리고,
그 과보수의 비용이 바로 '평균 보장 지연' 이다. 두 숫자는 같이 읽어야 한다.

실행  python src/step09_service.py              평가 구간 (보고서에 쓰는 값)
      python src/step09_service.py --split val  검증 구간에서 코드 경로만 점검

--split val 은 평가 구간을 열기 전에 계산이 제대로 돌아가는지 확인하는 용도다.
결과 파일에 _val 이 붙어 따로 저장되므로 보고서용 수치와 섞이지 않는다.
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

import common
import config
import plotting
from plotting import plt

NAMES = config.SCENARIO_NAMES


def heatmap(values, title, path, fmt="{:.3f}", centre=None, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    masked = np.ma.masked_invalid(values)
    if centre is not None:
        span = np.nanmax(np.abs(values - centre)) or 0.01
        im = ax.imshow(masked, cmap=cmap, vmin=centre - span, vmax=centre + span)
    else:
        im = ax.imshow(masked, cmap=cmap)
    ax.set_xticks(range(5), NAMES, rotation=35, ha="right")
    ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("도착지 유형")
    ax.set_ylabel("출발지 유형")
    for i in range(5):
        for j in range(5):
            text = "판단 불가" if not np.isfinite(values[i, j]) else fmt.format(values[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def example_flight(family: str, calibrator, meta: pd.DataFrame,
                   split: str) -> pd.DataFrame:
    """한 편을 골라 25칸을 전부 채운다 — 손님이 보는 화면 그대로."""
    columns = json.loads((config.RESULTS_DIR / "step05_columns.json")
                         .read_text(encoding="utf-8"))
    # 양쪽 다 큰 공항이고 오후에 떠나는 편을 하나 고른다(시드 고정)
    pool = meta.index[(meta["crs_dep_hour"] == 14)].to_numpy()
    row = int(np.random.default_rng(config.SEED).choice(pool))
    info = meta.loc[row]

    pairs = [(o, d) for o in range(5) for d in range(5)]

    if family == "xgboost":
        import xgboost as xgb
        frame = common.load_tree(split).iloc[[row] * 25].reset_index(drop=True)
        for i, (o, d) in enumerate(pairs):
            frame.loc[i, "scen_origin"] = str(o)
            frame.loc[i, "scen_dest"] = str(d)
        booster = xgb.Booster()
        booster.load_model(config.MODEL_DIR / "xgboost_best.json")
        span = tuple(json.loads((config.MODEL_DIR / "xgboost_best_span.json")
                                .read_text(encoding="utf-8"))["iteration_range"])
        probs = booster.predict(xgb.DMatrix(frame, enable_categorical=True),
                                iteration_range=span).astype("float64")
    else:
        names = columns["linear_columns"]
        matrix = common.load_linear(split)[[row] * 25].tolil()
        slots = {side: {k: names.index(f"cat__scen_{side}_{k}")
                        for k in range(5)} for side in ("origin", "dest")}
        for i, (o, d) in enumerate(pairs):
            for side, value in (("origin", o), ("dest", d)):
                for k, column in slots[side].items():
                    matrix[i, column] = 1.0 if k == value else 0.0
        matrix = matrix.tocsr()
        if family == "logistic":
            model = joblib.load(config.MODEL_DIR / "logistic_best.joblib")
            probs = model.predict_proba(matrix).astype("float64")
        else:
            import torch
            from torch import nn
            bundle = torch.load(config.MODEL_DIR / "mlp_best.pt",
                                weights_only=False)
            hidden, width = bundle["hidden"], bundle["n_features"]
            layers = []
            for i, size in enumerate(hidden):
                layers += [nn.Linear(width, size), nn.ReLU()]
                if i < len(hidden) - 1:
                    layers.append(nn.Dropout(config.MLP_DROPOUT))
                width = size
            layers.append(nn.Linear(width, config.N_CLASS))
            model = nn.Sequential(*layers)
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(
                    model(torch.from_numpy(matrix.toarray())), dim=1
                ).numpy().astype("float64")

    curve = calibrator.transform(probs)
    delay, has = common.guarantee(curve)
    out = pd.DataFrame({
        "출발지 유형": [NAMES[o] for o, _ in pairs],
        "도착지 유형": [NAMES[d] for _, d in pairs],
        "80% 보장 지연(분)": np.where(has, delay, np.nan),
        "보장 가능": has,
        "미도착 확률": probs[:, config.NOARRIVE].round(5),
    })
    out.attrs["flight"] = (f"{info['Reporting_Airline']} "
                           f"{info['Origin']}→{info['Dest']} "
                           f"{pd.Timestamp(info['FlightDate']).date()} "
                           f"{int(info['crs_dep_hour'])}시 출발")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=list(config.SPLITS), default="test")
    args = parser.parse_args()
    split = args.split
    suffix = "" if split == "test" else f"_{split}"

    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    decision = common.winner()
    family, label = decision["chosen_family"], decision["chosen_label"]
    calibrator = joblib.load(config.MODEL_DIR / "calibrator.joblib")

    if split == "test":
        print(f"=== 평가 구간 {config.TEST_YEAR} 를 여는 유일한 지점 ===")
    else:
        print(f"=== [{split}] 코드 경로 점검 — 보고서용 수치가 아니다 ===")
    print(f"모델: {label} ({decision['chosen_setting']})")
    print(f"분할: {split}")
    print("각 편에 그 편의 실제 조합을 입력으로 넣는다.\n")

    y = common.load_y(split)
    meta = common.load_meta(split)
    delay = meta["arr_delay_min"].to_numpy(dtype="float64")

    probs_raw = common.predict_split(family, split)
    curve_raw = common.cdf(probs_raw)
    curve_cal = calibrator.transform(probs_raw)
    probs_cal = calibrator.transform_probs(probs_raw)

    # ------------------------------------------------------------------ #
    # 보고서 6.2 — 보정 전후
    # ------------------------------------------------------------------ #
    calib = pd.DataFrame([
        {"단계": "보정 전",
         "log_loss": round(common.log_loss(probs_raw, y), 6),
         "cdf_보정오차": round(
             common.cdf_calibration_error(curve_raw, y, delay), 6),
         **common.promise_table(curve_raw, y, delay)},
        {"단계": "보정 후",
         "log_loss": round(common.log_loss(probs_cal, y), 6),
         "cdf_보정오차": round(
             common.cdf_calibration_error(curve_cal, y, delay), 6),
         **common.promise_table(curve_cal, y, delay)},
    ])
    calib.to_csv(config.RESULTS_DIR / f"step09_calibration_test{suffix}.csv",
                 index=False, encoding="utf-8-sig")
    print(f"=== 6.2 보정 전후 ({split}) ===")
    print(calib.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 표 1 — 전체 (보정 후가 배포되는 값)
    # ------------------------------------------------------------------ #
    overall = pd.DataFrame([{
        "요구 확률": config.CONFIDENCE,
        **common.promise_table(curve_cal, y, delay),
    }])
    overall.to_csv(config.RESULTS_DIR / f"step09_overall{suffix}.csv",
                   index=False, encoding="utf-8-sig")
    print("\n=== 표 1 — 전체 (보정 후) ===")
    print(overall.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 표 2·3 — 조합별 25칸
    # ------------------------------------------------------------------ #
    origin = meta["scen_origin"].to_numpy()
    dest = meta["scen_dest"].to_numpy()
    rate = np.full((5, 5), np.nan)
    err = np.full((5, 5), np.nan)
    counts = np.zeros((5, 5), dtype="int64")
    mean_delay = np.full((5, 5), np.nan)
    no_guarantee = np.full((5, 5), np.nan)

    cell_rows = []
    for o in range(5):
        for d in range(5):
            mask = (origin == o) & (dest == d)
            n = int(mask.sum())
            counts[o, d] = n
            judged = n >= config.MIN_CELL_FLIGHTS
            stats = (common.promise_table(curve_cal[mask], y[mask], delay[mask])
                     if n else None)
            if stats:
                no_guarantee[o, d] = stats["보장불가비율"]
                mean_delay[o, d] = stats["평균보장지연"]
                if judged:
                    rate[o, d] = stats["이행률"]
                    err[o, d] = stats["이행률오차"]
            cell_rows.append({
                "출발지 유형": NAMES[o], "도착지 유형": NAMES[d], "편수": n,
                "판단": "가능" if judged else "판단 불가",
                "이행률": stats["이행률"] if stats else np.nan,
                "이행률오차": stats["이행률오차"] if stats else np.nan,
                "평균보장지연": stats["평균보장지연"] if stats else np.nan,
                "보장불가비율": stats["보장불가비율"] if stats else np.nan,
            })

    cells = pd.DataFrame(cell_rows)
    rate_table = pd.DataFrame(rate, index=NAMES, columns=NAMES).round(4)
    count_table = pd.DataFrame(counts, index=NAMES, columns=NAMES)
    delay_table = pd.DataFrame(mean_delay, index=NAMES, columns=NAMES).round(1)
    nog_table = pd.DataFrame(no_guarantee, index=NAMES, columns=NAMES).round(4)

    cells.to_csv(config.RESULTS_DIR / f"step09_cells{suffix}.csv",
                 index=False, encoding="utf-8-sig")
    pd.concat({"이행률": rate_table, "편수": count_table}).to_csv(
        config.RESULTS_DIR / f"step09_cell_rate{suffix}.csv", encoding="utf-8-sig")
    pd.concat({"평균보장지연(분)": delay_table, "보장불가비율": nog_table}).to_csv(
        config.RESULTS_DIR / f"step09_cell_delay{suffix}.csv", encoding="utf-8-sig")

    print(f"\n=== 표 2 — 조합별 이행률 (목표 {config.CONFIDENCE}) ===")
    print(rate_table.to_string())
    print("\n--- 편수 ---")
    print(count_table.to_string())
    print("\n=== 표 3 — 조합별 평균 보장 지연(분) ===")
    print(delay_table.to_string())
    print("\n--- 보장 불가 비율 ---")
    print(nog_table.to_string())

    thin = int((counts < config.MIN_CELL_FLIGHTS).sum())
    below = int(np.nansum(rate < config.CONFIDENCE))
    print(f"\n판단 불가 칸 {thin}개 / 0.80 미달 칸 {below}개")

    heatmap(rate, f"step09 — 조합별 약속 이행률 (목표 {config.CONFIDENCE:.2f})",
            config.FIGURE_DIR / f"step09_cell_rate{suffix}.png",
            centre=config.CONFIDENCE)
    heatmap(mean_delay, "step09 — 조합별 평균 80% 보장 지연(분)",
            config.FIGURE_DIR / f"step09_cell_delay{suffix}.png",
            fmt="{:.0f}", cmap="YlOrRd")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(calib["단계"], calib["이행률"],
                  color=["#adb5bd", "#2b8a3e"], width=0.5)
    for bar, v in zip(bars, calib["이행률"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}",
                ha="center")
    ax.axhline(config.CONFIDENCE, color="red", ls="--", lw=1.3,
               label=f"약속 {config.CONFIDENCE:.0%}")
    ax.set_ylim(min(calib["이행률"].min(), config.CONFIDENCE) - 0.05,
                max(calib["이행률"]) + 0.05)
    ax.set_ylabel("실제 이행률")
    ax.set_title("step09 — 약속한 확률이 실제로 지켜졌는가 (평가 2026)")
    ax.legend()
    fig.savefig(config.FIGURE_DIR / f"step09_promise{suffix}.png")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 보고서 4.4 — 한 편의 25칸 예시 화면
    # ------------------------------------------------------------------ #
    try:
        demo = example_flight(family, calibrator, meta, split)
        demo.to_csv(config.RESULTS_DIR / f"step09_example_flight{suffix}.csv",
                    index=False, encoding="utf-8-sig")
        grid = demo.pivot(index="출발지 유형", columns="도착지 유형",
                          values="80% 보장 지연(분)").reindex(index=NAMES,
                                                          columns=NAMES)
        print(f"\n=== 4.4 예시 화면 — {demo.attrs['flight']} ===")
        print("80% 보장 지연(분), 예정 도착 시각 기준")
        print(grid.to_string())
        heatmap(grid.to_numpy(dtype="float64"),
                f"step09 — 한 편의 25칸 ({demo.attrs['flight']})",
                config.FIGURE_DIR / f"step09_example_flight{suffix}.png",
                fmt="{:.0f}", cmap="YlOrRd")
    except Exception as error:                       # 예시는 본 결과가 아니다
        print(f"\n예시 화면 생성 실패(본 결과에는 영향 없음): {error!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
