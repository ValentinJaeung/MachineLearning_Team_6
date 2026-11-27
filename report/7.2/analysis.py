import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SRC = OUT.parents[1] / "final" / "src"
sys.path.insert(0, str(SRC))

import common  # noqa: E402
import config  # noqa: E402

NAMES = config.SCENARIO_NAMES
cal = joblib.load(config.MODEL_DIR / "calibrator.joblib")


def edge_errors(curve, y, delay):
    arrived = y < config.NOARRIVE
    return [curve[:, i].mean() - (arrived & (delay <= e)).mean()
            for i, e in enumerate(config.EDGES)]


# 1) 연도별 정답 분포 + MLP 보정 전 예측 평균 vs 실제
year_rows = []
for split in ("train", "val", "test"):
    y = common.load_y(split)
    meta = common.load_meta(split)
    delay = meta["arr_delay_min"].to_numpy("float64")
    year = pd.to_datetime(meta["FlightDate"]).dt.year.to_numpy()
    curve = common.cdf(common.predict_split("mlp", split))
    for yr in np.unique(year):
        m = year == yr
        errs = edge_errors(curve[m], y[m], delay[m])
        so = meta["scen_origin"].to_numpy()[m]
        year_rows.append({
            "연도": yr, "분할": split, "편수": int(m.sum()),
            "정시(≤0분)": round(float((y[m] == 0).mean()), 4),
            "15분 이내": round(float((y[m] <= 1).mean()), 4),
            "60분 초과 도착": round(float(((y[m] >= 4) & (y[m] < config.NOARRIVE)).mean()), 4),
            "미도착": round(float((y[m] == config.NOARRIVE).mean()), 4),
            "출발지 날씨일 비율": round(float((so == 4).mean()), 4),
            "15분 경계 예측-실제": round(float(errs[1]), 6),
            "30분 경계 예측-실제": round(float(errs[2]), 6),
            "8경계 평균 예측-실제": round(float(np.mean(errs)), 6),
            "CDF 보정오차": round(float(np.mean(np.abs(errs))), 6),
        })
pd.DataFrame(year_rows).to_csv(OUT / "year_summary.csv", index=False, encoding="utf-8-sig")

# 2) 평가 2026 칸별 보정 후 오차 (검산: 전체 CDF 보정오차 = 0.011507)
y = common.load_y("test")
meta = common.load_meta("test")
delay = meta["arr_delay_min"].to_numpy("float64")
curve = cal.transform(common.predict_split("mlp", "test"))
total = np.mean(np.abs(edge_errors(curve, y, delay)))
print(f"검산 전체 CDF 보정오차 (보정 후, 2026) = {total:.6f}")
so, sd = meta["scen_origin"].to_numpy(), meta["scen_dest"].to_numpy()
rows = []
for o in range(5):
    for d in range(5):
        m = (so == o) & (sd == d)
        errs = edge_errors(curve[m], y[m], delay[m])
        rows.append({"출발지 유형": NAMES[o], "도착지 유형": NAMES[d], "편수": int(m.sum()),
                     "8경계 평균 예측-실제": round(float(np.mean(errs)), 6),
                     "CDF 보정오차": round(float(np.mean(np.abs(errs))), 6),
                     "낙관 경계 수": int(sum(e > 0 for e in errs)),
                     "60분 경계 예측-실제": round(float(errs[3]), 6)})
pd.DataFrame(rows).to_csv(OUT / "test_cell_calibration.csv", index=False, encoding="utf-8-sig")

# 3) 학습 구간 25칸 편수
meta = common.load_meta("train")
ct = pd.crosstab(meta["scen_origin"].map(dict(enumerate(NAMES))),
                 meta["scen_dest"].map(dict(enumerate(NAMES)))).reindex(index=NAMES, columns=NAMES)
ct.to_csv(OUT / "train_cell_counts.csv", encoding="utf-8-sig")
(ct / ct.values.sum()).round(4).to_csv(OUT / "train_cell_share.csv", encoding="utf-8-sig")
print("done")

# 4) 보장 경계에서의 보정 누적확률 (0.80 을 얼마나 넘겨서 닿는가), 평가 2026
y = common.load_y("test")
meta = common.load_meta("test")
curve = cal.transform(common.predict_split("mlp", "test"))
guaranteed, has = common.guarantee(curve)
idx = np.searchsorted(config.EDGES, guaranteed)
reached = curve[np.arange(len(curve)), idx][has]
pd.DataFrame([{
    "약속 가능 편수": int(has.sum()),
    "보장 경계에서 누적확률 평균": round(float(reached.mean()), 4),
    "중앙값": round(float(np.median(reached)), 4),
    "0.85 이상 비율": round(float((reached >= 0.85).mean()), 4),
}]).to_csv(OUT / "test_reached_prob.csv", index=False, encoding="utf-8-sig")
print(open(OUT / "test_reached_prob.csv", encoding="utf-8-sig").read())
