"""6.4 실제 데이터 예시: 2026 평가 항공편 중 '정답 조합'의 80% 보장 지연과 실제 도착 지연이
가장 가까운 편을 찾고, 그 편의 25조합 결과를 만든다.

정답 조합 결과 = 그 편의 실제 시나리오(출발지×도착지)로 예측한 값 = step09 와 같은
predict_split("mlp", "test") → calibrator.transform → guarantee.
25조합 = step09_service.example_flight 와 같은 방식(시나리오 one-hot 만 바꿔 MLP 재예측).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "final" / "src"))
import common  # noqa: E402
import config  # noqa: E402

NAMES = config.SCENARIO_NAMES
cal = joblib.load(config.MODEL_DIR / "calibrator.joblib")
PAIRS = [(o, d) for o in range(5) for d in range(5)]

# ---------- 1) 정답 조합 결과로 후보 고르기 ----------
meta = common.load_meta("test")
y = common.load_y("test")
delay = meta["arr_delay_min"].to_numpy("float64")
curve = cal.transform(common.predict_split("mlp", "test"))
g, has = common.guarantee(curve)
arrived = y < config.NOARRIVE
slack = g - delay                        # 보장 지연 - 실제 지연 (0 이상이면 약속 지킴)
normal = (meta["scen_origin"] == 0) & (meta["scen_dest"] == 0)

cand = has & arrived & (slack >= 0) & (g >= 30) & ~normal.to_numpy()
print("후보(약속 지킴, 보장≥30분, 정상×정상 제외):", int(cand.sum()))
best_slack = slack[cand].min()
print("최소 차이(분):", best_slack)
pool = np.where(cand & (slack <= best_slack + 1))[0]
print("최소 차이 +1분 이내 편수:", len(pool))

# ---------- 2) 후보들의 25조합 결과 ----------
columns = json.loads((config.RESULTS_DIR / "step05_columns.json").read_text(encoding="utf-8"))
names = columns["linear_columns"]
slots = {side: [names.index(f"cat__scen_{side}_{k}") for k in range(5)]
         for side in ("origin", "dest")}
X = common.load_linear("test")

bundle = torch.load(config.MODEL_DIR / "mlp_best.pt", weights_only=False)
layers, width = [], bundle["n_features"]
for i, size in enumerate(bundle["hidden"]):
    layers += [nn.Linear(width, size), nn.ReLU()]
    if i < len(bundle["hidden"]) - 1:
        layers.append(nn.Dropout(config.MLP_DROPOUT))
    width = size
layers.append(nn.Linear(width, config.N_CLASS))
model = nn.Sequential(*layers)
model.load_state_dict(bundle["state_dict"])
model.eval()


def grid(row):
    m = X[[row] * 25].toarray().astype("float32")
    for i, (o, d) in enumerate(PAIRS):
        m[i, slots["origin"]] = 0
        m[i, slots["dest"]] = 0
        m[i, slots["origin"][o]] = 1
        m[i, slots["dest"][d]] = 1
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(m)), dim=1).numpy().astype("float64")
    c = cal.transform(p)
    gg, hh = common.guarantee(c)
    return np.where(hh, gg, np.nan), c, p


rows = []
for r in pool:
    gv, _, _ = grid(r)
    k = int(meta["scen_origin"].iat[r]) * 5 + int(meta["scen_dest"].iat[r])
    rows.append({
        "row": int(r), "날짜": meta["FlightDate"].iat[r].date(), "항공사": meta["Reporting_Airline"].iat[r],
        "출발": meta["Origin"].iat[r], "도착": meta["Dest"].iat[r], "출발시": int(meta["crs_dep_hour"].iat[r]),
        "정답조합": f'{NAMES[int(meta["scen_origin"].iat[r])]}×{NAMES[int(meta["scen_dest"].iat[r])]}',
        "보장(분)": g[r], "실제(분)": delay[r], "차이": slack[r],
        "정답조합_재예측": gv[k],                    # 검산: 1)의 보장과 같아야 함
        "서로 다른 보장값 수": len(set(gv[~np.isnan(gv)])),
        "보장불가 칸": int(np.isnan(gv).sum()),
        "최소": np.nanmin(gv), "최대": np.nanmax(gv),
    })
res = pd.DataFrame(rows)
assert (res["정답조합_재예측"] == res["보장(분)"]).all(), "재예측 불일치"
res = res.sort_values(["차이", "서로 다른 보장값 수", "보장불가 칸"],
                      ascending=[True, False, True])
res.to_csv(OUT / "candidates.csv", index=False, encoding="utf-8-sig")
print(res.head(25).to_string())
