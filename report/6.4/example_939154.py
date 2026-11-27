"""선택한 예시 편(평가 2026 행 939154)의 25조합 결과와 정답 조합의 누적확률을 저장한다."""
import runpy
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ns = runpy.run_path(str(HERE / "find_example.py"), run_name="lib")
ROW = 939154
meta, NAMES, common, config = ns["meta"], ns["NAMES"], ns["common"], ns["config"]
gv, c, p = ns["grid"](ROW)
info = meta.iloc[ROW]
tree = common.load_tree("test").iloc[ROW]
o, d = int(info["scen_origin"]), int(info["scen_dest"])
k = o * 5 + d

pd.DataFrame([{
    "날짜": info["FlightDate"].date(), "항공사": info["Reporting_Airline"],
    "출발": info["Origin"], "도착": info["Dest"],
    "예정 출발 시": int(info["crs_dep_hour"]), "예정 도착 시": tree["crs_arr_hour"],
    "거리(마일)": tree["Distance"], "예정 비행시간(분)": tree["CRSElapsedTime"],
    "실제 출발지 유형": NAMES[o], "실제 도착지 유형": NAMES[d],
    "80% 보장 지연(분)": gv[k], "실제 도착 지연(분)": info["arr_delay_min"],
    "정답 구간 class": int(ns["y"][ROW]),
}]).to_csv(HERE / "example_flight.csv", index=False, encoding="utf-8-sig")

pd.DataFrame(gv.reshape(5, 5), index=NAMES, columns=NAMES).rename_axis("출발지\도착지") \
  .to_csv(HERE / "example_grid.csv", encoding="utf-8-sig")

pd.DataFrame({"경계(분)": config.EDGES, "보정 누적확률": c[k].round(4)}) \
  .to_csv(HERE / "example_true_curve.csv", index=False, encoding="utf-8-sig")

print(open(HERE / "example_flight.csv", encoding="utf-8-sig").read())
print(open(HERE / "example_grid.csv", encoding="utf-8-sig").read())
print(open(HERE / "example_true_curve.csv", encoding="utf-8-sig").read())
print("검산 - step09 경로의 누적곡선:", ns["curve"][ROW].round(4))
