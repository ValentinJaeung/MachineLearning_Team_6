import sys, runpy
import numpy as np, pandas as pd
ns = runpy.run_path(str(__import__("pathlib").Path(__file__).resolve().parent / "find_example.py"), run_name="lib")
for r in [756208, 939154, 951360, 1252450, 1254214, 1472204]:
    gv, c, p = ns["grid"](r)
    print(r, ns["meta"].iloc[r][["FlightDate","Reporting_Airline","Origin","Dest","crs_dep_hour","scen_origin","scen_dest","arr_delay_min"]].tolist())
    print(pd.DataFrame(gv.reshape(5,5), index=ns["NAMES"], columns=ns["NAMES"]).to_string()); print()
