"""step05 — 설계행렬을 두 벌 만든다. 트리용과 선형·MLP용.

읽기  data/scen/{year}.parquet, results/step04_airport_list.json
쓰기  data/prep/{split}_tree.parquet     13열 — XGBoost 가 그대로 먹는다
      data/prep/{split}_linear.npz       희소 원-핫 — 로지스틱·MLP 용
      data/prep/{split}_y.npy            정답 (10구간)
      data/prep/{split}_meta.parquet     채점에 필요한 열 (모델 입력 아님)
      data/prep/preprocessor.joblib      학습 구간에서만 적합한 변환기
      results/step05_columns.json        열 이름과 개수
      results/step05_shapes.csv          분할별 크기

같은 13열인데 모델에 따라 폭이 다르다
--------------------------------------
XGBoost 는 "ATL" 같은 범주를 그대로 먹는다. 반면 **선형 모델과 MLP 는 숫자만**
받으므로, 공항에 1·2·3 번호를 매기면 "ATL 과 DEN 사이에 DFW 가 있다" 는 거짓
순서가 생긴다. 그래서 범주 하나하나를 0/1 열로 펼친다. 변수의 개수는 13개로
같고 **모델이 받는 열의 수만** 달라진다.

시각을 어느 쪽에 두는가
-----------------------
예정 출발·도착 시각(시)은 트리에는 **숫자**로 준다 — 트리는 "14시 이전/이후"
같은 순서 조건을 스스로 만들 수 있다. 선형·MLP 에는 **0/1 로 펼쳐** 준다 —
직선 하나로는 step04 에서 본 하루 동안의 비선형 모양을 담을 수 없다.

누출을 막는 규칙
----------------
결측 채움값(중앙값), 표준화의 평균·표준편차, 원-핫의 범주 목록은 **모두 학습
구간에서만** 정하고 그 값을 검증·평가에 그대로 적용한다. 검증·평가 데이터를
보고 정하면 그 구간의 정보가 변환기에 새어 든다.

실행  python src/step05_preprocess.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

# 13개 입력 변수 — 모델에 들어가는 전부
TREE_CATEGORICAL = ["Reporting_Airline", "Origin", "Dest", "Month", "DayOfWeek",
                    "scen_origin", "scen_dest"]
TREE_NUMERIC = ["crs_dep_hour", "crs_arr_hour", "Distance", "CRSElapsedTime",
                "dest_arr_load", "days_to_nearest_holiday"]
TREE_COLUMNS = TREE_CATEGORICAL + TREE_NUMERIC

# 선형·MLP: 시각까지 0/1 로 펼친다
ONEHOT = ["Reporting_Airline", "Origin", "Dest", "DayOfWeek", "Month",
          "crs_dep_hour", "crs_arr_hour", "scen_origin", "scen_dest"]
SCALED = ["Distance", "CRSElapsedTime", "dest_arr_load",
          "days_to_nearest_holiday"]

META = ["FlightDate", "Origin", "Dest", "Reporting_Airline",
        "crs_dep_hour", "arr_delay_min", "y_class", "scen_origin", "scen_dest"]

READ = sorted(set(TREE_COLUMNS + ONEHOT + SCALED + META))


def load(split: str, airports: set[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(config.SCEN_DIR / f"{year}.parquet", columns=READ)
             for year in config.SPLIT_YEARS[split]]
    frame = pd.concat(parts, ignore_index=True)
    # 학습 연도에서 운항이 적었던 공항은 '기타' 로 묶는다 (목록은 step04 가 정함)
    for column in ("Origin", "Dest"):
        frame[column] = frame[column].where(
            frame[column].isin(airports), config.OTHER_LABEL)
    return frame


def build_preprocessor() -> ColumnTransformer:
    """원-핫 + 표준화. 범주 목록과 통계는 fit 에서 학습 구간만 보고 정해진다."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), ONEHOT),
         ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), SCALED)],
        sparse_threshold=1.0, remainder="drop", n_jobs=1)


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()

    info = json.loads((config.RESULTS_DIR / "step04_airport_list.json")
                      .read_text(encoding="utf-8"))
    airports = set(info["airports"])
    print(f"공항 {len(airports)}개 + '{config.OTHER_LABEL}'\n")

    pre = build_preprocessor()
    shapes = []

    for split in config.SPLITS:
        frame = load(split, airports)

        if split == "train":
            print("변환기를 학습 구간에서 적합 ...", flush=True)
            pre.fit(frame)
            joblib.dump(pre, config.MODEL_DIR / "preprocessor.joblib")
            names = list(pre.get_feature_names_out())
            (config.RESULTS_DIR / "step05_columns.json").write_text(
                json.dumps({
                    "n_variables": len(TREE_COLUMNS),
                    "tree_categorical": TREE_CATEGORICAL,
                    "tree_numeric": TREE_NUMERIC,
                    "n_tree_columns": len(TREE_COLUMNS),
                    "n_linear_columns": len(names),
                    "linear_columns": names,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  변수 {len(TREE_COLUMNS)}개 → 트리 {len(TREE_COLUMNS)}열 / "
                  f"선형·MLP {len(names)}열\n")

        # --- 트리용: 범주형은 category dtype 으로 ------------------------- #
        tree = frame[TREE_COLUMNS].copy()
        for column in TREE_CATEGORICAL:
            tree[column] = tree[column].astype(str).astype("category")
        tree.to_parquet(config.PREP_DIR / f"{split}_tree.parquet", index=False)

        # --- 선형·MLP용: 희소 원-핫 -------------------------------------- #
        linear = sparse.csr_matrix(pre.transform(frame), dtype=np.float32)
        sparse.save_npz(config.PREP_DIR / f"{split}_linear.npz", linear)

        np.save(config.PREP_DIR / f"{split}_y.npy",
                frame["y_class"].to_numpy(dtype="int64"))
        frame[META].to_parquet(config.PREP_DIR / f"{split}_meta.parquet",
                               index=False)

        shapes.append({
            "split": split, "편수": len(frame),
            "트리 열": len(TREE_COLUMNS), "선형 열": linear.shape[1],
            "희소 nnz": int(linear.nnz),
            "밀도": round(linear.nnz / (linear.shape[0] * linear.shape[1]), 5),
        })
        print(f"  {split:<6} {len(frame):>9,} 편  "
              f"linear {linear.shape[1]}열 nnz {linear.nnz:,}", flush=True)
        del frame, tree, linear

    table = pd.DataFrame(shapes)
    table.to_csv(config.RESULTS_DIR / "step05_shapes.csv",
                 index=False, encoding="utf-8-sig")
    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
