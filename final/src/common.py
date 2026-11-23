"""여러 단계가 같이 쓰는 함수들 — 데이터 읽기, 지표, 보장 시각 계산.

여기 있는 함수 하나가 '서비스가 답을 만드는 방식'을 그대로 구현한다.
step08(보정)과 step09(검증)이 같은 함수를 쓰므로 둘이 어긋날 수 없다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

import config

N_EDGES = len(config.EDGES)          # 8
TOLERANCE = 1e-9                     # 누적합의 부동소수점 오차 허용폭


# --------------------------------------------------------------------------- #
# 읽기
# --------------------------------------------------------------------------- #
def load_tree(split: str) -> pd.DataFrame:
    return pd.read_parquet(config.PREP_DIR / f"{split}_tree.parquet")


def load_linear(split: str) -> sparse.csr_matrix:
    return sparse.load_npz(config.PREP_DIR / f"{split}_linear.npz").tocsr()


def load_y(split: str) -> np.ndarray:
    return np.load(config.PREP_DIR / f"{split}_y.npy")


def load_meta(split: str) -> pd.DataFrame:
    return pd.read_parquet(config.PREP_DIR / f"{split}_meta.parquet")


# --------------------------------------------------------------------------- #
# step07 이 고른 모델로 예측하기 — step08 과 step09 가 같은 함수를 쓴다
# --------------------------------------------------------------------------- #
def winner() -> dict:
    """results/step07_decision.json 의 채택 결과를 읽는다."""
    import json
    return json.loads((config.RESULTS_DIR / "step07_decision.json")
                      .read_text(encoding="utf-8"))


def predict_split(family: str, split: str) -> np.ndarray:
    """채택된 계열의 모델로 한 분할을 예측한다 (10구간 확률)."""
    if family == "xgboost":
        import json
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(config.MODEL_DIR / "xgboost_best.json")
        span = tuple(json.loads(
            (config.MODEL_DIR / "xgboost_best_span.json")
            .read_text(encoding="utf-8"))["iteration_range"])
        matrix = xgb.DMatrix(load_tree(split), enable_categorical=True)
        return booster.predict(matrix, iteration_range=span).astype("float64")

    if family == "logistic":
        import joblib
        model = joblib.load(config.MODEL_DIR / "logistic_best.joblib")
        return model.predict_proba(load_linear(split)).astype("float64")

    if family == "mlp":
        import torch
        from torch import nn
        bundle = torch.load(config.MODEL_DIR / "mlp_best.pt", weights_only=False)
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
        matrix = load_linear(split)
        out = np.empty((matrix.shape[0], config.N_CLASS), dtype="float64")
        with torch.no_grad():
            for start in range(0, matrix.shape[0], config.MLP_BATCH * 2):
                stop = min(start + config.MLP_BATCH * 2, matrix.shape[0])
                dense = torch.from_numpy(matrix[start:stop].toarray())
                out[start:stop] = torch.softmax(model(dense), dim=1).numpy()
        return out

    raise ValueError(f"알 수 없는 계열: {family}")


# --------------------------------------------------------------------------- #
# 지표
# --------------------------------------------------------------------------- #
def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass log loss. 정답 칸에 준 확률이 높을수록 작아진다."""
    p = np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)
    return float(-np.log(p).mean())


def cdf(probs: np.ndarray) -> np.ndarray:
    """10구간 확률 → 경계 8개에서의 '그 시간 안에 도착할 확률'.

    class 8(300분+)과 class 9(미도착)는 어느 경계 안에도 들어가지 않으므로
    누적에 넣지 않는다. 그래서 곡선의 상한이 1.0 보다 낮고, 모자란 만큼이
    '그 시간 안에 도착하지 못할 확률' 이다.
    """
    return probs[:, :N_EDGES].cumsum(axis=1)


def cdf_calibration_error(curve: np.ndarray, y: np.ndarray,
                          delay: np.ndarray) -> float:
    """경계마다 '모델이 말한 평균 확률 − 실제 비율' 의 절댓값 평균."""
    arrived = y < config.NOARRIVE
    errors = []
    for i, edge in enumerate(config.EDGES):
        actual = (arrived & (delay <= edge)).mean()
        errors.append(abs(curve[:, i].mean() - actual))
    return float(np.mean(errors))


# --------------------------------------------------------------------------- #
# 서비스가 답을 만드는 방식
# --------------------------------------------------------------------------- #
def guarantee(curve: np.ndarray,
              confidence: float = config.CONFIDENCE) -> tuple[np.ndarray, np.ndarray]:
    """누적 곡선에서 요구 확률에 처음 닿는 경계를 찾는다.

    반환: (보장 지연[분], 보장 가능 여부)
    곡선이 요구 확률에 끝까지 닿지 못하면 보장 지연을 주지 않는다('보장 불가').

    TOLERANCE 를 두는 이유: 누적합은 부동소수점이라 0.5+0.2+0.1 이 0.8 이 아니라
    0.7999999999999999 로 나온다. 그대로 비교하면 정확히 경계에 닿은 편이 한 칸
    뒤로 밀려 보장 시각이 괜히 늦어진다.
    """
    reach = curve >= confidence - TOLERANCE
    has = reach.any(axis=1)
    first = np.where(has, reach.argmax(axis=1), 0)
    return np.asarray(config.EDGES)[first], has


def promise_table(curve: np.ndarray, y: np.ndarray, delay: np.ndarray,
                  confidence: float = config.CONFIDENCE) -> dict:
    """약속 이행률과 함께 봐야 하는 네 숫자. `curve` 는 누적 곡선(경계 8개).

    이행률만 보면 보장 시각을 뒤로 밀어 얼마든지 올릴 수 있다. 그래서 평균
    보장 지연과 보장 불가 비율을 항상 같이 낸다.
    """
    guaranteed, has = guarantee(curve, confidence)
    arrived = y < config.NOARRIVE
    # 보장을 받은 편만 분모에 넣는다. 결항·미도착은 '못 지킴'.
    kept = has & arrived & (delay <= guaranteed)
    n = int(has.sum())
    rate = float(kept[has].mean()) if n else float("nan")
    return {
        "n": int(len(y)),
        "약속가능편수": n,
        "이행률": round(rate, 6),
        "이행률오차": round(rate - confidence, 6),
        "평균보장지연": round(float(guaranteed[has].mean()), 2) if n else float("nan"),
        "보장불가비율": round(float((~has).mean()), 6),
    }


# --------------------------------------------------------------------------- #
# 확률 보정
# --------------------------------------------------------------------------- #
class CdfCalibrator:
    """경계 8개마다 isotonic regression 을 하나씩 걸어 누적확률을 교정한다.

    모델을 버리지 않고 **읽는 값만** 고친다. 온도계를 바꾸는 대신 환산표를
    만드는 것과 같다. 반드시 모델이 학습하지 않은 구간(검증)에서 적합한다.
    """

    def __init__(self) -> None:
        self.models: list = []

    def fit(self, probs: np.ndarray, y: np.ndarray,
            delay: np.ndarray) -> "CdfCalibrator":
        from sklearn.isotonic import IsotonicRegression

        curve = cdf(probs)
        arrived = y < config.NOARRIVE
        self.models = []
        for i, edge in enumerate(config.EDGES):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(curve[:, i], (arrived & (delay <= edge)).astype("float64"))
            self.models.append(iso)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """보정된 **누적 곡선**(경계 8개)을 돌려준다.

        경계마다 따로 고치면 누적이 거꾸로 가는 칸이 생길 수 있으므로
        단조증가하도록 정렬한다.
        """
        curve = cdf(probs)
        out = np.empty_like(curve)
        for i, iso in enumerate(self.models):
            out[:, i] = iso.predict(curve[:, i])
        return np.maximum.accumulate(out, axis=1)

    def transform_probs(self, probs: np.ndarray) -> np.ndarray:
        """보정된 누적 곡선을 다시 10구간 확률로 되돌린다.

        log loss 처럼 10칸 분포가 필요한 지표를 보정 전후로 비교하기 위한 것이다.
        경계 안쪽 8칸은 보정된 누적의 차이로 얻고, 남은 확률(300분+ 와 미도착)은
        보정 전의 두 칸 비율대로 나눈다 — 두 칸은 경계 안에 들어가지 않으므로
        보정기가 직접 다루지 않는다.
        """
        curve = self.transform(probs)
        out = np.zeros_like(probs)
        out[:, 0] = curve[:, 0]
        out[:, 1:N_EDGES] = np.diff(curve, axis=1)
        out[:, :N_EDGES] = np.clip(out[:, :N_EDGES], 0.0, None)

        rest = np.clip(1.0 - out[:, :N_EDGES].sum(axis=1), 0.0, None)
        tail = probs[:, N_EDGES:].sum(axis=1)
        ratio = np.divide(probs[:, N_EDGES:], tail[:, None],
                          out=np.full_like(probs[:, N_EDGES:], 0.5),
                          where=tail[:, None] > 0)
        out[:, N_EDGES:] = ratio * rest[:, None]
        return out / out.sum(axis=1, keepdims=True)
