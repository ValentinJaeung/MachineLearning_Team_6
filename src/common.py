"""
Shared pieces for the modelling steps (10 onwards): loading, metrics, calibration.

Why these metrics and not accuracy
----------------------------------
The product is a probability. "Which single bin is most likely" is never
shown to anyone, so accuracy would score something the service does not sell.
Each metric here asks a question the service actually depends on:

  log_loss      is the whole distribution right?            (the training loss)
  brier         same question, less punishing on the tail
  cdf_ece       when the model says 90%, is it 90%?         (the promise)
  coverage_90   of the flights given a 90% guarantee, how many kept it?
  auc_late      can it still rank a late flight above an on-time one?

cdf_ece is the one that decides the product. A model can win on log loss and
still promise 90% while delivering 85%, and then the service is lying.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

import config

EDGES = config.EDGES
N_CLASS = config.N_CLASS
NOARRIVE = config.NOARRIVE
N_DELAY = config.N_DELAY_BINS


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def feature_names() -> dict:
    return json.loads((config.PREP_DIR / "feature_names.json")
                      .read_text(encoding="utf-8"))


def load_split(split: str, limit: int = 0, drop: list[str] | None = None,
               with_scenario: bool = False) -> dict:
    """Both design matrices, the target, and the meta frame for one split.

    drop          feature names to remove from BOTH encodings (step 10 uses
                  this to add and remove blocks on the same footing)
    with_scenario attach the step 6 columns; they are stored apart because
                  they are NOT known at booking time (step 14 only)
    limit         take every n-th row, for a quick smoke run
    """
    drop = drop or []
    names = feature_names()
    columns = names["linear_columns"]
    keep = np.array([i for i, c in enumerate(columns)
                     if c.split("__", 1)[-1].rsplit("_", 1)[0] not in drop
                     and c.split("__", 1)[-1] not in drop])

    linear = sparse.load_npz(config.PREP_DIR / f"{split}_linear.npz").tocsr()[:, keep]
    xgb = pd.read_parquet(config.PREP_DIR / f"{split}_xgb.parquet")
    xgb = xgb.drop(columns=[c for c in drop if c in xgb.columns])
    meta = pd.read_parquet(config.PREP_DIR / f"{split}_meta.parquet")
    cls = np.load(config.PREP_DIR / f"{split}_y.npy").astype("int64")
    delay = meta["arr_delay_min"].to_numpy(dtype="float64")

    if with_scenario:
        scen = pd.read_parquet(config.PREP_DIR / f"{split}_scen.parquet")
        for col in ("scen_origin", "scen_dest"):
            xgb[col] = pd.Categorical(scen[col].astype(int).astype(str),
                                      categories=[str(i) for i in range(5)])

    if limit:
        stride = max(1, len(cls) // limit)
        index = np.arange(0, len(cls), stride)
        linear = linear[index]
        xgb = xgb.iloc[index].reset_index(drop=True)
        meta = meta.iloc[index].reset_index(drop=True)
        cls, delay = cls[index], delay[index]

    return {"X_linear": linear, "X_xgb": xgb, "meta": meta,
            "cls": cls, "delay": delay,
            "linear_columns": [columns[i] for i in keep]}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def evaluate(probs: np.ndarray, cls: np.ndarray, delay: np.ndarray) -> dict:
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n = len(cls)

    logloss = float(-np.log(probs[np.arange(n), cls]).mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), cls] = 1.0
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())

    cdf = probs.cumsum(axis=1)
    arrived = cls < NOARRIVE

    rel, ece = {}, 0.0
    for i, edge in enumerate(EDGES):
        pred = float(cdf[:, i].mean())
        actual = float((arrived & (delay <= edge)).mean())
        rel[f"<= {edge}분"] = {"예측": round(pred, 4), "실제": round(actual, 4)}
        ece += abs(pred - actual)
    ece /= len(EDGES)

    # the promise, read the way the service reads it
    reach = cdf[:, :len(EDGES)] >= 0.90
    has = reach.any(axis=1)
    first = np.where(has, reach.argmax(axis=1), len(EDGES) - 1)
    promised = np.asarray(EDGES)[first]
    kept = arrived & (delay <= promised)
    coverage = float(kept[has].mean()) if has.any() else float("nan")

    late = (~arrived) | (delay > 15)
    ranks = rankdata(1.0 - cdf[:, 1])
    pos, neg = int(late.sum()), int(n - late.sum())
    auc = float((ranks[late].sum() - pos * (pos + 1) / 2) / (pos * neg))

    return {"log_loss": round(logloss, 5), "brier": round(brier, 5),
            "cdf_ece": round(ece, 5), "coverage_90": round(coverage, 4),
            "coverage_90_share": round(float(has.mean()), 4),
            "auc_late": round(auc, 5), "reliability": rel}


def flat(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "reliability"}


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


class CdfCalibrator:
    """Isotonic regression on every CDF edge, plus one on "arrives at all".

    The conversion table of the thermometer analogy: the model is not thrown
    away, the reading is corrected. Fitted on data the model never trained on.
    """

    def __init__(self) -> None:
        self.edge_models: list = []
        self.arrive_model = None

    def fit(self, probs, cls, delay) -> "CdfCalibrator":
        from sklearn.isotonic import IsotonicRegression

        cdf = probs.cumsum(axis=1)
        arrived = cls < NOARRIVE
        for i, edge in enumerate(EDGES):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(cdf[:, i], (arrived & (delay <= edge)).astype(float))
            self.edge_models.append(iso)
        self.arrive_model = IsotonicRegression(y_min=0.0, y_max=1.0,
                                               out_of_bounds="clip")
        self.arrive_model.fit(cdf[:, N_DELAY - 1], arrived.astype(float))
        return self

    def transform(self, probs):
        cdf = probs.cumsum(axis=1)
        out = np.empty((len(probs), len(EDGES)), dtype="float64")
        for i, iso in enumerate(self.edge_models):
            out[:, i] = iso.predict(cdf[:, i])
        arrive = self.arrive_model.predict(cdf[:, N_DELAY - 1])
        out = np.maximum.accumulate(out, axis=1)        # a CDF cannot go down
        arrive = np.clip(np.maximum(arrive, out[:, -1]), 1e-6, 1.0)

        new = np.zeros_like(probs)
        new[:, 0] = out[:, 0]
        new[:, 1:len(EDGES)] = np.diff(out, axis=1)
        new[:, N_DELAY - 1] = arrive - out[:, -1]
        new[:, NOARRIVE] = 1.0 - arrive
        new = np.clip(new, 1e-9, None)
        return new / new.sum(axis=1, keepdims=True)


def prior_shift(probs: np.ndarray, source: np.ndarray,
                target: np.ndarray) -> np.ndarray:
    """Re-weight a distribution from one class prior to another.

    Step 12 finds that most of the miscalibration is a LEVEL shift between
    seasons rather than a change of shape, and one line of prior correction
    removes most of it.
    """
    ratio = np.clip(target, 1e-9, None) / np.clip(source, 1e-9, None)
    out = probs * ratio
    return out / out.sum(axis=1, keepdims=True)


def guaranteed_minutes(probs: np.ndarray, level: float = 0.90) -> np.ndarray:
    """Minutes of delay by which the flight arrives with probability `level`.

    NaN when the distribution never reaches `level` even at the last edge.
    The service shows "보장 불가" there rather than inventing a time.
    """
    cdf = probs.cumsum(axis=1)[:, :len(EDGES)]
    out = np.full(len(probs), np.nan)
    ok = cdf[:, -1] >= level
    for i in np.flatnonzero(ok):
        out[i] = np.interp(level, cdf[i], EDGES)
    return out


def deadline_probability(probs: np.ndarray, slack_min: np.ndarray) -> np.ndarray:
    """P(arrives within `slack_min` minutes of the scheduled arrival)."""
    cdf = probs.cumsum(axis=1)[:, :len(EDGES)]
    out = np.empty(len(probs))
    edges = np.asarray(EDGES, dtype="float64")
    for i in range(len(probs)):
        out[i] = np.interp(slack_min[i], edges, cdf[i], left=0.0,
                           right=float(cdf[i, -1]))
    return out
