"""
Step 6 - what kind of day was it, and does that division really exist.

Reads  data/clean/{year}_{mm}.parquet   (2022-2026)
Writes data/scenario/{year}_{mm}.parquet   one row per flight, same order
       results/step06_scenario_summary.csv
       results/step06_cluster_validation.csv
       figures/step06_pca_clusters.png
       figures/step06_cluster_vs_rule.png
       figures/step06_k_selection.png

The scenario is an ASSUMPTION, not a prediction
-----------------------------------------------
At booking time nobody knows whether tomorrow will be a storm day. So the
model is never asked to guess the day type: the passenger picks it, and the
model answers "if it IS that kind of day, when do you land". Step 14 then
checks, with data, that this was the right design - that the day type is in
fact unpredictable from booking-time information.

Five types, per (airport, date)
-------------------------------
  0 정상일        no worse than this airport usually is
  1 항공사귀책일   a bad day where carrier minutes dominate
  2 연결편붕괴일   a bad day where late-aircraft minutes dominate
  3 공역혼잡일     a bad day where NAS (ATC / airspace) minutes dominate
  4 날씨일        a bad day where weather minutes dominate

Both the origin and the destination get a label. A flight can leave a healthy
airport and arrive into a collapsing one, and deciding in advance which side
matters would be a guess, so both go in and the model picks.

Two rules that keep this honest
-------------------------------
1. The label goes on the AIRPORT-DAY, never on the flight. Labelling a flight
   "this one was late because of weather" explains an outcome with the
   outcome: BTS only records a cause for flights that were already late, so
   there is nothing to compare the on-time flights against.

2. LEAVE-ONE-OUT. The day's label is built from that day's flights and the
   flight being labelled is one of them. At a 500-flight airport that is
   nothing; at a 20-flight airport the flight would be reading its own answer.
   Every aggregate therefore subtracts the focal flight first.
   Airport-days with fewer than 20 flights are not labelled at all (they get
   정상일 and applied = 0): a handful of flights cannot characterise a day.

Unsupervised check
------------------
The five types are a rule we wrote. Whether that division actually exists in
the data is a separate question, so the airport-day cause mix is put through
PCA and k-means and the clusters are compared with the rule labels (adjusted
Rand index + cross-table). k is chosen by silhouette rather than assumed.

Usage:  python src/step06_scenario.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
import plotting
from plotting import plt

YEARS = range(2022, 2027)
CAUSES = ["carrier", "late_aircraft", "nas", "weather"]
CAUSE_COLUMN = {
    "carrier": "CarrierDelay",
    "late_aircraft": "LateAircraftDelay",
    "nas": "NASDelay",
    "weather": "WeatherDelay",
}
TYPE_NAMES = config.SCENARIO_NAMES
MIN_FLIGHTS = config.SCENARIO_MIN_FLIGHTS
FAIL_FROM_CLASS = 2

READ = (["FlightDate", "Origin", "Dest", "y_class", "flight_key"]
        + list(CAUSE_COLUMN.values()))


def load_year(year: int) -> pd.DataFrame:
    parts = []
    for month in config.MONTHS:
        path = config.CLEAN_DIR / f"{year}_{month:02d}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing step 2 output: {path}")
        parts.append(pd.read_parquet(path, columns=READ))
    frame = pd.concat(parts, ignore_index=True)
    frame["failed"] = (frame["y_class"] >= FAIL_FROM_CLASS).astype("float64")
    for cause, column in CAUSE_COLUMN.items():
        frame[f"min_{cause}"] = frame[column].fillna(0.0).astype("float64")
    return frame


def label_side(frame: pd.DataFrame, airport: str) -> tuple[np.ndarray, np.ndarray,
                                                           pd.DataFrame]:
    """Leave-one-out day type for every flight, seen from `airport`'s side."""
    group = frame.groupby([airport, "FlightDate"], observed=True)

    n = group["failed"].transform("size").to_numpy().astype("float64")
    fail_sum = group["failed"].transform("sum").to_numpy()
    totals = {c: group[f"min_{c}"].transform("sum").to_numpy() for c in CAUSES}

    # --- leave the focal flight out of every aggregate --------------------- #
    own = frame["failed"].to_numpy()
    loo_n = np.maximum(n - 1.0, 1.0)
    loo_fail = (fail_sum - own) / loo_n
    loo_min = {c: totals[c] - frame[f"min_{c}"].to_numpy() for c in CAUSES}
    loo_total = np.sum([loo_min[c] for c in CAUSES], axis=0)

    shares = np.stack(
        [np.divide(loo_min[c], loo_total, out=np.zeros_like(loo_total),
                   where=loo_total > 0) for c in CAUSES], axis=1)

    # --- this airport's own usual level, within this season ----------------- #
    day_fail = group["failed"].mean().rename("day_fail").reset_index()
    usual_map = day_fail.groupby(airport, observed=True)["day_fail"].mean()
    usual = frame[airport].map(usual_map).to_numpy(dtype="float64")

    # --- which cause stands out FOR THIS AIRPORT ---------------------------- #
    # An airport beside the Rockies always has weather minutes; what marks the
    # day is the mix being unusual for that airport, so the shares are
    # standardised within the airport before the argmax.
    share_frame = pd.DataFrame(shares, columns=CAUSES, index=frame.index)
    share_frame[airport] = frame[airport].to_numpy()
    grouped = share_frame.groupby(airport, observed=True)[CAUSES]
    stats = grouped.mean().add_suffix("_mean").join(
        grouped.std().add_suffix("_std")).reset_index()
    merged = frame[[airport]].merge(stats, on=airport, how="left")
    z = np.stack([
        (shares[:, i] - merged[f"{c}_mean"].to_numpy())
        / np.maximum(merged[f"{c}_std"].to_numpy(), 1e-6)
        for i, c in enumerate(CAUSES)], axis=1)

    applied = (n >= MIN_FLIGHTS) & (loo_total > 0) & np.isfinite(usual)
    bad = applied & (loo_fail > usual)
    kind = np.where(bad, np.nanargmax(np.nan_to_num(z, nan=-np.inf), axis=1) + 1, 0)

    # airport-day level view, for the unsupervised check
    day_view = pd.DataFrame({
        "airport": frame[airport].to_numpy(),
        "FlightDate": frame["FlightDate"].to_numpy(),
        "kind": kind, "n": n, "loo_fail": loo_fail, "usual": usual,
        **{c: shares[:, i] for i, c in enumerate(CAUSES)},
    }).groupby(["airport", "FlightDate"], observed=True).first().reset_index()

    return kind.astype("int8"), applied.astype("int8"), day_view


# --------------------------------------------------------------------------- #
# Unsupervised validation: PCA + k-means on the airport-day cause mix
# --------------------------------------------------------------------------- #


def unsupervised_check(day_view: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    bad = day_view[(day_view["kind"] > 0) & (day_view["n"] >= MIN_FLIGHTS)].copy()
    if len(bad) < 500:
        raise SystemExit("too few bad airport-days for the unsupervised check")

    matrix = StandardScaler().fit_transform(bad[CAUSES].to_numpy())
    pca = PCA(n_components=2, random_state=config.SEED).fit(matrix)
    coords = pca.transform(matrix)

    # a silhouette sample, because 100k+ points is far more than it needs
    rng = np.random.default_rng(config.SEED)
    sample = rng.choice(len(matrix), size=min(20000, len(matrix)), replace=False)

    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(matrix)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(matrix[sample], km.labels_[sample])),
            "ari_vs_rule": float(adjusted_rand_score(bad["kind"], km.labels_)),
        })
    scan = pd.DataFrame(rows)

    best_k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=config.SEED).fit(matrix)
    bad["cluster"] = km.labels_
    ari_at_4 = float(scan.loc[scan["k"] == 4, "ari_vs_rule"].iloc[0])

    print("\n=== Unsupervised check: PCA + k-means on the airport-day cause mix ===")
    print(f"bad airport-days: {len(bad):,}   "
          f"PCA explained variance: {pca.explained_variance_ratio_.round(3).tolist()} "
          f"(sum {pca.explained_variance_ratio_.sum():.3f})")
    print(scan.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsilhouette picks k = {best_k}; "
          f"ARI against the four rule types at k=4 is {ari_at_4:.3f}")

    table = pd.crosstab(bad["kind"].map(dict(enumerate(TYPE_NAMES))), bad["cluster"])
    print("\nrule type (rows) vs cluster (columns), k = {}:".format(best_k))
    print(table.to_string())

    # --- figures ------------------------------------------------------------ #
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    show = rng.choice(len(bad), size=min(12000, len(bad)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (values, title) in zip(axes, [
        (bad["cluster"].to_numpy(), f"k-means clusters (k={best_k})"),
        (bad["kind"].to_numpy(), "rule-based day types"),
    ]):
        for value in sorted(np.unique(values)):
            pick = show[values[show] == value]
            name = (TYPE_NAMES[value] if title.startswith("rule")
                    else f"cluster {value}")
            ax.scatter(coords[pick, 0], coords[pick, 1], s=4, alpha=0.35, label=name)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.suptitle("Step 6 — 공항-날짜 지연원인 구성비의 PCA 평면", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step06_pca_clusters.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    normed = table.div(table.sum(axis=1), axis=0)
    im = ax.imshow(normed.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(normed.shape[1]),
                  [f"cluster {c}" for c in normed.columns], fontsize=9)
    ax.set_yticks(range(normed.shape[0]), normed.index, fontsize=9)
    for i in range(normed.shape[0]):
        for j in range(normed.shape[1]):
            v = normed.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else "black")
    ax.set_title(f"규칙 유형 vs 군집 (행 기준 비율, k={best_k}, ARI={ari_at_4:.3f} @k=4)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step06_cluster_vs_rule.png", dpi=140)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(scan["k"], scan["inertia"], "o-", color="#4c6ef5", label="inertia")
    ax1.set_xlabel("k"); ax1.set_ylabel("inertia", color="#4c6ef5")
    ax2 = ax1.twinx()
    ax2.plot(scan["k"], scan["silhouette"], "s--", color="#e8590c", label="silhouette")
    ax2.set_ylabel("silhouette", color="#e8590c")
    ax1.axvline(best_k, color="grey", ls=":", lw=1)
    ax1.set_title("Step 6 — 군집 개수 선택 (엘보 + 실루엣)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step06_k_selection.png", dpi=140)
    plt.close(fig)

    scan["best_k_by_silhouette"] = best_k
    scan["pca_evr_1"] = pca.explained_variance_ratio_[0]
    scan["pca_evr_2"] = pca.explained_variance_ratio_[1]
    scan["n_bad_airport_days"] = len(bad)
    scan.to_csv(config.RESULTS_DIR / "step06_cluster_validation.csv",
                index=False, encoding="utf-8-sig")
    table.to_csv(config.RESULTS_DIR / "step06_cluster_crosstab.csv",
                 encoding="utf-8-sig")
    return scan


def main() -> int:
    config.set_seeds()
    config.ensure_dirs()
    plotting.setup()

    rows, train_day_views = [], []
    for year in YEARS:
        frame = load_year(year)
        out = pd.DataFrame(index=frame.index)
        for side, airport in (("origin", "Origin"), ("dest", "Dest")):
            kind, applied, day_view = label_side(frame, airport)
            out[f"scen_{side}"] = kind
            out[f"scen_{side}_applied"] = applied
            if side == "origin" and year in config.SPLIT_YEARS["train"]:
                train_day_views.append(day_view)

        # written back month by month, in the same row order as data/clean/
        start = 0
        for month in config.MONTHS:
            length = int((frame["FlightDate"].dt.month == month).sum())
            piece = out.iloc[start:start + length]
            piece.to_parquet(config.SCENARIO_DIR / f"{year}_{month:02d}.parquet",
                             index=False)
            start += length
        if start != len(out):
            raise SystemExit(f"{year}: month split did not cover every row")

        share = out["scen_origin"].value_counts(normalize=True).sort_index()
        row = {"year": year, "rows": len(out)}
        row.update({TYPE_NAMES[i]: round(float(share.get(i, 0.0)), 4)
                    for i in range(5)})
        row["미적용"] = round(float(1 - out["scen_origin_applied"].mean()), 4)
        rows.append(row)
        print(f"[{year}] " + "  ".join(f"{TYPE_NAMES[i]} {share.get(i, 0):.3f}"
                                       for i in range(5))
              + f"   미적용 {row['미적용']:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(config.RESULTS_DIR / "step06_scenario_summary.csv",
                   index=False, encoding="utf-8-sig")
    print("\n=== Day type distribution, origin side ===")
    print(summary.to_string(index=False))

    unsupervised_check(pd.concat(train_day_views, ignore_index=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
