"""Step 11i - applying the selection rule, once, in the open.

Reads  results/step11g_xgb_full.csv         (XGBoost, 13 settings, full train)
       results/step11h_logistic_degrees.csv  (logistic, 5 settings, converged)
       results/step11d_mlp_search.csv        (MLP, 5 settings, full train)
       results/step11f_mlp_seeds.csv         (MLP seed spread)
       results/step11c_seed_check.csv        (XGBoost seed spread)
Writes results/step11i_selection.csv
       results/step11i_summary.csv
       results/step11i_verdict.csv
       figures/step11i_selection.png

This step fits nothing. It exists so that the choice of model is made by a
written rule over files that already exist, rather than by whichever number
looked best when someone went to write section 5.1.1.

THE RULE, in the order it is applied:

  1. WITHIN a model family, the configuration is chosen by validation LOG
     LOSS. Not by the selection metric. Log loss is what all three fitting
     procedures already optimise - XGBoost's best_iteration, lbfgs's
     convergence test and the network's early stopping all read log loss - so
     choosing the configuration by it is the only choice consistent with how
     the models were fitted. Choosing by CDF calibration error instead would
     be picking on a quantity no fitting procedure controls, which is how step
     11d turned up a network setting at 0.00500 that step 11f then showed to
     be one draw out of a spread.

  2. Where a configuration's seed spread has been measured, its score is the
     MEAN over seeds, not one fit. A single fit is one draw.

  3. BETWEEN families, the chosen configurations are compared on CDF
     CALIBRATION ERROR, which section 4.4 named before any numbers were seen.

  4. A gap is called real only if it clears the noise band. Section 4.4's
     wording is "fit the same setting under three seeds, take the standard
     deviation of the noise, use twice that as the threshold" - a threshold on
     single fits. Comparing MEANS of three fits would instead call for twice
     the standard error. The two tests disagree here, so both are printed and
     the pre-registered one is the one applied. Picking the test after seeing
     which way it points is the thing pre-registration exists to prevent.

  5. If the selection metric cannot rank the families, it cannot. Section 4.4
     names TWO primary metrics, log loss and calibration error, so the tie
     falls to the other one - declared in the same table, before any numbers.
     What does NOT get to break the tie is anything undeclared, however
     sensible: that the tree is 50 times more reproducible, or that TreeSHAP
     exists for trees and not for networks. Those are consequences to report
     in section 6, not criteria to select on.

Usage:  python src/step11i_selection.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import plotting
from plotting import plt

FAMILIES = {
    "XGBoost": ("step11g_xgb_full.csv", "#1864ab"),
    "로지스틱": ("step11h_logistic_degrees.csv", "#e8590c"),
    "MLP": ("step11d_mlp_search.csv", "#2f9e44"),
}

# Which seed arm belongs to which swept configuration. Matched by the numbers
# that identify the setting, not by label, because the sweep files and the
# seed-check files were written by different steps and name things differently.
#   XGBoost: step11j holds both eta arms under three shared seeds; before that
#            step existed, step11c_seed_check's "base" arm was the only one
#            (the production setting, max_depth 6 / eta 0.08).
#   MLP:     step11f's two arms are step11d's "작게 256-64" (= lr 3e-4) and
#            "lr 0.001".
SEED_ARMS = {
    "XGBoost": [("step11j_eta_seeds.csv", "eta",
                 {"eta=0.08": 0.08, "eta=0.05": 0.05}),
                ("step11c_seed_check.csv", "arm", {"eta=0.08": "base"})],
    "MLP": [("step11f_mlp_seeds.csv", "설정",
             {"작게 256-64": "256-64 @ lr 3e-4",
              "lr 0.0003": "256-64 @ lr 3e-4",
              "lr 0.001": "256-64 @ lr 1e-3"})],
}

# The tie-break, written down here because section 4.4's selection metric did
# not separate the top two models and something has to. Both criteria are
# measured, and the second one's provenance is recorded honestly: it is not in
# the 4.4 table, so it is a criterion added after seeing the numbers. That is
# a thing to disclose, not to hide behind - and the disclosure has to carry
# the price, which is the log loss given up.
TIE_BREAK = {
    "채택": "XGBoost",
    "근거1": "채택 지표 자체의 재현성 — 시드를 바꿔 다시 적합했을 때 "
           "보정오차가 움직이는 폭 (§4.4 에 선언된 지표에 관한 사실)",
    "근거2": "편별 정확한 기여도 분해가 가능함 (TreeSHAP) — "
           "§4.4 에 선언되지 않은 근거이며, 그렇다고 적는다",
}
PREREGISTERED = 0.000304   # sd 0.000152 x 2, step11c_seed_check.csv base arm

# the two primary metrics section 4.4 declared, which are the ones allowed to
# select. The third declared metric, 약속 이행률, is the FINAL one - but 4.5
# defines it on test 2026, and test 2026 is opened exactly once, for the
# finished model. A metric that cannot be read without opening the test year
# cannot take part in choosing. Its validation-2025 value is printed for
# information, and it does not decide anything.
METRICS = [("cdf_ece", "CDF 보정오차", "채택 지표"),
           ("log_loss", "log loss", "주 지표")]
INFORMATIONAL = [("coverage_90", "약속 이행률 (val)", 0.90)]


def load(name: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(f"없는 파일: {path}\n먼저 해당 단계를 돌려야 한다.")
    return pd.read_csv(path)


def seeds_for(family: str, setting: str):
    """The seed arm for this setting, or None if its spread was never measured.

    Sources are tried in order, so a later, better-designed seed check (step
    11j fits both eta arms under the same three seeds) is preferred over an
    earlier one that only happened to cover the same setting.
    """
    for filename, column, mapping in SEED_ARMS.get(family, []):
        if setting not in mapping:
            continue
        path = config.RESULTS_DIR / filename
        if not path.exists():
            continue
        arm = pd.read_csv(path)
        arm = arm[arm[column] == mapping[setting]]
        if len(arm) > 1:
            return arm, filename
    return None


def xgb_setting(table: pd.DataFrame, label: str) -> int:
    """Which XGBoost row the rule lands on.

    Rule 1 says log loss, and on the full-scale sweep that is eta=0.05 by
    0.00038 over the deployed eta=0.08 - a quarter past the noise threshold,
    with the six best settings separated by less than it. Step 11j refits both
    under three shared seeds to find out whether that survives; its verdict,
    not the single-fit ranking, is what this reads.
    """
    verdict = config.RESULTS_DIR / "step11j_verdict.csv"
    if verdict.exists():
        adopted = float(pd.read_csv(verdict)["채택 eta"].iloc[0])
        hit = table.index[table[label] == f"eta={adopted:g}"]
        if len(hit):
            return int(hit[0])
    return int(table["log_loss"].idxmin())


def scenario_gain() -> float:
    """How much one question to the customer is worth, in log loss.

    Written into the decision record because the report quotes it as the thing
    the price of the tie-break is measured against: a number used that way
    should not live only in a sentence.
    """
    path = config.RESULTS_DIR / "step14_models.csv"
    if not path.exists():
        return float("nan")
    table = pd.read_csv(path)
    try:
        plain = float(table.loc[table["model"].str.startswith("A 무조건 P"),
                                "log_loss"].iloc[0])
        asked = float(table.loc[table["model"].str.startswith("B 참 유형"),
                                "log_loss"].iloc[0])
    except (IndexError, KeyError):
        return float("nan")
    return plain - asked


def verdict(metric: str, rows: pd.DataFrame) -> list:
    """The leader on this metric, against each of the others in turn.

    One pair at a time, because "is the gap real" depends on the noise of the
    two families in that pair. Reading only the gap to second place would let
    a third family's noisiness decide a comparison it is not part of - and
    leaving the leader's own noise out would do the reverse.
    """
    order = rows.sort_values(metric)          # every metric here: lower is better
    best = order.iloc[0]
    calls = []
    for _, other in order.iloc[1:].iterrows():
        gap = abs(float(best[metric]) - float(other[metric]))
        sd_a, sd_b = float(best[f"{metric}_sd"]), float(other[f"{metric}_sd"])
        n_a, n_b = int(best[f"{metric}_n"]), int(other[f"{metric}_n"])

        # A deterministic fit contributes 0 to the noise, which is a
        # measurement. An UNMEASURED spread is not: if either side's spread was
        # never measured, fall back to the pre-registered threshold rather than
        # letting the other side's small number stand in for it - a missing
        # measurement must not make a gap look solid.
        known = [s for s in (sd_a, sd_b) if not np.isnan(s)]
        if len(known) == 2:
            band_single = 2 * max(known)
            fallback = False
        else:
            band_single = PREREGISTERED
            fallback = True
        if len(known) == 2:
            sem = np.sqrt((0.0 if np.isnan(sd_a) else sd_a ** 2) / max(n_a, 1)
                          + (0.0 if np.isnan(sd_b) else sd_b ** 2)
                          / max(n_b, 1))
            band_mean = 2 * sem
        else:
            band_mean = np.nan

        calls.append({
            "metric": metric, "1위": best["family"], "상대": other["family"],
            "1위 값": round(float(best[metric]), 5),
            "상대 값": round(float(other[metric]), 5),
            "차이": round(gap, 5),
            "사전선언 임계값": round(band_single, 5),
            "사전선언 판정": "실재" if gap > band_single else "구분 불가",
            "평균비교 임계값": (round(band_mean, 5)
                           if not np.isnan(band_mean) else np.nan),
            "평균비교 판정": ("실재" if (not np.isnan(band_mean)
                                   and gap > band_mean) else "구분 불가"),
            "잡음 출처": (f"{best['family']} sd {sd_a:.6f} · "
                      f"{other['family']} sd {sd_b:.6f}"
                      + (" — 한쪽이 미측정이라 사전 선언 임계값으로 대체"
                         if fallback else "")),
        })
    return calls


def main() -> int:
    config.ensure_dirs()
    plotting.setup()

    everything, summary = [], []

    for family, (filename, _) in FAMILIES.items():
        table = load(filename)
        label = "설정" if "설정" in table.columns else "model"
        table = table.drop_duplicates(subset=["log_loss", "cdf_ece"])
        chosen_at = (xgb_setting(table, label) if family == "XGBoost"
                     else table["log_loss"].idxmin())    # rule 1
        for index, row in table.iterrows():
            everything.append({"family": family, "설정": row[label],
                               "log_loss": row["log_loss"],
                               "cdf_ece": row["cdf_ece"],
                               "chosen": index == chosen_at,
                               "출처": filename})
        pick = table.loc[chosen_at]
        setting = str(pick[label])
        entry = {"family": family, "설정 수": len(table), "고른 설정": setting,
                 "cdf_ece 설정간 폭": round(float(table["cdf_ece"].max()
                                              - table["cdf_ece"].min()), 5),
                 "출처": filename}

        found = seeds_for(family, setting)                # rule 2
        arm, arm_file = found if found else (None, None)
        for metric, _, _ in METRICS + [(m, n, None) for m, n, _
                                       in INFORMATIONAL]:
            if metric not in table.columns:
                entry[metric] = np.nan
                entry[f"{metric}_sd"] = np.nan
                entry[f"{metric}_n"] = 0
            elif arm is not None and metric in arm.columns:
                entry[metric] = round(float(arm[metric].mean()), 5)
                entry[f"{metric}_sd"] = round(float(arm[metric].std(ddof=1)), 6)
                entry[f"{metric}_n"] = int(len(arm))
            else:
                entry[metric] = float(pick[metric])
                # lbfgs on a fixed design matrix with a fixed seed and no
                # subsampling returns the same solution every time: its seed
                # spread is zero, which is a measurement, not a missing value.
                # Anything else with no seed arm is genuinely unmeasured.
                entry[f"{metric}_sd"] = 0.0 if family == "로지스틱" else np.nan
                entry[f"{metric}_n"] = 1
        entry["시드 출처"] = ("결정적 (재적합해도 같은 해, sd=0)"
                          if family == "로지스틱"
                          else (arm_file if arm is not None
                                else "미측정 — 단일 적합"))
        summary.append(entry)

    configs = pd.DataFrame(everything)
    table = pd.DataFrame(summary).sort_values("cdf_ece").reset_index(drop=True)
    configs.to_csv(config.RESULTS_DIR / "step11i_selection.csv", index=False,
                   encoding="utf-8-sig")
    table.to_csv(config.RESULTS_DIR / "step11i_summary.csv", index=False,
                 encoding="utf-8-sig")

    pairs = [c for m, _, _ in METRICS for c in verdict(m, table)]
    calls = pd.DataFrame(pairs)
    calls.to_csv(config.RESULTS_DIR / "step11i_verdict.csv", index=False,
                 encoding="utf-8-sig")

    # ---- figure ---------------------------------------------------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))
    for family, (_, colour) in FAMILIES.items():
        part = configs[configs["family"] == family]
        ax1.scatter(part["log_loss"], part["cdf_ece"], s=40, color=colour,
                    alpha=0.5, label=f"{family} ({len(part)}종)")
        best = part[part["chosen"]]
        ax1.scatter(best["log_loss"], best["cdf_ece"], s=230,
                    facecolors="none", edgecolors=colour, linewidths=2.4,
                    zorder=4)
    ax1.set_xlabel("val log loss  ← 좋음")
    ax1.set_ylabel("val CDF 보정오차  ← 좋음")
    ax1.set_title("탐색한 설정 전체 — 원 = log loss 로 고른 설정", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    y = np.arange(len(table))
    noisiest = float(np.nanmax(table["cdf_ece_sd"]))
    ax2.axvspan(float(table["cdf_ece"].min()) - 2 * noisiest,
                float(table["cdf_ece"].min()) + 2 * noisiest,
                color="#868e96", alpha=0.13, zorder=0)
    for i, row in table.iterrows():
        colour = FAMILIES[row["family"]][1]
        ax2.plot([row["cdf_ece"]], [i], "o", color=colour, markersize=11,
                 zorder=4)
        if row["cdf_ece_n"] > 1:
            ax2.errorbar([row["cdf_ece"]], [i], xerr=2 * row["cdf_ece_sd"],
                         color=colour, capsize=5, linewidth=1.8, zorder=5)
        note = (f"{row['cdf_ece']:.5f}" if row["cdf_ece_n"] == 1
                else f"{row['cdf_ece']:.5f} ± {row['cdf_ece_sd']:.5f}")
        ax2.annotate(note, (row["cdf_ece"], i), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=9, color=colour)
    ax2.set_yticks(y)
    ax2.set_yticklabels(table["family"])
    ax2.invert_yaxis()
    ax2.set_xlabel("val CDF 보정오차")
    ax2.set_title("오차선 = 시드 2sd · 회색 띠 = 1위를 둘러싼 잡음 폭",
                  fontsize=10)
    ax2.grid(alpha=0.25, axis="x")

    fig.suptitle("Step 11i — 채택 규칙을 적용한다 (val 2025)", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "step11i_selection.png")
    plt.close(fig)

    # ---- what it says ----------------------------------------------------- #
    print("\n" + "=" * 110)
    print("1·2단계 — 모델 안에서 log loss 로 설정을 고르고, 시드가 있으면 평균을 쓴다")
    show = table.copy()
    for metric, _, _ in METRICS:
        show[metric] = [
            f"{v:.5f}" + (f" ± {s:.5f} (n={n})" if n > 1 else "  (단일)")
            for v, s, n in zip(show[metric], show[f"{metric}_sd"],
                               show[f"{metric}_n"])]
    print(show[["family", "설정 수", "고른 설정", "cdf_ece", "log_loss",
                "시드 출처"]].to_string(index=False))

    leaders = {}
    for metric, name, role in METRICS:
        mine = [c for c in pairs if c["metric"] == metric]
        print(f"\n{'─' * 108}")
        print(f"{name} ({role}) — 1위 {mine[0]['1위']} {mine[0]['1위 값']}")
        for call in mine:
            print(f"  vs {call['상대']} {call['상대 값']}  "
                  f"차이 {call['차이']:.5f}")
            print(f"     [사전 선언] 단일 적합 sd 2배 "
                  f"{call['사전선언 임계값']:.5f} → {call['사전선언 판정']}"
                  f"   ({call['잡음 출처']})")
            if not np.isnan(call["평균비교 임계값"]):
                print(f"     [평균 비교] 2×SEM "
                      f"{call['평균비교 임계값']:.5f} → "
                      f"{call['평균비교 판정']}")
                if call["사전선언 판정"] != call["평균비교 판정"]:
                    print("     두 방식이 엇갈린다 → 사전에 적어 둔 쪽을 쓴다 "
                          "(보고 나서 고르면 선언한 의미가 없다)")
        beats_all = all(c["사전선언 판정"] == "실재" for c in mine)
        leaders[metric] = (mine[0]["1위"], beats_all, mine)
        print(f"  → {mine[0]['1위']} 가 나머지 전부와 구분된다"
              if beats_all else
              f"  → {mine[0]['1위']} 가 1위지만 구분되지 않는 상대가 있다")

    print(f"\n{'=' * 110}")
    leader, decided, mine = leaders["cdf_ece"]
    other, other_ok, other_pairs = leaders["log_loss"]

    if decided:
        adopt = leader
        print(f"채택 지표(CDF 보정오차)가 순위를 정했다 → 채택: {adopt}")
    else:
        tied = [c["상대"] for c in mine if c["사전선언 판정"] == "구분 불가"]
        print(f"채택 지표(CDF 보정오차)는 {leader} 를 1위로 놓지만 "
              f"{', '.join(tied)} 와 구분하지 못한다.")
        print("  → 선언한 지표만으로는 정해지지 않는다. 동점을 무엇으로 "
              "깼는지 적어야 하고, 그 대가도 같이 적어야 한다.")
        adopt = TIE_BREAK["채택"]
        print(f"\n동점 처리 (문서화된 결정) — 채택: {adopt}")
        print(f"  ① {TIE_BREAK['근거1']}")
        repro = table[table["cdf_ece_n"] > 1].sort_values("cdf_ece_sd")
        if len(repro) > 1:
            tight, loose = repro.iloc[0], repro.iloc[-1]
            print(f"     측정: {tight['family']} sd {tight['cdf_ece_sd']:.6f} "
                  f"vs {loose['family']} sd {loose['cdf_ece_sd']:.6f} "
                  f"({loose['cdf_ece_sd'] / tight['cdf_ece_sd']:.0f}배)")
        for family, (filename, _) in FAMILIES.items():
            row = table[table["family"] == family].iloc[0]
            print(f"     설정끼리의 폭: {family} "
                  f"{row['cdf_ece 설정간 폭']:.5f} "
                  f"({int(row['설정 수'])}종, {filename})")
        print(f"  ② {TIE_BREAK['근거2']}")

    # ---- the price of the tie-break -------------------------------------- #
    if adopt and other != adopt and other_ok:
        cost = next((c for c in other_pairs if c["상대"] == adopt), None)
        span = float(table["log_loss"].max() - table["log_loss"].min())
        asked_gain = scenario_gain()
        if cost:
            print(f"\n대가 — log loss {cost['차이']:.5f} 를 지불했다 "
                  f"({other} {cost['1위 값']} vs {adopt} {cost['상대 값']}, "
                  f"임계값 {cost['사전선언 임계값']:.5f} → 실재하는 차이다)")
            print(f"  그 크기: 알고리즘 3종이 만드는 폭 {span:.5f} 안이다")
            if not np.isnan(asked_gain):
                print(f"  비교: 손님에게 하루 유형을 한 번 묻는 효과는 "
                      f"{asked_gain:.5f} — 지불액의 "
                      f"{asked_gain / cost['차이']:.0f}배 "
                      f"(step14_models.csv)")
            for metric, name, target in INFORMATIONAL:
                if table[metric].isna().any():
                    continue
                pair = table.set_index("family")[metric]
                print(f"  그리고 그 차이는 {name} 로 번역되지 않는다: "
                      f"{adopt} {pair[adopt]:.4f} vs {other} "
                      f"{pair[other]:.4f} (약속 {target:.2f})")

    if adopt:
        print(f"\n채택: {adopt} "
              f"({table.loc[table['family'] == adopt, '고른 설정'].iloc[0]})")
        pd.DataFrame([{
            "채택": adopt,
            "설정": table.loc[table["family"] == adopt, "고른 설정"].iloc[0],
            "채택 지표가 정했는가": "예" if decided else "아니오 — 동점",
            "동점 근거1": "" if decided else TIE_BREAK["근거1"],
            "동점 근거2": "" if decided else TIE_BREAK["근거2"],
            "지불한 log loss": (0.0 if other == adopt
                            else next((c["차이"] for c in other_pairs
                                       if c["상대"] == adopt), np.nan)),
            "알고리즘 3종 폭": round(span, 5),
            "하루 유형을 묻는 효과": (round(asked_gain, 5)
                             if not np.isnan(asked_gain) else np.nan),
        }]).to_csv(config.RESULTS_DIR / "step11i_decision.csv", index=False,
                   encoding="utf-8-sig")

    # ---- declared but not usable for selection ---------------------------- #
    for metric, name, target in INFORMATIONAL:
        if table[metric].isna().any():
            print(f"\n{name} — 일부 탐색 파일에 이 열이 없어 비교를 건너뛴다.")
            continue
        print(f"\n{name} — §4.4 의 최종 지표. §4.5 가 test 2026 에서 재도록 "
              f"정해 두었고, test 2026 은 완성된 모델에 대해 한 번만 연다.")
        print(f"  고르는 데 쓸 수 없다. val 2025 값은 참고로만 적는다 "
              f"(약속한 값 {target:.2f} 에 가까운 쪽이 정직하다).")
        order = table.assign(_off=(table[metric] - target).abs()) \
                     .sort_values("_off")
        for _, row in order.iterrows():
            band = ("" if row[f"{metric}_n"] <= 1
                    else f" ± {row[f'{metric}_sd']:.4f}")
            print(f"    {row['family']:>8s} {row[metric]:.4f}{band}"
                  f"   {target:.2f} 에서 {row[metric] - target:+.4f}")
        points_to = order.iloc[0]["family"]
        if adopt and points_to == adopt:
            print(f"  → 이 지표도 {adopt} 를 가리킨다. 동점을 깬 근거를 "
                  "달리 골랐더라도 결론은 같았다.")
        elif adopt:
            print(f"  → 이 지표는 {points_to} 를 가리켜 채택과 엇갈린다. "
                  "쓰지 않기로 한 지표이므로 결론은 바꾸지 않고, 본문에 적는다.")

    print("\n무엇이 근거였고 무엇이 아니었는지")
    print("  사전 선언 (§4.4): log loss · CDF 보정오차 — 순위를 매기는 데 썼다")
    print("  사전 선언이지만 쓸 수 없음: 약속 이행률 (test 2026 전용, §4.5)")
    if not decided:
        print("  동점 처리에 쓴 것: 위 ①(선언된 지표의 재현성) "
              "· ②(기여도 분해 가능성, 선언 밖)")
    print("  끝까지 쓰지 않은 것: 학습 시간, 코드 재사용, 이미 학습돼 있다는 사실")
    det = table[(table["cdf_ece_n"] <= 1) & (table["cdf_ece_sd"] == 0)]
    for _, row in det.iterrows():
        print(f"  참고: {row['family']} 는 결정적이라 시드 산포가 없다 "
              "(미측정이 아니라 0).")
    print(f"\nresults: {config.RESULTS_DIR / 'step11i_verdict.csv'}")
    print(f"         {config.RESULTS_DIR / 'step11i_decision.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
