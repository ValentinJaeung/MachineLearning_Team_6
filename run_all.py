"""
Run the whole pipeline, in order.

    python run_all.py              every step
    python run_all.py --from 10    from step 10 onwards
    python run_all.py --only 8     one step
    python run_all.py --dry-run    print the plan and stop

Each step reads what the one before it wrote, so the order matters. Changing
an early step means re-running everything after it: mixing outputs from two
settings produces a table whose rows were never true at the same time.

Timings are from the reported run (Windows 11, 16-core CPU, no GPU).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    (1, "step01_download.py", "BTS 수집 (이미 받은 월은 건너뜀)", "5분 / 최초 40분"),
    (2, "step02_clean.py", "정제·시각 변환·이름표·10구간 라벨", "3분"),
    (3, "step03_schedule_features.py", "시간표 파생 변수 3종", "4분"),
    (6, "step06_scenario.py", "그날의 유형 + PCA·군집 검증", "8분"),
    (7, "step07_dataset.py", "결합·연도 분할", "4분"),
    (8, "step08_eda.py", "EDA — 설정값 확정과 구조 기술", "5분"),
    (9, "step09_preprocess.py", "선형용·트리용 두 벌 전처리", "4분"),
    (None, "verify_leakage.py", "누출 감사 44항목", "2분"),
    (11, "step11_models.py", "모델 3종 + 과적합·탐색", "45분"),
    (None, "step11c_polynomial.py", "다항 특성·규제 비교", "15분"),
    (None, "step11c_seed_check.py", "다항 확장의 시드 재현 검사", "50분"),
    (12, "step12_calibration.py", "확률 보정과 보정 구간 비교", "35분"),
    (13, "step13_explain.py", "SHAP + 시각 what-if", "10분"),
    (14, "step14_scenario_models.py", "모델 A·B·C와 합성", "45분"),
    (15, "step15_service_eval.py", "test 2026 서비스 평가 (한 번만)", "10분"),
    (16, "step16_demo_data.py", "데모용 사전 계산", "5분"),
    (None, "figure_guarantee.py", "방법 설명 그림 (분포 → 보장 시각)", "1분"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=0)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = [s for s in STEPS
            if (args.only is None or s[0] == args.only)
            and (s[0] is None or s[0] >= args.start)]

    print(f"{len(plan)} step(s) to run\n")
    for number, script, title, cost in plan:
        tag = f"step {number:>2}" if number else "     +  "
        print(f"  {tag}  {title:<40} ~{cost}")
    if args.dry_run:
        return 0

    started = time.time()
    for number, script, title, _ in plan:
        tag = f"step {number}" if number else "+"
        print(f"\n{'=' * 78}\n{tag}: {title}\n{'=' * 78}", flush=True)
        result = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n{tag} failed with code {result.returncode}; stopping.",
                  file=sys.stderr)
            return result.returncode

    print(f"\nall done in {(time.time() - started) / 60:.0f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
