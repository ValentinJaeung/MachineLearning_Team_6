"""전체 파이프라인을 순서대로 돌린다.

    python run_all.py              처음부터 끝까지
    python run_all.py --from 6     6단계부터
    python run_all.py --only 9     9단계만
    python run_all.py --list       단계 목록만 보기

원자료(BTS)는 이 프로젝트에서 받지 않는다. 이미 내려받아 둔 것을 읽는다 —
경로는 src/config.py 의 RAW_DIR, 환경변수 FINAL_RAW_DIR 로 덮어쓸 수 있다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    (1, "step01_label.py", "정제 + 10구간 정답", "~3분"),
    (2, "step02_features.py", "파생 변수 2개", "~2분"),
    (3, "step03_scenario.py", "그날의 유형 2열 + 라벨 불가 행 제거", "~4분"),
    (4, "step04_eda.py", "EDA, 공항 목록 결정", "~2분"),
    (5, "step05_preprocess.py", "설계행렬 두 벌", "~3분"),
    (6, "step06_train.py", "3개 모델 학습·튜닝", "~3시간"),
    (7, "step07_select.py", "모델 1개 선정", "~1분"),
    (8, "step08_calibrate.py", "확률 보정", "~1분"),
    (9, "step09_service.py", "서비스 검증 (전체 + 25칸)", "~5분"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--only", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for number, script, what, span in STEPS:
            print(f"  {number:>2}  {script:<24} {what}  ({span})")
        return 0

    if args.only:
        todo = [s for s in STEPS if s[0] == args.only]
    else:
        todo = [s for s in STEPS if s[0] >= args.start]

    started = time.time()
    for number, script, what, _ in todo:
        print(f"\n{'=' * 70}\n[{number}] {script} — {what}\n{'=' * 70}",
              flush=True)
        step_started = time.time()
        result = subprocess.run([sys.executable, str(ROOT / "src" / script)],
                                cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[{number}] {script} 실패 (코드 {result.returncode})",
                  file=sys.stderr)
            return result.returncode
        print(f"\n[{number}] 완료 — {(time.time() - step_started) / 60:.1f}분",
              flush=True)

    print(f"\n전체 완료 — {(time.time() - started) / 60:.0f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
