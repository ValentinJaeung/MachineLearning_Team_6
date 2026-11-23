"""그림 설정 — 한글 폰트와 공통 스타일.

matplotlib 은 기본 폰트에 한글이 없어서 축 이름이 네모로 나온다. 윈도우에 기본
설치된 맑은 고딕을 쓰고, 없으면 영문 폰트로 떨어진다(그림은 그려진다).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                       # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib import font_manager         # noqa: E402

CANDIDATES = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]


def setup() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["font.size"] = 10
