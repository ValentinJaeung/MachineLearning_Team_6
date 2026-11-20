"""
Shared matplotlib setup.

Every figure in this project carries Korean labels, and matplotlib's default
font draws Korean as empty boxes. This module picks the first Korean-capable
font actually installed and fixes the minus sign, which the Korean fonts get
wrong. Import it before pyplot is used:

    import plotting
    plotting.setup()
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# in preference order; the first one present wins
CANDIDATES = [
    "Malgun Gothic",        # Windows
    "AppleGothic",          # macOS
    "NanumGothic",          # common Linux install
    "Noto Sans CJK KR",
    "Noto Sans KR",
]

_done = False


def setup() -> str:
    """Set a Korean-capable font. Returns the family that was chosen."""
    global _done
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in CANDIDATES if name in available), None)

    if chosen is not None:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25

    if not _done:
        print(f"[plot] font: {chosen or 'default (Korean labels may not render)'}")
        _done = True
    return chosen or ""
