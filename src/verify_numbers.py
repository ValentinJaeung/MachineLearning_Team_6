"""Every number in the report has to come from a file in results/.

Reads  docs/REPORT.md, docs/REPORT_OUTLINE.md, README.md
       results/*.csv, results/*.json
Writes results/verify_numbers.csv

A report nobody can check is a report nobody should believe. This step takes
every number written in the documents with four or more decimal places and
asks where it came from. There are three honest answers:

  직접       the value appears in a results file as it is written
  차이       it is |a - b| for two values in the same column of one file,
             which is how every "gap", "span" and "price" is built
  배수       it is a small whole multiple of such a value - the noise
             threshold is a standard deviation times two, and nothing else in
             the report multiplies

Anything else is a number somebody typed, and that is what this step is for.
Rounding is allowed in exactly one direction: a written 0.0473 matches a
stored 0.04734, because the report rounds for reading. It does NOT match a
stored 0.047 or 0.05 - comparing at fewer digits than were written would let a
five-decimal claim be "supported" by a three-decimal coincidence, and with 345
columns of numbers to draw on there is always one. Each number is therefore
checked at the precision it was written with.

A difference also has to be a difference somebody could have meant. Every
table in the report cites the file it came from, so an explanation counts only
when the file is the one cited at or above that line, or when the column name
is the metric the line is talking about. Anything else is reported as having
no source, which is the point: an audit that always says yes is decoration.

"Cited at or above that line" is read as a window of a dozen lines either
way, because the report puts a table's 출처 line underneath the table.

One limit worth knowing: where a line names two metrics ("log loss 0.0558,
AUC +0.0553"), two columns can both explain a number and the one reported is
whichever matched first. The number is still checked; only the label naming
which column it came from can land on the wrong one of the two.

The search for a difference is deliberately narrow: both operands must sit in
the same **column** of the same file, because a gap is always a gap between
two measurements of the same quantity. A wider search finds a subtraction for
any number at all - with every value in results/ to choose from, 0.0558 can be
"explained" by two unrelated silhouette scores - and an audit that explains
everything has checked nothing. Where several columns could explain the same
number, the one whose name matches the metric named on that line wins, and the
answer records file and column so a wrong attribution is visible rather than
hidden.

Usage:  python src/verify_numbers.py
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pandas as pd

import config

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "docs" / "REPORT.md",
        ROOT / "docs" / "REPORT_OUTLINE.md",
        ROOT / "README.md"]

NUMBER = re.compile(r"\d+\.\d{4,}")
CITED = re.compile(r"([A-Za-z0-9_]+\.(?:csv|json))")
RAW = re.compile(r"-?\d+\.\d+")
SELF = "verify_numbers.csv"   # this step's own output is not a source


# a metric word in the report line, and the column names that would carry it
METRIC_WORDS = {
    "log_loss": ("log loss", "logloss", "mlogloss"),
    "cdf_ece": ("보정오차", "ece"),
    "auc": ("auc",),
    "coverage": ("이행률", "준수율", "coverage"),
    "brier": ("brier",),
    "minute": ("분",),
    "sd": ("임계값", "표준편차", "잡음", "sd"),
}


def columns_by_file() -> dict[tuple[str, str], set[float]]:
    """Every number with 4+ decimals, keyed by (filename, column)."""
    found: dict[tuple[str, str], set[float]] = {}
    for path in sorted(config.RESULTS_DIR.iterdir()):
        if path.name == SELF:
            continue
        if path.suffix == ".csv":
            try:
                table = pd.read_csv(path, encoding="utf-8-sig")
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            for column in table.columns:
                values = pd.to_numeric(table[column], errors="coerce").dropna()
                values = {abs(float(v)) for v in values if float(v) != 0}
                if values:
                    found[(path.name, str(column))] = values
        elif path.suffix == ".json":
            try:
                text = json.dumps(json.loads(
                    io.open(path, encoding="utf-8-sig").read()))
            except (OSError, ValueError):
                continue
            values = {abs(float(m)) for m in RAW.findall(text)}
            found[(path.name, "(json)")] = {v for v in values if v > 0}
    return found


def close(written: float, stored: float, digits: int) -> bool:
    """Is `stored` the number the report wrote, at the precision it wrote it?"""
    return round(stored, digits) == written


def fits_line(column: str, line: str) -> bool:
    """Does this column carry the metric the report line is talking about?"""
    low = line.lower()
    col = column.lower()
    for key, words in METRIC_WORDS.items():
        if key in col and any(w in low for w in words):
            return True
    return False


def score(name: str, column: str, line: str, cited: set[str]) -> int:
    """How plausible is this file/column as the source for this line?"""
    return (2 if name in cited else 0) + (1 if fits_line(column, line) else 0)


MULTIPLES = (2, 3)        # the threshold is sd x 2; nothing else multiplies


def explain(written: str, pool: dict[tuple[str, str], set[float]],
            line: str, cited: set[str]) -> tuple[str, str]:
    """Where does this number come from? Direct hit, then a same-column gap."""
    target = float(written)
    digits = len(written.split(".")[1])

    hits = [(score(name, column, line, cited), name, column)
            for (name, column), values in pool.items()
            if any(close(target, v, digits) for v in values)]
    if hits:
        _, name, column = max(hits, key=lambda h: h[0])
        return "직접", f"{name} · {column}"

    # a gap, a span, a price: two readings of the SAME quantity, subtracted
    candidates = []
    for (name, column), values in pool.items():
        ordered = sorted(values)
        if len(ordered) > 4000:          # a per-flight dump, not a result table
            continue
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if b - a > target + 10 ** -digits:
                    break
                if close(target, b - a, digits):
                    candidates.append((score(name, column, line, cited),
                                       name, column, b, a))
    candidates = [c for c in candidates if c[0] > 0]
    if candidates:
        _, name, column, b, a = max(candidates, key=lambda c: c[0])
        return "차이", f"{name} · {column} ({b:.5f} − {a:.5f})"

    # a small whole multiple: the noise threshold is a standard deviation x 2
    scaled = [(score(name, column, line, cited), name, column, k, v)
              for (name, column), values in pool.items()
              for v in values for k in MULTIPLES
              if close(target, k * v, digits)]
    scaled = [c for c in scaled if c[0] > 0]
    if scaled:
        _, name, column, k, v = max(scaled, key=lambda c: c[0])
        return "배수", f"{name} · {column} ({v:.6f} × {k})"
    return "출처 없음", ""


def main() -> int:
    config.ensure_dirs()
    pool = columns_by_file()
    files = len({name for name, _ in pool})
    print(f"results/ 파일 {files}개 · 열 {len(pool)}개에서 숫자를 모았다")

    rows = []
    for doc in DOCS:
        if not doc.exists():
            continue
        lines = io.open(doc, encoding="utf-8").read().splitlines()
        for line_no, line in enumerate(lines, 1):
            window = lines[max(0, line_no - 13):line_no + 12]
            cited = set(CITED.findall(chr(10).join(window)))
            for written in NUMBER.findall(line):
                kind, source = explain(written, pool, line, cited)
                rows.append({"문서": doc.name, "줄": line_no, "숫자": written,
                             "판정": kind, "출처": source,
                             "문맥": line.strip()[:70]})

    table = pd.DataFrame(rows).drop_duplicates(subset=["문서", "줄", "숫자"])
    path = config.RESULTS_DIR / "verify_numbers.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    counts = table["판정"].value_counts()
    print("\n" + "=" * 78)
    for kind in ("직접", "차이", "배수", "출처 없음"):
        print(f"  {kind:<8} {int(counts.get(kind, 0)):>4}건")
    shown = table[table["판정"] != "직접"]
    if len(shown):
        print("\n차이로 설명된 숫자 — 어느 열의 두 값을 뺀 것인지:")
        print(shown[["문서", "줄", "숫자", "출처"]].to_string(index=False))
    orphans = table[table["판정"] == "출처 없음"]
    if len(orphans):
        print("\n출처를 못 찾은 숫자 — 고치거나, 어떻게 나온 값인지 적어야 한다:")
        print(orphans[["문서", "줄", "숫자", "문맥"]].to_string(index=False))
    else:
        print("\n문서의 모든 숫자가 results/ 로 돌아간다.")
    print(f"\nresults: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
