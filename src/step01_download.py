"""
Step 1 - data collection (PROPOSAL_FLOW §4, 1단계).

Downloads the BTS "Reporting Carrier On-Time Performance (1987-present)"
monthly files for June/July/August 2021-2026 (18 months), keeps the 40 source
columns the project needs, and writes one parquet per month to data/raw/.

Why summer only
---------------
The splits are by YEAR (train 2022-24 / val 2025 / test 2026). For a year-to-
year comparison to mean anything the three groups must have the same seasonal
make-up, so the period is pinned to the same three months every year and only
one axis - the year - is left free.

Why 2021 is downloaded but never trained on
-------------------------------------------
Each year's "report card" features (step 5) are built from the PREVIOUS year
only. 2022 is the first training year, so 2021 is needed as its source.

The 40 columns cover all four variable groups of PROPOSAL_FLOW §3-1:
A-1 booking-time inputs, A-2 identifiers, B outcomes, C post-hoc detail.
Which of them a model may see is decided in src/config.py, not here.

Months already present in data/raw/ are skipped, so the step is re-runnable.

Source:  https://transtats.bts.gov/  (U.S. DOT, public domain)
Usage:   python src/step01_download.py
"""

from __future__ import annotations

import csv
import os
import shutil
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

YEARS = range(2021, 2027)
MONTHS = (6, 7, 8)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TMP_DIR = RAW_DIR / "_tmp"
LOG_PATH = RAW_DIR / "download_log.csv"

MAX_ATTEMPTS = 3
RETRY_SLEEP = 5
TIMEOUT = 900

# One month is a ~32 MB download and a ~250 MB CSV parse, so a single worker
# leaves 15 of 16 cores and most of the link idle. Five concurrent months is
# roughly a 5x wall-clock win and still a modest number of connections to a
# public government server. Each worker starts STAGGER seconds after the last
# so the run does not open five sockets in the same instant.
WORKERS = min(5, max(1, (os.cpu_count() or 4) - 1))
STAGGER = 2.0

USER_AGENT = "Mozilla/5.0 (compatible; research-data-downloader)"
KST = timezone(timedelta(hours=9))

_log_lock = threading.Lock()
_print_lock = threading.Lock()


def say(message: str) -> None:
    """print() from several threads without interleaving lines."""
    with _print_lock:
        print(message, flush=True)

# --------------------------------------------------------------------------- #
# 40 columns, grouped the way the BTS download page groups them
# --------------------------------------------------------------------------- #

COL_TIME = ["Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate"]
COL_AIRLINE = ["Reporting_Airline", "Tail_Number", "Flight_Number_Reporting_Airline"]
COL_ORIGIN = ["Origin", "OriginCityName", "OriginState", "OriginCityMarketID"]
COL_DEST = ["Dest", "DestCityName", "DestState", "DestCityMarketID"]
COL_DEP = ["CRSDepTime", "DepTime", "DepDelay", "DepDel15", "TaxiOut"]
COL_ARR = ["CRSArrTime", "ArrTime", "ArrDelay", "ArrDel15", "TaxiIn"]
COL_CANCEL = ["Cancelled", "CancellationCode", "Diverted"]
COL_SUMMARY = ["CRSElapsedTime", "ActualElapsedTime", "AirTime", "Distance"]
COL_CAUSE = [
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]
COL_DIV = ["DivReachedDest", "DivArrDelay"]

KEEP_COLUMNS = (
    COL_TIME
    + COL_AIRLINE
    + COL_ORIGIN
    + COL_DEST
    + COL_DEP
    + COL_ARR
    + COL_CANCEL
    + COL_SUMMARY
    + COL_CAUSE
    + COL_DIV
)
assert len(KEEP_COLUMNS) == 40, len(KEEP_COLUMNS)

COLUMN_GROUPS = [
    ("Time Period", COL_TIME),
    ("Airline", COL_AIRLINE),
    ("Origin", COL_ORIGIN),
    ("Destination", COL_DEST),
    ("Departure", COL_DEP),
    ("Arrival", COL_ARR),
    ("Cancel/Divert", COL_CANCEL),
    ("Flight Summaries", COL_SUMMARY),
    ("Cause of Delay", COL_CAUSE),
    ("Diverted info", COL_DIV),
]

STRING_COLUMNS = [
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginCityName",
    "OriginState",
    "Dest",
    "DestCityName",
    "DestState",
    "CancellationCode",
]
DATE_COLUMNS = ["FlightDate"]
# HHMM clock readings and integer ids; blank for cancelled flights, so nullable
INT_COLUMNS = [
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "CRSDepTime",
    "DepTime",
    "CRSArrTime",
    "ArrTime",
    "OriginCityMarketID",
    "DestCityMarketID",
]
FLOAT_COLUMNS = [
    c for c in KEEP_COLUMNS if c not in STRING_COLUMNS + DATE_COLUMNS + INT_COLUMNS
]

LOG_FIELDS = ["year", "month", "url", "status", "download_datetime", "rows", "columns"]


class Stop(Exception):
    """Fatal condition: stop the whole run and report."""


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %z")


def log_row(year, month, url: str, status: str, rows, columns=None) -> None:
    with _log_lock:
        is_new = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    "year": year,
                    "month": month,
                    "url": url,
                    "status": status,
                    "download_datetime": now_kst(),
                    "rows": "" if rows is None else rows,
                    "columns": "" if columns is None else columns,
                }
            )


def download(url: str, dst: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        with dst.open("wb") as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)


def extract_csv(zip_path: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise Stop(
                "Expected exactly one CSV inside {}, found: {!r}".format(
                    zip_path.name, names
                )
            )
        name = names[0]
        target = out_dir / Path(name).name
        with zf.open(name) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    return target


def check_columns(csv_path: Path) -> int:
    """Verify every requested column exists. Returns the source column count."""
    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = [c for c in KEEP_COLUMNS if c not in header]
    if missing:
        listing = "\n".join("  {:3d}  {}".format(i, c) for i, c in enumerate(header))
        raise Stop(
            "Columns missing from the source file {}: {}\n"
            "Original header ({} columns):\n{}".format(
                csv_path.name, missing, len(header), listing
            )
        )
    return len(header)


def read_csv_arrow(csv_path: Path) -> pd.DataFrame:
    """
    Parse the CSV with pyarrow, which is multithreaded and several times faster
    than pandas on a 250 MB file. Only the 40 requested columns are decoded;
    the trailing unnamed column BTS appends to every row is simply not read.
    """
    read_options = pacsv.ReadOptions(use_threads=True, encoding="utf-8-sig")
    convert_options = pacsv.ConvertOptions(
        include_columns=KEEP_COLUMNS,
        column_types={c: pa.string() for c in STRING_COLUMNS},
        strings_can_be_null=True,
    )
    table = pacsv.read_csv(
        csv_path, read_options=read_options, convert_options=convert_options
    )
    return table.to_pandas()


def read_csv_pandas(csv_path: Path) -> pd.DataFrame:
    """Fallback parser, used only if pyarrow rejects the file."""
    return pd.read_csv(
        csv_path,
        usecols=KEEP_COLUMNS,
        dtype={c: "string" for c in STRING_COLUMNS},
        encoding="utf-8-sig",
        low_memory=False,
    )


def read_and_cast(csv_path: Path, tag: str) -> pd.DataFrame:
    try:
        frame = read_csv_arrow(csv_path)
    except Exception as exc:
        say(
            "[{}] pyarrow CSV 파싱 실패 ({}: {}), pandas로 재시도".format(
                tag, type(exc).__name__, exc
            )
        )
        frame = read_csv_pandas(csv_path)

    missing = [c for c in KEEP_COLUMNS if c not in frame.columns]
    if missing:
        raise Stop("{} 파싱 결과에 열이 없습니다: {}".format(csv_path.name, missing))
    frame = frame[KEEP_COLUMNS]

    for col in DATE_COLUMNS:
        frame[col] = pd.to_datetime(frame[col], errors="raise")

    for col in INT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").round().astype("Int32")

    for col in FLOAT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    # Trailing whitespace shows up in the city name / state fields.
    for col in STRING_COLUMNS:
        frame[col] = frame[col].str.strip()

    return frame


def process_month(year: int, month: int) -> str:
    url = URL_TEMPLATE.format(year=year, month=month)
    tag = "{}-{:02d}".format(year, month)
    parquet_path = RAW_DIR / "{}_{:02d}.parquet".format(year, month)

    if parquet_path.exists():
        say("[{}] parquet already exists, skipping".format(tag))
        log_row(year, month, url, "skipped_exists", None)
        return "skipped_exists"

    started = time.monotonic()
    work_dir = TMP_DIR / "{}_{:02d}".format(year, month)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_path = work_dir / "{}_{:02d}.zip".format(year, month)
    last_error = None

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                say(
                    "[{}] downloading (attempt {}/{})".format(
                        tag, attempt, MAX_ATTEMPTS
                    )
                )
                download(url, zip_path)
                break
            except ssl.SSLError as exc:
                raise Stop(
                    "SSL error while downloading {}: {}\n"
                    "Stopped without bypassing certificate verification.".format(
                        url, exc
                    )
                ) from exc
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    say("[{}] HTTP 404 - not published, skipping".format(tag))
                    log_row(year, month, url, "not_published_404", None)
                    return "not_published_404"
                last_error = "HTTP {} {}".format(exc.code, exc.reason)
                say("[{}] {}".format(tag, last_error))
            except Exception as exc:
                last_error = "{}: {}".format(type(exc).__name__, exc)
                say("[{}] {}".format(tag, last_error))

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP)
        else:
            raise Stop(
                "Download failed {} times for {}\nLast error: {}\n"
                "Stopped without falling back to another data source.".format(
                    MAX_ATTEMPTS, url, last_error
                )
            )

        if not zipfile.is_zipfile(zip_path):
            size = zip_path.stat().st_size
            with zip_path.open("rb") as fh:
                head = fh.read(200)
            raise Stop(
                "{} did not return a zip file (size={} bytes). "
                "First bytes: {!r}".format(url, size, head)
            )

        csv_path = extract_csv(zip_path, work_dir)
        n_source_cols = check_columns(csv_path)
        frame = read_and_cast(csv_path, tag)

        frame.to_parquet(parquet_path, engine="pyarrow", index=False)
        rows = len(frame)
        say(
            "[{}] saved {} ({:,} rows, {} of {} source columns) in {:.0f}s".format(
                tag, parquet_path.name, rows, frame.shape[1], n_source_cols,
                time.monotonic() - started,
            )
        )
        log_row(year, month, url, "ok", rows, n_source_cols)
        return "ok"

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    targets = [(y, m) for y in YEARS for m in MONTHS]

    print("keeping {} columns:".format(len(KEEP_COLUMNS)), flush=True)
    for name, group in COLUMN_GROUPS:
        print("  {:18s} {}".format(name, ", ".join(group)), flush=True)
    todo = [t for t in targets if not (RAW_DIR / "{}_{:02d}.parquet".format(*t)).exists()]
    print(
        "\n{}개월 중 {}개월 남음, 동시 작업 {}개\n".format(
            len(targets), len(todo), WORKERS
        ),
        flush=True,
    )

    started = time.monotonic()
    summary = []
    failures = []

    def worker(index_target):
        index, (year, month) = index_target
        # stagger the opening connections rather than firing all at once
        time.sleep(STAGGER * min(index, WORKERS))
        try:
            return year, month, process_month(year, month), None
        except Stop as exc:
            return year, month, "stopped_error", exc
        except Exception as exc:  # pragma: no cover - unexpected worker crash
            return year, month, "worker_error", exc

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(worker, it) for it in enumerate(targets)]
        done = 0
        for future in as_completed(futures):
            year, month, status, error = future.result()
            summary.append((year, month, status))
            done += 1
            if error is not None:
                failures.append((year, month, error))
                log_row(
                    year,
                    month,
                    URL_TEMPLATE.format(year=year, month=month),
                    status,
                    None,
                )
            say(
                "진행 {}/{}  경과 {:.1f}분".format(
                    done, len(targets), (time.monotonic() - started) / 60.0
                )
            )

    print("\n=== Summary ===", flush=True)
    table = pd.DataFrame(summary, columns=["year", "month", "status"]).sort_values(
        ["year", "month"]
    )
    print(table.to_string(index=False), flush=True)
    print(
        "\nstatus counts:\n{}".format(table["status"].value_counts().to_string()),
        flush=True,
    )
    print("총 소요 {:.1f}분".format((time.monotonic() - started) / 60.0), flush=True)

    if failures:
        print("\n=== FAILED MONTHS ===", file=sys.stderr)
        for year, month, error in failures:
            print("{}-{:02d}: {}".format(year, month, error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
