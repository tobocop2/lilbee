"""Arguments and output files of the driver scripts; stdlib only, so every side can run it."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import psutil
except ImportError:  # a side's environment need not have it
    psutil = None

PAGES_FILE = "pages.jsonl"
RUN_FILE = "run.json"
MODES = ("http", "browser")


@dataclass
class PageOut:
    """One page as a driver reports it."""

    url: str
    markdown: str | None
    error: str | None = None
    saved_at: float | None = None


@dataclass
class RunOut:
    """What a driver measured inside its own process."""

    started: float
    crawl_started: float = 0.0
    crawl_ended: float = 0.0
    versions: dict[str, str] = field(default_factory=dict)
    threads_before: int = 0
    threads_after: int = 0


def crawl_arguments(description: str) -> argparse.Namespace:
    """The arguments every crawl driver takes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", required=True, help="URL to start from")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--depth", type=int, default=1, help="0 fetches the seed only")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds for one page")
    parser.add_argument("--out", required=True, type=Path, help="directory for the result files")
    parser.add_argument(
        "--chrome", default=None, help="browser binary, for a driver that needs one"
    )
    return parser.parse_args()


def serve_conversions(convert: Callable[[str, str], str]) -> None:
    """Answer one JSON line on stdout for each ``{"html", "base_url"}`` line on stdin."""
    for line in sys.stdin:
        request = json.loads(line)
        try:
            answer = {"markdown": convert(request["html"], request["base_url"]), "error": None}
        # A converter failure of any type, a Rust panic included, is the answer for this input.
        except BaseException as failure:
            answer = {"markdown": None, "error": f"{type(failure).__name__}: {failure}"}
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


def thread_count() -> int:
    """The Python threads alive in this process; the count the harness asserts on."""
    return threading.active_count()


def native_thread_count() -> int | None:
    """Every thread of this process, where psutil is installed; reported, never asserted on.

    The system starts and ends worker threads on its own, so two equal runs give two counts.
    """
    return int(psutil.Process().num_threads()) if psutil is not None else None


def write_pages(out: Path, pages: list[PageOut]) -> None:
    """Write one JSON line for each page."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / PAGES_FILE).open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")


def write_run(out: Path, run: RunOut) -> None:
    """Write the run record; its presence says the driver reached its end."""
    out.mkdir(parents=True, exist_ok=True)
    record = {**asdict(run), "native_threads_at_end": native_thread_count()}
    (out / RUN_FILE).write_text(json.dumps(record), encoding="utf-8")


def now() -> float:
    """Wall-clock seconds, the clock file times use."""
    return time.time()
