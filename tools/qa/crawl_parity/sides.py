"""The two sides as configured interpreters, and the run of one driver with its measurements."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import psutil
from tools.qa.crawl_parity import leftover
from tools.qa.crawl_parity.leftover import LeftBehind, Tracker
from tools.qa.crawl_parity.model import Layer, Mode, Page, Side

DRIVERS_DIR = Path(__file__).parent / "drivers"
LILBEE_DRIVER = "lilbee_crawl.py"
PAGES_FILE = "pages.jsonl"
RUN_FILE = "run.json"
STDERR_TAIL_CHARS = 2000
KILLED = -9
NEUTRAL_ORIGIN = "http://replay.test"
# lilbee logs one line for a page it does not save; the reason is the text after the URL.
_LILBEE_FAILURE = re.compile(r"Crawled page yields no content: (\S+?): (.*)$", re.MULTILINE)


@dataclass(frozen=True)
class SideConfig:
    """The interpreters and driver scripts of one side; a layer with no interpreter is not run."""

    side: Side
    label: str
    interpreters: dict[Layer, Path]
    crawler_driver: str
    converter_driver: str
    chrome: str | None = None

    def has(self, layer: Layer) -> bool:
        """Whether this side can run *layer*."""
        return layer in self.interpreters

    def crawl_driver(self, layer: Layer) -> Path:
        """The script that crawls at *layer*."""
        return DRIVERS_DIR / (LILBEE_DRIVER if layer is Layer.LILBEE else self.crawler_driver)


@dataclass(frozen=True)
class CrawlRequest:
    """One crawl to run: where it starts and how."""

    seed_url: str
    mode: Mode
    depth: int = 1
    page_timeout: float = 30.0
    run_timeout: float = 600.0


@dataclass(frozen=True)
class CrawlResult:
    """What one driver process returned and what it left behind."""

    side: Side
    layer: Layer
    mode: Mode
    pages: dict[str, Page]
    return_code: int
    wall_seconds: float
    crawl_seconds: float | None
    first_page_seconds: float | None
    versions: dict[str, str]
    threads_started: int | None
    left: LeftBehind
    load_before: float
    stderr_tail: str
    work: Path = Path()

    def saved(self) -> dict[str, Page]:
        """The pages that have markdown, by corpus path."""
        return {path: page for path, page in self.pages.items() if page.markdown}


def load_sides(path: Path) -> dict[Side, SideConfig]:
    """The sides a ``sides.toml`` names."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    sides: dict[Side, SideConfig] = {}
    for side in Side:
        table = document.get(side.value)
        if table is None:
            continue
        interpreters = {layer: Path(table[layer.value]) for layer in Layer if layer.value in table}
        sides[side] = SideConfig(
            side=side,
            label=str(table.get("label", side.value)),
            interpreters=interpreters,
            crawler_driver=str(table.get("crawler_driver", "")),
            converter_driver=str(table.get("converter_driver", "")),
            chrome=table.get("chrome"),
        )
    return sides


def path_of(url: str) -> str:
    """The corpus path of a URL, query included."""
    parts = urlsplit(url)
    return (parts.path or "/") + (f"?{parts.query}" if parts.query else "")


def load_average() -> float:
    """The one-minute load average divided by the number of processors."""
    return float(psutil.getloadavg()[0]) / (psutil.cpu_count() or 1)


def origin_of(url: str) -> str:
    """Scheme, host and port of *url*."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def neutral_origin(markdown: str | None, origin: str) -> str | None:
    """*markdown* with the replay origin written as one fixed name (``replay-origin``).

    The replay port changes from run to run; a signature must not.
    """
    return markdown.replace(origin, NEUTRAL_ORIGIN) if markdown else markdown


def _read_pages(out: Path, origin: str) -> dict[str, Page]:
    pages_file = out / PAGES_FILE
    if not pages_file.is_file():
        return {}
    pages: dict[str, Page] = {}
    for line in pages_file.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        markdown = neutral_origin(entry["markdown"], origin)
        page = Page(entry["url"], markdown, entry["error"], entry["saved_at"])
        known = pages.get(path_of(page.url))
        if known is None or (page.markdown and not known.markdown):
            pages[path_of(page.url)] = page
    return pages


def _add_logged_failures(pages: dict[str, Page], stderr: str) -> None:
    """Add the pages lilbee logged as not saved, with the reason it logged."""
    for url, reason in _LILBEE_FAILURE.findall(stderr):
        pages.setdefault(path_of(url), Page(url, None, reason.strip()))


def _command(config: SideConfig, layer: Layer, request: CrawlRequest, out: Path) -> list[str]:
    command = [
        str(config.interpreters[layer]),
        str(config.crawl_driver(layer)),
        "--seed",
        request.seed_url,
        "--mode",
        request.mode.value,
        "--depth",
        str(request.depth),
        "--timeout",
        str(request.page_timeout),
        "--out",
        str(out),
    ]
    if config.chrome:
        command += ["--chrome", config.chrome]
    return command


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill a driver that ran past its limit, with the children it has now."""
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    process.kill()
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            continue


def run_crawl(config: SideConfig, layer: Layer, request: CrawlRequest, work: Path) -> CrawlResult:
    """Run one crawl driver in its own process, in a directory no earlier run has used."""
    work.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{work.name}.", dir=work.parent))
    out = work / "out"
    tracker = Tracker(work / "tmp")
    environment = {**_clean_environment(), **tracker.environment()}
    load_before = load_average()
    started = time.time()
    with (work / "stdout.txt").open("wb") as stdout, (work / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(
            _command(config, layer, request, out),
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            env=environment,
            cwd=work,
        )
        tracker.watch(process.pid)
        try:
            return_code = process.wait(timeout=request.run_timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            process.wait()
            return_code = KILLED
    wall = time.time() - started
    left = tracker.collect(leftover.GRACE_SECONDS)
    stderr_text = (work / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    pages = _read_pages(out, origin_of(request.seed_url))
    if layer is Layer.LILBEE:
        _add_logged_failures(pages, stderr_text)
    record = _read_run(out)
    saved_times = [page.saved_at for page in pages.values() if page.markdown and page.saved_at]
    return CrawlResult(
        side=config.side,
        layer=layer,
        mode=request.mode,
        pages=pages,
        return_code=return_code,
        wall_seconds=wall,
        crawl_seconds=record.crawl_ended - record.crawl_started if record else None,
        first_page_seconds=min(saved_times) - record.crawl_started
        if saved_times and record
        else None,
        versions=record.versions if record else {},
        threads_started=record.threads_after - record.threads_before if record else None,
        left=left,
        load_before=load_before,
        stderr_tail=stderr_text[-STDERR_TAIL_CHARS:],
        work=work,
    )


@dataclass(frozen=True)
class _RunRecord:
    """What a driver measured inside its own process."""

    crawl_started: float
    crawl_ended: float
    versions: dict[str, str]
    threads_before: int
    threads_after: int


def _read_run(out: Path) -> _RunRecord | None:
    """The driver's own record; None when the driver did not reach its end."""
    run_file = out / RUN_FILE
    if not run_file.is_file():
        return None
    record = json.loads(run_file.read_text(encoding="utf-8"))
    return _RunRecord(
        record["crawl_started"],
        record["crawl_ended"],
        record["versions"],
        record["threads_before"],
        record["threads_after"],
    )


def _clean_environment() -> dict[str, str]:
    """The harness's environment without settings that would steer a side."""
    blocked = ("LILBEE_", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    return {name: value for name, value in os.environ.items() if not name.startswith(blocked)}
