"""Playwright Chromium detection and install."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from lilbee._frozen import is_frozen
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    SetupDoneEvent,
    SetupProgressEvent,
    SetupStartEvent,
)


class CrawlerBrowserError(RuntimeError):
    """Playwright is installed but its Chromium browser binary is not."""


class CrawlerBackendError(RuntimeError):
    """The ``crawler`` extra (crawl4ai) was never installed."""


_CHROMIUM_COMPONENT = "chromium"
# Rough size estimate for the Chromium download; Playwright bundles vary
# slightly per platform but this gives the UI a decent denominator before
# 'Total bytes' parses out of stdout.
_CHROMIUM_ESTIMATE_MB = 180
_CHROMIUM_SIZE_ESTIMATE_BYTES = _CHROMIUM_ESTIMATE_MB * 1024 * 1024

# Unit -> bytes scale for Playwright stdout progress lines.
_BYTE_UNIT_SCALE: dict[str, int] = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024 * 1024,
    "mib": 1024 * 1024,
}

# Playwright 1.58 prints lines like
# ``|■■■■■■■■                                  |  10% of 162.3 MiB`` during
# the chromium download. The percent comes first, then "of <total> <unit>".
_PROGRESS_LINE_RE = re.compile(
    r"(\d+)\s*%\s*of\s*(\d+(?:\.\d+)?)\s*(MiB|Mb|MB|KiB|KB|B)",
    re.IGNORECASE,
)


def _browsers_cache_path() -> Path:
    """Return the root path where Playwright stores browser binaries."""
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def _read_chromium_revision(browsers_json: Path) -> str | None:
    """Pull the ``chromium`` revision out of a Playwright ``browsers.json``."""
    import json

    try:
        data = json.loads(browsers_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for browser in data.get("browsers", []):
        if browser.get("name") == _CHROMIUM_COMPONENT:
            revision = browser.get("revision")
            return str(revision) if revision is not None else None
    return None


def _expected_chromium_revision() -> str | None:
    """Revision string Playwright was built against (e.g. ``"1217"``).

    ``None`` means "unknown" -- treat as "do install" so a missing
    browsers.json never short-circuits the bootstrap path.
    """
    try:
        import playwright as _pw
    except ImportError:
        return None
    for path in Path(_pw.__file__).parent.rglob("browsers.json"):
        return _read_chromium_revision(path)
    return None


def chromium_installed() -> bool:
    """Return True if the Chromium revision Playwright expects is on disk.

    Matching by any ``chromium-*`` directory isn't enough: when the
    system has chromium-1217 but the bundled Playwright driver expects
    chromium-1208, launch fails with ``Executable doesn't exist`` even
    though the bootstrap check thought everything was ready.
    """
    root = _browsers_cache_path()
    if not root.exists():
        return False
    expected = _expected_chromium_revision()
    if expected is None:
        return any(p.is_dir() and p.name.startswith("chromium-") for p in root.iterdir())
    return (root / f"{_CHROMIUM_COMPONENT}-{expected}").is_dir()


def crawler_browsers_path() -> Path:
    """Public accessor for the crawler browser cache root.

    Used by the HTTP status endpoint to tell plugins where Chromium
    lives. The underlying resolver stays private because callers should
    not depend on the Playwright-specific directory layout.
    """
    return _browsers_cache_path()


def _bytes_from_stdout(line: str) -> tuple[int, int] | None:
    """Extract (downloaded_bytes, total_bytes) from a Playwright stdout line.

    Matches the ``NN% of N.N MiB`` shape Playwright 1.58+ emits for the
    Chromium download. Returns None when the line doesn't match. The
    percent and total both parse out of the same line so callers never
    have to handle a missing total.
    """
    match = _PROGRESS_LINE_RE.search(line)
    if match is None:
        return None
    pct = int(match.group(1))
    raw_total = float(match.group(2))
    unit = match.group(3).lower()
    scale = _BYTE_UNIT_SCALE.get(unit, 1)
    total = int(raw_total * scale)
    downloaded = int(total * pct / 100)
    return downloaded, total


def _emit_setup_start(on_progress: DetailedProgressCallback | None) -> None:
    if on_progress is None:
        return
    on_progress(
        EventType.SETUP_START,
        SetupStartEvent(
            component=_CHROMIUM_COMPONENT,
            size_estimate_bytes=_CHROMIUM_SIZE_ESTIMATE_BYTES,
        ),
    )


def _emit_setup_done(
    on_progress: DetailedProgressCallback | None,
    *,
    success: bool,
    error: str | None,
) -> None:
    if on_progress is None:
        return
    on_progress(
        EventType.SETUP_DONE,
        SetupDoneEvent(component=_CHROMIUM_COMPONENT, success=success, error=error),
    )


async def _drain_stdout_to_progress(
    stream: asyncio.StreamReader,
    on_progress: DetailedProgressCallback | None,
) -> None:
    while True:
        line_bytes = await stream.readline()
        if not line_bytes:
            return
        line = line_bytes.decode(errors="replace").rstrip()
        parsed = _bytes_from_stdout(line)
        if parsed is None or on_progress is None:
            continue
        downloaded, total = parsed
        on_progress(
            EventType.SETUP_PROGRESS,
            SetupProgressEvent(
                component=_CHROMIUM_COMPONENT,
                downloaded_bytes=downloaded,
                total_bytes=total,
                detail=line,
            ),
        )


async def _drain_stderr(stream: asyncio.StreamReader, tail: list[str]) -> None:
    while True:
        line_bytes = await stream.readline()
        if not line_bytes:
            return
        tail.append(line_bytes.decode(errors="replace").rstrip())


_PLAYWRIGHT_MISSING_HINT = (
    "Chromium bootstrap requires the playwright Python package, which is "
    "bundled with the release binary and the lilbee[crawler] extra. "
    "Reinstall with 'pip install lilbee[crawler]' or download a fresh "
    "release binary."
)


def _resolve_playwright_runner() -> tuple[list[str], dict[str, str]]:
    """Return ``(argv_prefix, env)`` for invoking ``playwright install chromium``.

    Spawns Playwright's bundled Node driver directly so the call works under a
    pip install, ``uv tool install``, or a frozen (Nuitka onefile) binary. Falls
    back to ``[sys.executable, '-m', 'playwright']`` for unfrozen builds when the
    driver lookup fails; re-raises for frozen builds, where ``sys.executable``
    is the lilbee exe and ``-m playwright`` would leak into typer.
    """
    try:
        from playwright._impl._driver import compute_driver_executable, get_driver_env
    except ImportError as exc:
        raise CrawlerBrowserError(_PLAYWRIGHT_MISSING_HINT) from exc
    try:
        driver_exe, driver_cli = compute_driver_executable()
    except Exception:
        if not is_frozen():
            return [sys.executable, "-m", "playwright"], dict(os.environ)
        raise
    return [str(driver_exe), str(driver_cli)], dict(get_driver_env())


async def bootstrap_chromium(
    on_progress: DetailedProgressCallback | None = None,
) -> None:
    """Run ``playwright install chromium`` as a subprocess, emitting events.

    Short-circuits when ``chromium_installed()`` is already True. Emits
    ``setup_start`` before spawning, ``setup_progress`` for each recognizable
    progress line on stdout, and ``setup_done`` on exit (``success=False`` plus
    the subprocess stderr tail on failure). Raises :class:`CrawlerBrowserError`
    with the tail so task workers route to FAILED cleanly.
    """
    if chromium_installed():
        _emit_setup_done(on_progress, success=True, error=None)
        return

    _emit_setup_start(on_progress)

    try:
        runner, runner_env = _resolve_playwright_runner()
    except CrawlerBrowserError as exc:
        _emit_setup_done(on_progress, success=False, error=str(exc))
        raise

    proc = await asyncio.create_subprocess_exec(
        *runner,
        "install",
        "chromium",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=runner_env,
    )
    # mypy narrowing: asyncio.create_subprocess_exec with PIPE guarantees
    # non-None streams at runtime; the asserts only satisfy the type checker.
    assert proc.stdout is not None  # noqa: S101
    assert proc.stderr is not None  # noqa: S101

    stderr_tail: list[str] = []
    await asyncio.gather(
        _drain_stdout_to_progress(proc.stdout, on_progress),
        _drain_stderr(proc.stderr, stderr_tail),
    )
    returncode = await proc.wait()

    if returncode != 0:
        tail = "\n".join(stderr_tail[-10:]) or f"exit code {returncode}"
        _emit_setup_done(on_progress, success=False, error=tail)
        raise CrawlerBrowserError(f"Chromium bootstrap failed (exit {returncode}): {tail}")

    _emit_setup_done(on_progress, success=True, error=None)
