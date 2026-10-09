"""Crawl with a lilbee checkout's own entry point; run with that checkout's interpreter.

Two changes are made, the same on every side: loopback addresses are allowed, and the sync
that follows a crawl is replaced by one that does nothing, so no embedding model runs.
"""

from __future__ import annotations

import importlib.metadata
import ipaddress
import json
import os
import sys

from _driver_io import PageOut, RunOut, crawl_arguments, now, thread_count, write_pages, write_run

STARTED = now()
ARGS = crawl_arguments(__doc__ or "")
DATA_DIR = ARGS.out / "lilbee-data"
os.environ.update(
    {
        "LILBEE_DATA": str(DATA_DIR),
        "LILBEE_SKIP_TOML_CONFIG": "1",
        "LILBEE_NO_SPLASH": "1",
        "LILBEE_CRAWL_SYNC_INTERVAL": "0",
        "LILBEE_CRAWL_MEAN_DELAY": "0",
        "LILBEE_CRAWL_MAX_DELAY_RANGE": "0",
        "LILBEE_CRAWL_RENDER_MODE": ARGS.mode,
        "LILBEE_CRAWL_TIMEOUT": str(int(ARGS.timeout)),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "COLUMNS": "200",
    }
)

# lilbee reads its settings from the environment at import, so these imports follow it.
import lilbee.data.ingest as ingest  # noqa: E402
from lilbee.crawler import url_filter  # noqa: E402
from lilbee.data.types import SyncResult  # noqa: E402
from lilbee.runtime.launcher import main  # noqa: E402

LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))
CRAWLER_PACKAGES = ("lilbee", "crawl4ai", "crawlberg", "playwright")


async def _no_sync(*_args: object, **_kwargs: object) -> SyncResult:
    return SyncResult()


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for package in CRAWLER_PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def _saved_pages() -> list[PageOut]:
    """The pages lilbee saved, read from its own crawl record."""
    meta_path = DATA_DIR / "data" / "crawl_meta.json"
    if not meta_path.is_file():
        return []
    web_root = DATA_DIR / "documents" / "_web"
    pages: list[PageOut] = []
    for url, entry in json.loads(meta_path.read_text(encoding="utf-8")).items():
        saved = web_root / entry["file"]
        if saved.is_file():
            pages.append(
                PageOut(url, saved.read_text(encoding="utf-8"), saved_at=saved.stat().st_mtime)
            )
    return pages


def run() -> int:
    """Run ``lilbee add`` and write what it saved."""
    allowed = tuple(net for net in url_filter.get_blocked_networks() if net not in LOOPBACK)
    url_filter.get_blocked_networks = lambda: allowed  # type: ignore[assignment]
    ingest.sync = _no_sync  # type: ignore[assignment]
    argv = ["lilbee", "add", ARGS.seed]
    if ARGS.depth > 0:
        argv += ["--crawl", "--depth", str(ARGS.depth)]
    sys.argv = argv
    record = RunOut(started=STARTED, versions=_versions(), threads_before=thread_count())
    record.crawl_started = now()
    code: int | str | None = 0
    try:
        main()
    except SystemExit as exit_request:
        code = exit_request.code
    record.crawl_ended = now()
    record.threads_after = thread_count()
    write_pages(ARGS.out, _saved_pages())
    write_run(ARGS.out, record)
    return code if isinstance(code, int) else int(code is not None)  # SystemExit.code is untyped


if __name__ == "__main__":
    sys.exit(run())
