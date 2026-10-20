"""Writes the crawl metadata sidecar in the format released lilbee versions wrote it."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.data.page_record import CrawlMeta

CRAWL_META_FILE = "crawl_meta.json"


def write_crawl_metadata(meta: dict[str, CrawlMeta]) -> Path:
    """Write *meta* as the URL-keyed JSON sidecar under the data directory; its path."""
    path = cfg.data_dir / CRAWL_META_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {url: asdict(entry) for url, entry in meta.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path
