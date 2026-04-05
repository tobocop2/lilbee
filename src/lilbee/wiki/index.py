"""Wiki index and log management.

Maintains two auto-generated files in the wiki directory:
- index.md: table of contents listing all wiki pages
- log.md: append-only chronological record of wiki events
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from lilbee.config import Config, cfg
from lilbee.wiki.shared import SUBDIR_TO_TYPE

log = logging.getLogger(__name__)


def _wiki_root(config: Config) -> Path:
    return config.data_root / config.wiki_dir


def _parse_title(text: str) -> str:
    """Extract title from frontmatter or first heading."""
    lines = text.splitlines()
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and stripped.startswith("title:"):
            return stripped.removeprefix("title:").strip()
        if not in_frontmatter and stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return ""


def _parse_source_count(text: str) -> int:
    """Count sources from frontmatter sources field."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("sources:"):
            value = stripped.removeprefix("sources:").strip().strip("[]")
            if not value:
                return 0
            return len([s for s in value.split(",") if s.strip()])
    return 0


def update_wiki_index(config: Config | None = None) -> Path:
    """Scan summaries/ and concepts/ directories and write wiki/index.md.

    Returns the path to the generated index file.
    """
    if config is None:
        config = cfg
    root = _wiki_root(config)
    root.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Wiki Index", ""]

    for subdir in ("summaries", "concepts"):
        subdir_path = root / subdir
        if not subdir_path.is_dir():
            continue
        page_type = SUBDIR_TO_TYPE[subdir]
        for md_path in sorted(subdir_path.glob("*.md")):
            text = md_path.read_text(encoding="utf-8")
            title = _parse_title(text) or md_path.stem.replace("-", " ").title()
            source_count = _parse_source_count(text)
            rel = f"{subdir}/{md_path.stem}"
            lines.append(f"- [{title}]({rel}.md) | {page_type} | {source_count} sources")

    lines.append("")  # trailing newline
    index_path = root / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Updated wiki index: %d entries", len(lines) - 3)
    return index_path


def append_wiki_log(
    action: str,
    details: str,
    config: Config | None = None,
) -> Path:
    """Append an entry to wiki/log.md.

    Format: ## [YYYY-MM-DD] action | details

    Returns the path to the log file.
    """
    if config is None:
        config = cfg
    root = _wiki_root(config)
    root.mkdir(parents=True, exist_ok=True)

    log_path = root / "log.md"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = f"## [{timestamp}] {action} | {details}\n\n"

    if not log_path.exists():
        log_path.write_text("# Wiki Log\n\n", encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return log_path
