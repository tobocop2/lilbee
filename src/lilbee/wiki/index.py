"""Wiki index and log management.

Maintains two auto-generated files in the wiki directory:
- index.md: table of contents listing all wiki pages, grouped by type
- log.md: append-only chronological record of wiki events

index.md is regenerated end-to-end on every call. log.md is append-only
so the history survives rebuilds; each entry starts with
``## [YYYY-MM-DD HH:MM]`` so simple grep patterns still work.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from lilbee.config import Config, cfg
from lilbee.wiki.shared import (
    CONCEPTS_SUBDIR,
    ENTITIES_SUBDIR,
    SUBDIR_TO_TYPE,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
    WIKI_TYPE_HEADINGS,
    parse_frontmatter,
)

log = logging.getLogger(__name__)

_INDEX_SECTION_ORDER: tuple[str, ...] = (
    CONCEPTS_SUBDIR,
    ENTITIES_SUBDIR,
    SUMMARIES_SUBDIR,
    SYNTHESIS_SUBDIR,
)


def _wiki_root(config: Config) -> Path:
    return config.data_root / config.wiki_dir


def parse_title(text: str) -> str:
    """Extract title from YAML frontmatter ``title`` field or first H1 heading.

    Assumes wiki/Obsidian markdown conventions. Returns the empty string
    when neither is present.
    """
    return _title_from_frontmatter(parse_frontmatter(text), text)


def _title_from_frontmatter(fm: dict[str, object], text: str) -> str:
    """Return ``fm['title']`` when present, else the first H1 heading, else ``""``."""
    if "title" in fm:
        return str(fm["title"])
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return ""


# Backwards-compat alias for the former private name.
_parse_title = parse_title


def parse_source_count(text: str) -> int:
    """Count sources from frontmatter sources field."""
    return _source_count_from_frontmatter(parse_frontmatter(text))


def _source_count_from_frontmatter(fm: dict[str, object]) -> int:
    """Count entries in the ``sources`` frontmatter field."""
    sources = fm.get("sources")
    if isinstance(sources, list):  # yaml.safe_load may return str or list
        return len(sources)
    if isinstance(sources, str):  # yaml.safe_load may return str or list
        return len([s for s in sources.split(",") if s.strip()])
    return 0


def update_wiki_index(config: Config | None = None) -> Path:
    """Regenerate wiki/index.md, grouping pages by type.

    Sections appear in a fixed order (Concepts, Entities, Source
    Summaries, Synthesis). Empty sections are omitted. Each entry keeps
    the ``[title](subdir/slug.md) | type | N sources`` format so
    readers and existing tooling stay stable.
    """
    if config is None:
        config = cfg
    root = _wiki_root(config)
    root.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Wiki Index", ""]
    total = 0
    for subdir in _INDEX_SECTION_ORDER:
        section_lines = _render_section(root, subdir)
        if not section_lines:
            continue
        lines.append(f"## {WIKI_TYPE_HEADINGS[SUBDIR_TO_TYPE[subdir]]}")
        lines.append("")
        lines.extend(section_lines)
        lines.append("")
        total += len(section_lines)

    lines.append("")  # trailing newline
    index_path = root / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Updated wiki index: %d entries", total)
    return index_path


def _render_section(root: Path, subdir: str) -> list[str]:
    """Return formatted index lines for one subdir (empty if the subdir has no pages).

    Parses each file's frontmatter once and reuses it for title and
    source-count, halving file-read / YAML-parse work on a wiki with
    hundreds of pages.
    """
    subdir_path = root / subdir
    if not subdir_path.is_dir():
        return []
    page_type = SUBDIR_TO_TYPE[subdir]
    lines: list[str] = []
    for md_path in sorted(subdir_path.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        title = _title_from_frontmatter(fm, text) or md_path.stem.replace("-", " ").title()
        source_count = _source_count_from_frontmatter(fm)
        rel = md_path.relative_to(root).with_suffix("").as_posix()
        lines.append(f"- [{title}]({rel}.md) | {page_type} | {source_count} sources")
    return lines


def append_wiki_log(
    action: str,
    details: str,
    config: Config | None = None,
) -> Path:
    """Append an entry to wiki/log.md.

    Format: ``## [YYYY-MM-DD HH:MM] action | details``. The minute-level
    timestamp means audit entries written within the same build each
    have their own line and ``grep '## \\[2026-04-22'`` still works.
    Returns the path to the log file.
    """
    if config is None:
        config = cfg
    root = _wiki_root(config)
    root.mkdir(parents=True, exist_ok=True)

    log_path = root / "log.md"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    entry = f"## [{timestamp}] {action} | {details}\n\n"

    if not log_path.exists():
        log_path.write_text("# Wiki Log\n\n", encoding="utf-8")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return log_path
