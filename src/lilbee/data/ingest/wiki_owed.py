"""The mark a sync leaves from its first write until the wiki's page index has its files.

A sync that stops between the two leaves the mark, and the next sync gives the
page index every source written since the time the mark holds.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

WIKI_OWED_FILENAME = "wiki_pass_owed"


def note_wiki_pass_owed(records_root: Path) -> None:
    """Mark the page index owed the sources written from now on; an earlier mark stays.

    The time has the form of a source row's ``ingested_at``, which it is compared with.
    """
    try:
        with (records_root / WIKI_OWED_FILENAME).open("x", encoding="utf-8") as mark:
            mark.write(datetime.now(UTC).isoformat())
    except FileExistsError:
        return


def wiki_pass_owed_since(records_root: Path) -> str | None:
    """The time from which written sources are owed to the page index, or None when none is.

    A mark with no time in it owes every source: an empty time is before each of them.
    """
    try:
        return (records_root / WIKI_OWED_FILENAME).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def clear_wiki_pass_owed(records_root: Path) -> None:
    """Remove the mark: the page index has every source written so far."""
    (records_root / WIKI_OWED_FILENAME).unlink(missing_ok=True)
