"""The TUI slash-command table in ``docs/usage.md`` must list the live registry.

Every slash command is defined once in ``command_registry.COMMANDS``. The usage
guide's table is hand-written prose drawn from each command's own help text, so
nothing keeps the two in sync automatically: a new command can ship, or an old
one can go away, and the table drifts silently either way. This parses the
committed table and asserts its command names are exactly the registry's.
"""

from __future__ import annotations

import re
from pathlib import Path

from lilbee.cli.tui.command_registry import COMMANDS

REPO_ROOT = Path(__file__).resolve().parent.parent
USAGE_GUIDE = REPO_ROOT / "docs" / "usage.md"

_SECTION_HEADING = "### Slash commands"
_TABLE_HEADER_PREFIX = "| Command"
_TABLE_SEPARATOR_PREFIX = "|---"

# A row's first cell holds the command name, optionally followed by an args
# hint inside the same backticks, e.g. `` `/model [name]` `` or `` `/status` ``.
_ROW_NAME_RE = re.compile(r"^\|\s*`(/\S+?)(?:\s[^`]*)?`\s*\|")


def _parse_slash_command_rows(lines: list[str]) -> list[str]:
    """Return the command name from each data row of the Slash commands table."""
    start = lines.index(_SECTION_HEADING)
    names: list[str] = []
    in_table = False
    for line in lines[start + 1 :]:
        if line.startswith("### "):
            break
        if not in_table:
            if line.startswith(_TABLE_HEADER_PREFIX):
                in_table = True
            continue
        if line.startswith(_TABLE_SEPARATOR_PREFIX):
            continue
        if not line.startswith("|"):
            break
        match = _ROW_NAME_RE.match(line)
        assert match, f"table row does not match the expected `/command` shape: {line!r}"
        names.append(match.group(1))
    return names


def _slash_command_table_rows() -> list[str]:
    """Return the command name from each row of the committed usage guide's table."""
    return _parse_slash_command_rows(USAGE_GUIDE.read_text(encoding="utf-8").splitlines())


def test_every_registered_command_has_a_row() -> None:
    documented = set(_slash_command_table_rows())
    registered = {cmd.name for cmd in COMMANDS}
    missing = registered - documented
    assert not missing, f"docs/usage.md is missing rows for: {sorted(missing)}"


def test_no_row_for_a_command_that_does_not_exist() -> None:
    documented = set(_slash_command_table_rows())
    registered = {cmd.name for cmd in COMMANDS}
    extra = documented - registered
    assert not extra, f"docs/usage.md documents commands that are not registered: {sorted(extra)}"


def test_table_has_no_duplicate_rows() -> None:
    names = _slash_command_table_rows()
    assert len(names) == len(set(names)), "docs/usage.md lists the same slash command twice"


def test_table_parser_has_power() -> None:
    """A parser that always returns an empty list would pass the tests above for
    the wrong reason: prove it actually reads rows off the real file."""
    names = _slash_command_table_rows()
    assert len(names) > 20
    assert "/help" in names


def test_a_section_with_no_table_yields_no_rows() -> None:
    """A Slash-commands section that ends before any table starts must not
    scan into the next section looking for one."""
    lines = ["### Slash commands", "", "### Next heading", "irrelevant"]
    assert _parse_slash_command_rows(lines) == []
