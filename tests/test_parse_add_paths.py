"""Tests for the ``/add`` argument-to-path parser used by the chat screen."""

from __future__ import annotations

from pathlib import Path

import lilbee.cli.tui.screens.chat as chat_mod
from lilbee.cli.tui.screens.chat import _parse_add_paths


def test_single_path_with_spaces_and_apostrophe(tmp_path: Path) -> None:
    # macOS-style filename with spaces AND an apostrophe; shlex would split or
    # reject it, so an existing whole-arg path must be taken as one path.
    target = tmp_path / "Star Wars Collector's Edition.pdf"
    target.write_text("x")
    assert _parse_add_paths(str(target)) == [target]


def test_multiple_space_separated_existing_paths(tmp_path: Path) -> None:
    # The whole arg doesn't exist as one file, so it falls back to shell-style
    # splitting into the two separate existing paths.
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    assert _parse_add_paths(f"{a} {b}") == [a, b]


def test_quoted_path(tmp_path: Path) -> None:
    # A single quoted token with a space resolves to one path via shlex once the
    # whole-arg (with surrounding quotes stripped) does not itself exist.
    target = tmp_path / "my notes.md"
    target.write_text("x")
    assert _parse_add_paths(f'"{target}"') == [target]


def test_shlex_failure_falls_back_to_single_literal_path() -> None:
    # An unbalanced quote in a path that doesn't exist makes shlex.split raise;
    # the parser treats the (quote-stripped) argument as one literal path.
    raw = "/tmp/does not 'exist.pdf"
    result = _parse_add_paths(raw)
    assert result == [Path(raw.strip().strip('"').strip("'")).expanduser()]
    assert len(result) == 1


def test_windows_strips_quotes_from_split_tokens(monkeypatch) -> None:
    # On Windows shlex runs with posix=False, which keeps quotes on tokens; the
    # parser strips them so each token is a clean path. Force the nt branch.
    monkeypatch.setattr(chat_mod.os, "name", "nt")
    result = _parse_add_paths('"C:/a.txt" "C:/b.txt"')
    assert result == [Path("C:/a.txt").expanduser(), Path("C:/b.txt").expanduser()]
