"""Tests for the failed-file skip-marker sidecar.

The marker file makes a previously-failed file invisible to the next sync
until the file content changes (its hash differs) or the user runs
``/sync --force-rebuild``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.data.ingest.skip_marker import (
    SKIP_MARKER_FILENAME,
    SKIP_REASON_FILENAME,
    clear_skip_markers,
    load_skip_markers,
    load_skip_reasons,
    write_skip_markers,
    write_skip_reasons,
)


def test_load_empty_when_file_missing(tmp_path: Path) -> None:
    """A fresh data root has no markers; load returns an empty dict (not None)."""
    assert load_skip_markers(tmp_path) == {}


def test_round_trip(tmp_path: Path) -> None:
    """write → load returns the same mapping."""
    write_skip_markers(tmp_path, {"foo.txt": "deadbeef", "bar.pdf": "cafef00d"})
    assert load_skip_markers(tmp_path) == {"foo.txt": "deadbeef", "bar.pdf": "cafef00d"}


def test_write_overwrites_previous_state(tmp_path: Path) -> None:
    """A subsequent write replaces the file (no merging at this layer)."""
    write_skip_markers(tmp_path, {"a": "1"})
    write_skip_markers(tmp_path, {"b": "2"})
    assert load_skip_markers(tmp_path) == {"b": "2"}


def test_clear_removes_marker_file(tmp_path: Path) -> None:
    """clear_skip_markers deletes the file; load then returns empty."""
    write_skip_markers(tmp_path, {"foo": "x"})
    assert (tmp_path / SKIP_MARKER_FILENAME).exists()
    clear_skip_markers(tmp_path)
    assert not (tmp_path / SKIP_MARKER_FILENAME).exists()
    assert load_skip_markers(tmp_path) == {}


def test_clear_is_idempotent_when_missing(tmp_path: Path) -> None:
    """Clearing an absent marker file does not raise."""
    clear_skip_markers(tmp_path)  # no exception
    assert load_skip_markers(tmp_path) == {}


def test_clear_logs_and_continues_on_unlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the marker file can't be removed (e.g. locked on Windows), clear logs and returns."""
    write_skip_markers(tmp_path, {"f": "h"})

    def _raise(_self: Path, *, missing_ok: bool = False) -> None:
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(Path, "unlink", _raise)
    clear_skip_markers(tmp_path)  # must not raise
    # The file is still there because the unlink was blocked, proving we hit the
    # except branch rather than silently succeeding.
    assert (tmp_path / SKIP_MARKER_FILENAME).exists()


def test_load_handles_corrupt_json(tmp_path: Path) -> None:
    """A corrupted marker file is treated as empty so a single bad write
    doesn't lock the user into retrying every file forever."""
    marker = tmp_path / SKIP_MARKER_FILENAME
    marker.write_text("not json {{{", encoding="utf-8")
    assert load_skip_markers(tmp_path) == {}


def test_load_rejects_non_string_values(tmp_path: Path) -> None:
    """Filenames with non-string hash values are filtered out (defensive)."""
    import json

    marker = tmp_path / SKIP_MARKER_FILENAME
    marker.write_text(json.dumps({"good": "hash", "bad": 42}), encoding="utf-8")
    assert load_skip_markers(tmp_path) == {"good": "hash"}


def test_load_rejects_non_dict_top_level(tmp_path: Path) -> None:
    """A list (or any non-dict) at top level is treated as no markers."""
    import json

    marker = tmp_path / SKIP_MARKER_FILENAME
    marker.write_text(json.dumps(["this", "should", "be", "a", "dict"]), encoding="utf-8")
    assert load_skip_markers(tmp_path) == {}


def test_write_creates_parent_directory(tmp_path: Path) -> None:
    """write_skip_markers mkdir-s the data root if it doesn't exist yet."""
    nested = tmp_path / "newly_created"
    assert not nested.exists()
    write_skip_markers(nested, {"f": "h"})
    assert (nested / SKIP_MARKER_FILENAME).exists()


def test_write_is_atomic_via_tmp_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed write cleans up the temp file instead of leaving it behind.

    Simulate the rare case where os.replace fails (e.g. another process holds
    the file open on Windows). The function should log and continue, not leak
    the .tmp sidecar.
    """
    import os

    def _raise(_src: str, _dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", _raise)
    write_skip_markers(tmp_path, {"k": "v"})
    leftover = list(tmp_path.glob(f"{SKIP_MARKER_FILENAME}.tmp"))
    assert leftover == [], f"tmp file leaked: {leftover}"


class TestSkipReasons:
    """The reasons sidecar records WHY a file was skipped (filename -> error),
    so a report can show the cause, not just the hash. Separate from the
    hash-keyed markers, which drive the resume logic."""

    def test_load_empty_when_file_missing(self, tmp_path: Path) -> None:
        assert load_skip_reasons(tmp_path) == {}

    def test_round_trip(self, tmp_path: Path) -> None:
        reasons = {
            "a.pdf": "OCR timed out after 120s",
            "b.tiff": "no text extracted (0 chunks)",
        }
        write_skip_reasons(tmp_path, reasons)
        assert load_skip_reasons(tmp_path) == reasons
        assert (tmp_path / SKIP_REASON_FILENAME).exists()

    def test_reasons_file_is_separate_from_markers(self, tmp_path: Path) -> None:
        # The two sidecars are independent files; writing one leaves the other.
        write_skip_markers(tmp_path, {"a.pdf": "deadbeef"})
        write_skip_reasons(tmp_path, {"a.pdf": "decode failure"})
        assert (tmp_path / SKIP_MARKER_FILENAME) != (tmp_path / SKIP_REASON_FILENAME)
        assert load_skip_markers(tmp_path) == {"a.pdf": "deadbeef"}
        assert load_skip_reasons(tmp_path) == {"a.pdf": "decode failure"}

    def test_clear_removes_reasons_too(self, tmp_path: Path) -> None:
        # Clearing skip state (force-rebuild / retry-skipped) must drop the
        # reasons too, or stale errors linger after a clean re-run.
        write_skip_markers(tmp_path, {"a.pdf": "deadbeef"})
        write_skip_reasons(tmp_path, {"a.pdf": "decode failure"})
        clear_skip_markers(tmp_path)
        assert not (tmp_path / SKIP_REASON_FILENAME).exists()
        assert load_skip_reasons(tmp_path) == {}

    def test_load_handles_corrupt_json(self, tmp_path: Path) -> None:
        (tmp_path / SKIP_REASON_FILENAME).write_text("not json {{{", encoding="utf-8")
        assert load_skip_reasons(tmp_path) == {}
