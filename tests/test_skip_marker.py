"""Tests for the failed-file skip-marker sidecar.

The marker file makes a previously-failed file invisible to the next sync
until the file content changes (its hash differs) or the user runs
``/sync --force-rebuild``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from lilbee.data.ingest import skip_marker
from lilbee.data.ingest.skip_marker import (
    DEFAULT_SKIP_REASON,
    REMOVED_SKIP_REASON,
    SKIP_KIND_FILENAME,
    SKIP_MARKER_FILENAME,
    SKIP_REASON_FILENAME,
    SkipKind,
    SkipRecords,
    clear_failed_markers,
    clear_skip_markers,
    describe_skips,
    held_out_names,
    load_skip_kinds,
    load_skip_markers,
    load_skip_reasons,
    mark_removed,
    update_skip_records,
    write_skip_markers,
    write_skip_reasons,
    write_skip_records,
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
    marker = tmp_path / SKIP_MARKER_FILENAME
    marker.write_text(json.dumps({"good": "hash", "bad": 42}), encoding="utf-8")
    assert load_skip_markers(tmp_path) == {"good": "hash"}


def test_load_rejects_non_dict_top_level(tmp_path: Path) -> None:
    """A list (or any non-dict) at top level is treated as no markers."""
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


class TestDescribeSkips:
    """Pairing held-out filenames with their reasons, for the sync summary and
    the status payload."""

    def test_pairs_names_with_reasons_in_the_given_order(self, tmp_path: Path) -> None:
        write_skip_reasons(tmp_path, {"a.pdf": "OCR timed out", "b.tiff": "decode failure"})
        described = describe_skips(tmp_path, ["b.tiff", "a.pdf"])
        assert [(d.filename, d.reason) for d in described] == [
            ("b.tiff", "decode failure"),
            ("a.pdf", "OCR timed out"),
        ]

    def test_falls_back_when_no_reason_was_recorded(self, tmp_path: Path) -> None:
        # A marker written before the reasons sidecar existed still has to say
        # something, or the user sees a held-out file with a blank explanation.
        described = describe_skips(tmp_path, ["orphan.pdf"])
        assert [(d.filename, d.reason) for d in described] == [("orphan.pdf", DEFAULT_SKIP_REASON)]


class TestUpdateSkipRecords:
    """A read-modify-write of both sidecars as they are on disk, under a lock."""

    def test_changes_both_sidecars_from_what_is_on_disk(self, tmp_path: Path) -> None:
        write_skip_markers(tmp_path, {"kept.pdf": "h1", "gone.pdf": "h2"})
        write_skip_reasons(tmp_path, {"kept.pdf": "no text", "gone.pdf": "no text"})

        def _change(records: SkipRecords) -> None:
            del records.markers["gone.pdf"], records.reasons["gone.pdf"]
            records.markers["new.pdf"] = "h3"
            records.reasons["new.pdf"] = "decode failure"

        update_skip_records(tmp_path, _change)

        assert load_skip_markers(tmp_path) == {"kept.pdf": "h1", "new.pdf": "h3"}
        assert load_skip_reasons(tmp_path) == {"kept.pdf": "no text", "new.pdf": "decode failure"}

    def test_dropping_a_marker_drops_its_reason_and_kind(self, tmp_path: Path) -> None:
        write_skip_records(
            tmp_path,
            SkipRecords(
                markers={"scan.pdf": "h1", "keep.pdf": "h2"},
                reasons={"scan.pdf": "no text", "keep.pdf": "no text"},
                kinds={"scan.pdf": SkipKind.REMOVED, "keep.pdf": SkipKind.FAILED},
            ),
        )

        update_skip_records(tmp_path, lambda records: records.markers.pop("scan.pdf"))

        assert load_skip_markers(tmp_path) == {"keep.pdf": "h2"}
        assert load_skip_reasons(tmp_path) == {"keep.pdf": "no text"}
        assert load_skip_kinds(tmp_path) == {"keep.pdf": SkipKind.FAILED}
        assert "scan.pdf" not in (tmp_path / SKIP_KIND_FILENAME).read_text(encoding="utf-8")

    def test_an_unchanged_update_writes_nothing(self, tmp_path: Path) -> None:
        update_skip_records(tmp_path, lambda _records: None)
        assert not (tmp_path / SKIP_MARKER_FILENAME).exists()
        assert not (tmp_path / SKIP_REASON_FILENAME).exists()

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            pytest.param(
                lambda root: update_skip_records(
                    root, lambda records: records.markers.update({"new.pdf": "h2"})
                ),
                {"old.pdf": "h1", "new.pdf": "h2"},
                id="update",
            ),
            pytest.param(clear_skip_markers, {}, id="clear"),
            pytest.param(
                lambda root: mark_removed(root, {"new.pdf": "h2"}),
                {"old.pdf": "h1", "new.pdf": "h2"},
                id="mark_removed",
            ),
            pytest.param(clear_failed_markers, {}, id="clear_failed"),
        ],
    )
    def test_waits_for_another_holder_of_the_lock(self, tmp_path: Path, operation, expected):
        """A second process holding the records lock keeps this change out until it lets go."""
        import threading

        from filelock import FileLock

        write_skip_markers(tmp_path, {"old.pdf": "h1"})
        holder = FileLock(str(tmp_path / SKIP_MARKER_FILENAME) + ".lock")
        holder.acquire()
        worker = threading.Thread(target=operation, args=(tmp_path,))
        try:
            worker.start()
            worker.join(timeout=0.5)
            assert worker.is_alive()
            assert load_skip_markers(tmp_path) == {"old.pdf": "h1"}
        finally:
            holder.release()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert load_skip_markers(tmp_path) == expected

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(
                lambda root: update_skip_records(
                    root, lambda records: records.markers.update({"new.pdf": "h2"})
                ),
                id="update",
            ),
            pytest.param(clear_skip_markers, id="clear"),
            pytest.param(lambda root: mark_removed(root, {"new.pdf": "h2"}), id="mark_removed"),
            pytest.param(clear_failed_markers, id="clear_failed"),
        ],
    )
    def test_a_lock_that_stays_held_fails_the_change_without_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation
    ):
        """A change that cannot take the records lock raises and leaves the records as they were."""
        from lilbee.data.ingest.skip_marker import SkipRecordsLockError

        monkeypatch.setattr(skip_marker, "_RECORDS_LOCK_TIMEOUT_S", 0.05)
        _head_failure(tmp_path, "old.pdf", "h1", "no text")
        lock_path = str(tmp_path / SKIP_MARKER_FILENAME) + ".lock"
        holder = FileLock(lock_path)
        holder.acquire()
        try:
            with pytest.raises(SkipRecordsLockError, match=r"delete .*skipped_sources\.json\.lock"):
                operation(tmp_path)
        finally:
            holder.release()

        assert load_skip_markers(tmp_path) == {"old.pdf": "h1"}
        assert load_skip_kinds(tmp_path) == {"old.pdf": SkipKind.FAILED}


def test_kinds_round_trip_for_every_marker(tmp_path: Path) -> None:
    """A stored kind is read back for its marker."""
    write_skip_records(
        tmp_path,
        SkipRecords(
            markers={"scan.pdf": "h1", "gone.txt": "h2"},
            kinds={"scan.pdf": SkipKind.FAILED, "gone.txt": SkipKind.REMOVED},
        ),
    )
    assert load_skip_kinds(tmp_path) == {"scan.pdf": SkipKind.FAILED, "gone.txt": SkipKind.REMOVED}


def test_a_record_without_a_kind_is_failed_unless_its_reason_is_the_removal_text(
    tmp_path: Path,
) -> None:
    """Records written before the kinds sidecar existed read by their reason."""
    write_skip_markers(tmp_path, {"scan.pdf": "h1", "gone.txt": "h2", "bare.md": "h3"})
    write_skip_reasons(tmp_path, {"scan.pdf": "no text", "gone.txt": REMOVED_SKIP_REASON})
    assert not (tmp_path / SKIP_KIND_FILENAME).exists()
    assert load_skip_kinds(tmp_path) == {
        "scan.pdf": SkipKind.FAILED,
        "gone.txt": SkipKind.REMOVED,
        "bare.md": SkipKind.FAILED,
    }


@pytest.mark.parametrize(
    ("gone_entry", "scan_entry"),
    [
        pytest.param("failed", "removed", id="not-an-object"),
        pytest.param(
            {"kind": "archived", "hash": "h1", "reason": REMOVED_SKIP_REASON},
            {"kind": "archived", "hash": "h2", "reason": None},
            id="unknown-kind",
        ),
        pytest.param(
            {"kind": ["failed"], "hash": "h1", "reason": REMOVED_SKIP_REASON},
            {"kind": ["removed"], "hash": "h2", "reason": None},
            id="kind-not-a-string",
        ),
        pytest.param(
            {"kind": "failed", "reason": REMOVED_SKIP_REASON},
            {"kind": "removed", "reason": None},
            id="no-hash",
        ),
    ],
)
def test_an_unreadable_stored_kind_falls_back_to_the_reason(
    tmp_path: Path, gone_entry: object, scan_entry: object
) -> None:
    """A kinds entry this version cannot read is read like a missing one."""
    write_skip_markers(tmp_path, {"gone.txt": "h1", "scan.pdf": "h2"})
    write_skip_reasons(tmp_path, {"gone.txt": REMOVED_SKIP_REASON})
    stored = {"gone.txt": gone_entry, "scan.pdf": scan_entry}
    (tmp_path / SKIP_KIND_FILENAME).write_text(json.dumps(stored), encoding="utf-8")
    assert load_skip_kinds(tmp_path) == {"gone.txt": SkipKind.REMOVED, "scan.pdf": SkipKind.FAILED}


def test_a_kind_without_a_marker_is_not_reported(tmp_path: Path) -> None:
    """Kinds describe markers; a stray kind entry holds nothing out."""
    write_skip_records(
        tmp_path,
        SkipRecords(
            markers={"scan.pdf": "h1"},
            kinds={"scan.pdf": SkipKind.FAILED, "stray.md": SkipKind.REMOVED},
        ),
    )
    assert load_skip_kinds(tmp_path) == {"scan.pdf": SkipKind.FAILED}
    assert "stray.md" not in (tmp_path / SKIP_KIND_FILENAME).read_text(encoding="utf-8")


def test_held_out_names_lists_failures_only(tmp_path: Path) -> None:
    """A removed source is not a held-out file; a failure is."""
    write_skip_records(
        tmp_path,
        SkipRecords(
            markers={"b.pdf": "h1", "gone.txt": "h2", "a.pdf": "h3"},
            kinds={
                "b.pdf": SkipKind.FAILED,
                "gone.txt": SkipKind.REMOVED,
                "a.pdf": SkipKind.FAILED,
            },
        ),
    )
    assert held_out_names(tmp_path) == ["a.pdf", "b.pdf"]


def test_mark_removed_writes_marker_reason_and_kind(tmp_path: Path) -> None:
    """A removal records all three and leaves other records alone."""
    write_skip_markers(tmp_path, {"scan.pdf": "h1", "other.pdf": "h2"})
    write_skip_reasons(tmp_path, {"scan.pdf": "no text", "other.pdf": "no text"})

    mark_removed(tmp_path, {"scan.pdf": "h9", "gone.txt": "h3"})

    assert load_skip_markers(tmp_path) == {"scan.pdf": "h9", "gone.txt": "h3", "other.pdf": "h2"}
    assert load_skip_reasons(tmp_path) == {
        "scan.pdf": REMOVED_SKIP_REASON,
        "gone.txt": REMOVED_SKIP_REASON,
        "other.pdf": "no text",
    }
    assert load_skip_kinds(tmp_path) == {
        "scan.pdf": SkipKind.REMOVED,
        "gone.txt": SkipKind.REMOVED,
        "other.pdf": SkipKind.FAILED,
    }
    assert (tmp_path / SKIP_KIND_FILENAME).exists()


def test_clear_removes_the_kinds_sidecar(tmp_path: Path) -> None:
    """clear_skip_markers deletes the kinds file with the other two."""
    write_skip_records(
        tmp_path,
        SkipRecords(markers={"gone.txt": "h1"}, kinds={"gone.txt": SkipKind.REMOVED}),
    )
    clear_skip_markers(tmp_path)
    assert not (tmp_path / SKIP_KIND_FILENAME).exists()


def test_mark_removed_keeps_the_recorded_hash_of_an_unreachable_file(tmp_path: Path) -> None:
    """An unreachable marked file is held at its marker's hash; one with no marker is skipped."""
    write_skip_markers(tmp_path, {"away.pdf": "h1"})

    mark_removed(tmp_path, {"here.txt": "h2"}, unreachable=["away.pdf", "never.pdf"])

    assert load_skip_markers(tmp_path) == {"away.pdf": "h1", "here.txt": "h2"}
    assert load_skip_kinds(tmp_path) == {"away.pdf": SkipKind.REMOVED, "here.txt": SkipKind.REMOVED}


def _older_clear(data_root: Path) -> None:
    """What retry-skipped does in a lilbee without the kinds file: drop markers and reasons."""
    (data_root / SKIP_MARKER_FILENAME).unlink(missing_ok=True)
    (data_root / SKIP_REASON_FILENAME).unlink(missing_ok=True)


def _older_write(data_root: Path, markers: dict[str, str], reasons: dict[str, str]) -> None:
    """What a lilbee without the kinds file writes: the markers and reasons, nothing else."""
    write_skip_markers(data_root, {**load_skip_markers(data_root), **markers})
    write_skip_reasons(data_root, {**load_skip_reasons(data_root), **reasons})


def _head_failure(data_root: Path, name: str, fhash: str, reason: str) -> None:
    def _fail(records: SkipRecords) -> None:
        records.markers[name] = fhash
        records.reasons[name] = reason
        records.kinds[name] = SkipKind.FAILED

    update_skip_records(data_root, _fail)


class TestAnOlderLilbeeWritingTheRecords:
    """A lilbee without the kinds file clears and rewrites markers; a stale kind must not win."""

    @pytest.mark.parametrize(
        ("fhash", "reasons"),
        [
            pytest.param("h1", {"x.pdf": "no text extracted"}, id="same-hash"),
            pytest.param("h2", {"x.pdf": "no text extracted"}, id="new-hash"),
            pytest.param("h1", {}, id="same-hash-no-reason"),
        ],
    )
    def test_a_failure_after_an_older_retry_is_a_failure(
        self, tmp_path: Path, fhash: str, reasons: dict[str, str]
    ) -> None:
        mark_removed(tmp_path, {"x.pdf": "h1"})
        _older_clear(tmp_path)
        _older_write(tmp_path, {"x.pdf": fhash}, reasons)

        assert (tmp_path / SKIP_KIND_FILENAME).exists()
        assert load_skip_kinds(tmp_path) == {"x.pdf": SkipKind.FAILED}
        assert held_out_names(tmp_path) == ["x.pdf"]
        assert clear_failed_markers(tmp_path) == ["x.pdf"]
        assert load_skip_markers(tmp_path) == {}

    @pytest.mark.parametrize(
        ("failed_reason", "fhash"),
        [
            pytest.param("no text extracted", "h1", id="same-hash"),
            pytest.param(REMOVED_SKIP_REASON, "h2", id="new-hash-same-reason"),
        ],
    )
    def test_an_older_removal_of_a_former_failure_is_a_removal(
        self, tmp_path: Path, failed_reason: str, fhash: str
    ) -> None:
        _head_failure(tmp_path, "x.pdf", "h1", failed_reason)
        _older_clear(tmp_path)
        _older_write(tmp_path, {"x.pdf": fhash}, {"x.pdf": REMOVED_SKIP_REASON})

        assert load_skip_kinds(tmp_path) == {"x.pdf": SkipKind.REMOVED}
        assert held_out_names(tmp_path) == []
        assert clear_failed_markers(tmp_path) == []
        assert load_skip_markers(tmp_path) == {"x.pdf": fhash}

    def test_records_an_older_lilbee_did_not_touch_keep_their_kind(self, tmp_path: Path) -> None:
        """Control: the stored kind still wins over the reason while its marker is unchanged."""
        _head_failure(tmp_path, "x.pdf", "h1", REMOVED_SKIP_REASON)
        _older_write(tmp_path, {"y.pdf": "h2"}, {"y.pdf": "no text extracted"})

        assert load_skip_kinds(tmp_path) == {"x.pdf": SkipKind.FAILED, "y.pdf": SkipKind.FAILED}


_OPERATIONS = [
    pytest.param(
        lambda root: update_skip_records(
            root, lambda records: records.markers.update({"new.pdf": "h2"})
        ),
        id="update",
    ),
    pytest.param(clear_skip_markers, id="clear"),
    pytest.param(lambda root: mark_removed(root, {"new.pdf": "h2"}), id="mark_removed"),
    pytest.param(clear_failed_markers, id="clear_failed"),
]


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_every_sidecar_changes_while_the_records_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: Callable[[Path], object]
) -> None:
    """Another process asking for the records lock is refused during each sidecar write."""
    _head_failure(tmp_path, "old.pdf", "h1", "no text")
    lock_path = str(tmp_path / SKIP_MARKER_FILENAME) + ".lock"
    held_during: dict[str, bool] = {}

    def _lock_is_held() -> bool:
        refused: list[bool] = []

        def _probe() -> None:
            probe = FileLock(lock_path)
            try:
                probe.acquire(timeout=0)
            except Timeout:
                refused.append(True)
            else:
                probe.release()
                refused.append(False)

        prober = threading.Thread(target=_probe)
        prober.start()
        prober.join()
        return refused == [True]

    real_write, real_unlink = skip_marker._write_json_map, skip_marker._unlink

    def _write(path: Path, data: object) -> None:
        held_during[path.name] = _lock_is_held()
        real_write(path, data)

    def _unlink(path: Path) -> None:
        held_during[path.name] = _lock_is_held()
        real_unlink(path)

    monkeypatch.setattr(skip_marker, "_write_json_map", _write)
    monkeypatch.setattr(skip_marker, "_unlink", _unlink)
    assert not _lock_is_held()

    operation(tmp_path)

    assert held_during == dict.fromkeys(
        [SKIP_MARKER_FILENAME, SKIP_REASON_FILENAME, SKIP_KIND_FILENAME], True
    )
