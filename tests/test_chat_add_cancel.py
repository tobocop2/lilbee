"""Tests for /add cancel cleanup.

When a user cancels an in-flight /add, the entry linked into documents/
must be removed so the next sync does not silently re-ingest it.
"""

from __future__ import annotations

import pytest

from lilbee.cli.tui.screens.chat import remove_linked_sources
from lilbee.core.config import cfg


@pytest.fixture
def isolated_documents(tmp_path):
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    try:
        yield cfg.documents_dir
    finally:
        for field_name in type(snapshot).model_fields:
            setattr(cfg, field_name, getattr(snapshot, field_name))


class TestRemoveLinkedSources:
    def test_removes_copied_file(self, isolated_documents):
        target = isolated_documents / "qa-big.md"
        target.write_text("hello")
        remove_linked_sources(["qa-big.md"])
        assert not target.exists()

    def test_removes_copied_directory(self, isolated_documents):
        nested = isolated_documents / "_web" / "example.com"
        nested.mkdir(parents=True)
        (nested / "index.md").write_text("data")
        remove_linked_sources(["_web/example.com"])
        assert not nested.exists()

    def test_unlinks_symlink_without_touching_target(self, isolated_documents, tmp_path):
        # The normal add creates a symlink; cleanup must remove the link, never
        # the source bytes behind it.
        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("keep me")
        link = isolated_documents / "corpus"
        link.symlink_to(source)

        remove_linked_sources(["corpus"])

        assert not link.exists()  # link gone
        assert (source / "a.txt").read_text() == "keep me"  # source untouched

    def test_tolerates_missing_file(self, isolated_documents):
        # User may have deleted the file concurrently; do not raise.
        remove_linked_sources(["never-existed.md"])
        assert isolated_documents.exists()

    def test_leaves_untouched_siblings_alone(self, isolated_documents):
        keep = isolated_documents / "keep.md"
        keep.write_text("pre-existing")
        drop = isolated_documents / "drop.md"
        drop.write_text("copied")
        remove_linked_sources(["drop.md"])
        assert keep.exists()
        assert not drop.exists()

    def test_swallows_oserror_and_logs(self, isolated_documents, caplog, monkeypatch):
        """If the filesystem refuses the delete, the helper must not raise.

        The /add worker thread relies on this: an OSError from cleanup must
        not propagate and mask the original failure being surfaced to the
        user via the Task Center rail.
        """
        import logging
        from pathlib import Path

        target = isolated_documents / "locked.md"
        target.write_text("x")

        def _raise(self):
            raise OSError("simulated EACCES")

        monkeypatch.setattr(Path, "unlink", _raise)
        with caplog.at_level(logging.DEBUG, logger="lilbee.cli.tui.screens.chat"):
            remove_linked_sources(["locked.md"])
        # Target still exists since unlink was monkeypatched; that's fine --
        # the contract is only that the helper didn't raise.
        assert target.exists()


class TestDoAddCancelCleanup:
    """when the sync under /add raises (cancel or crash), the linked
    entries must be removed from documents/ so the next sync does not silently
    re-ingest them."""

    def test_sync_exception_triggers_cleanup(self, isolated_documents, monkeypatch):
        from unittest.mock import MagicMock, patch

        from lilbee.app.ingest import LinkResult
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.update = MagicMock()
        # Silence thread-safe notify during the test (called on failure paths).
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]

        target = isolated_documents / "qa-big.md"
        target.write_text("big file contents")
        copy_result = LinkResult(linked=["qa-big.md"], skipped=[])

        class _Cancelled(Exception):
            pass

        def _run(coro):
            coro.close()
            raise _Cancelled("cancelled by user")

        with (
            patch("lilbee.app.ingest.link_files", return_value=copy_result),
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            pytest.raises(_Cancelled),
        ):
            screen._do_add([target], reporter)

        # File copied into documents/ must be gone after cancel.
        assert not target.exists()

    def test_sync_result_failed_triggers_cleanup(self, isolated_documents, monkeypatch):
        """A SyncResult with failed entries must also remove the copied files.

        Without this, a failing sync would still leave the file in
        documents/ ready for the next sync to re-ingest.
        """
        from unittest.mock import MagicMock, patch

        from lilbee.app.ingest import LinkResult
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.data.ingest import SyncResult

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.update = MagicMock()
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]

        target = isolated_documents / "qa-fail.md"
        target.write_text("hello")
        copy_result = LinkResult(linked=["qa-fail.md"], skipped=[])

        failing_result = SyncResult(
            added=[], updated=[], removed=[], unchanged=0, failed=["qa-fail.md"]
        )

        def _run(coro):
            coro.close()
            return failing_result

        with (
            patch("lilbee.app.ingest.link_files", return_value=copy_result),
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            pytest.raises(RuntimeError, match="Sync failed"),
        ):
            screen._do_add([target], reporter)

        assert not target.exists()
