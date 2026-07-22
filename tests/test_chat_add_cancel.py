"""Tests for /add cancel cleanup.

When a user cancels an in-flight /add, the source root it registered must be
un-registered so the next sync does not silently re-ingest it. The source bytes
on disk are never touched.
"""

from __future__ import annotations

import pytest

from lilbee.cli.tui.screens.chat_helpers import unregister_added_roots
from lilbee.core.config import cfg


@pytest.fixture
def isolated_documents(tmp_path):
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_root = tmp_path
    cfg.linked_roots = {}
    try:
        yield cfg.documents_dir
    finally:
        for field_name in type(snapshot).model_fields:
            setattr(cfg, field_name, getattr(snapshot, field_name))


class TestUnregisterAddedRoots:
    def test_unregisters_root_without_touching_source(self, isolated_documents, tmp_path):
        from lilbee.core import settings

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("keep me")
        settings.set_value(cfg.data_root, "linked_roots", {"corpus": str(source)})

        unregister_added_roots(["corpus"])

        assert "corpus" not in cfg.linked_roots  # registry entry dropped
        assert (source / "a.txt").read_text() == "keep me"  # source bytes untouched

    def test_tolerates_unknown_label(self, isolated_documents):
        # User may have removed the source concurrently; do not raise.
        unregister_added_roots(["never-registered"])
        assert cfg.linked_roots == {}

    def test_leaves_other_roots_alone(self, isolated_documents, tmp_path):
        from lilbee.core import settings

        settings.set_value(
            cfg.data_root,
            "linked_roots",
            {"keep": str(tmp_path / "keep"), "drop": str(tmp_path / "drop")},
        )
        unregister_added_roots(["drop"])
        assert "keep" in cfg.linked_roots
        assert "drop" not in cfg.linked_roots

    def test_empty_list_is_a_noop(self, isolated_documents, tmp_path):
        cfg.linked_roots = {"keep": str(tmp_path / "keep")}
        unregister_added_roots([])
        assert "keep" in cfg.linked_roots


class TestDoAddCancelCleanup:
    """When the sync under /add raises (cancel or crash), the root it registered
    must be un-registered so the next sync does not silently re-ingest it."""

    def test_sync_exception_triggers_cleanup(self, isolated_documents, tmp_path):
        from unittest.mock import MagicMock, patch

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.update = MagicMock()
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("big file contents")

        class _Cancelled(Exception):
            pass

        def _run(coro):
            coro.close()
            raise _Cancelled("cancelled by user")

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            pytest.raises(_Cancelled),
        ):
            screen._do_add([source], reporter)

        # The root registered by this /add must be gone after cancel.
        assert "corpus" not in cfg.linked_roots
        assert (source / "a.txt").exists()  # source bytes never touched

    def test_sync_result_failed_triggers_cleanup(self, isolated_documents, tmp_path):
        """A SyncResult with failed entries must also un-register the root.

        Without this, a failing sync would leave the root registered, ready for
        the next sync to re-ingest.
        """
        from unittest.mock import MagicMock, patch

        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.data.ingest import SyncResult

        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock()
        reporter.update = MagicMock()
        screen.notify = lambda *a, **kw: None  # type: ignore[assignment]

        source = tmp_path / "corpus"
        source.mkdir()
        (source / "a.txt").write_text("hello")

        failing_result = SyncResult(
            added=[], updated=[], removed=[], unchanged=0, failed=["corpus/a.txt"]
        )

        def _run(coro):
            coro.close()
            return failing_result

        with (
            patch("lilbee.runtime.asyncio_loop.run", side_effect=_run),
            pytest.raises(RuntimeError, match="Sync failed"),
        ):
            screen._do_add([source], reporter)

        assert "corpus" not in cfg.linked_roots
        assert (source / "a.txt").exists()
