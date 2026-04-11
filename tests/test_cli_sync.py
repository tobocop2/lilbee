"""Tests for lilbee.cli.sync — background sync, executor, and status."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

from lilbee.cli import sync as sync_mod
from lilbee.cli import theme
from lilbee.progress import (
    EmbedEvent,
    EventType,
    ExtractEvent,
    FileStartEvent,
    SyncDoneEvent,
)


class TestFormatSyncSummary:
    def test_all_zeros_returns_none(self):
        assert sync_mod._format_sync_summary(0, 0, 0, 0) is None

    def test_added_only(self):
        assert sync_mod._format_sync_summary(3, 0, 0, 0) == "3 added"

    def test_updated_only(self):
        assert sync_mod._format_sync_summary(0, 2, 0, 0) == "2 updated"

    def test_removed_only(self):
        assert sync_mod._format_sync_summary(0, 0, 1, 0) == "1 removed"

    def test_failed_only(self):
        assert sync_mod._format_sync_summary(0, 0, 0, 5) == "5 failed"

    def test_multiple_counts(self):
        result = sync_mod._format_sync_summary(1, 2, 3, 4)
        assert result == "1 added, 2 updated, 3 removed, 4 failed"

    def test_partial_counts(self):
        result = sync_mod._format_sync_summary(1, 0, 2, 0)
        assert result == "1 added, 2 removed"


class TestSyncProgressPrinter:
    def test_file_start_event(self):
        con = MagicMock()
        callback = sync_mod._sync_progress_printer(con)
        data = FileStartEvent(file="readme.md", total_files=5, current_file=2)

        callback(EventType.FILE_START, data)

        con.print.assert_called_once()
        printed = con.print.call_args[0][0]
        assert "readme.md" in printed
        assert "2/5" in printed

    def test_done_event_with_changes(self):
        con = MagicMock()
        callback = sync_mod._sync_progress_printer(con)
        data = SyncDoneEvent(added=1, updated=0, removed=0, failed=0)

        callback(EventType.DONE, data)

        con.print.assert_called_once()
        printed = con.print.call_args[0][0]
        assert "1 added" in printed

    def test_done_event_no_changes(self):
        con = MagicMock()
        callback = sync_mod._sync_progress_printer(con)
        data = SyncDoneEvent(added=0, updated=0, removed=0, failed=0)

        callback(EventType.DONE, data)

        con.print.assert_not_called()

    def test_unhandled_event_type_ignored(self):
        con = MagicMock()
        callback = sync_mod._sync_progress_printer(con)

        callback(EventType.EMBED, EmbedEvent(file="x", chunk=1, total_chunks=1))

        con.print.assert_not_called()


class TestGetExecutorAndShutdown:
    def test_get_executor_creates_pool(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_bg_executor", None)
        executor = sync_mod._get_executor()
        assert isinstance(executor, ThreadPoolExecutor)
        # Clean up
        executor.shutdown(wait=False)
        monkeypatch.setattr(sync_mod, "_bg_executor", None)

    def test_get_executor_returns_same_instance(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_bg_executor", None)
        first = sync_mod._get_executor()
        second = sync_mod._get_executor()
        assert first is second
        first.shutdown(wait=False)
        monkeypatch.setattr(sync_mod, "_bg_executor", None)

    def test_shutdown_executor_when_none(self, monkeypatch):
        monkeypatch.setattr(sync_mod, "_bg_executor", None)
        sync_mod.shutdown_executor()
        assert sync_mod._bg_executor is None

    def test_shutdown_executor_cleans_up(self, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        sync_mod.shutdown_executor()

        mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        assert sync_mod._bg_executor is None


class TestOnSyncDone:
    def test_no_exception_returns_early(self):
        con = MagicMock()
        future = MagicMock(spec=Future)
        future.exception.return_value = None

        sync_mod._on_sync_done(con, future)

        con.print.assert_not_called()

    def test_cancelled_error_ignored(self):
        con = MagicMock()
        future = MagicMock(spec=Future)
        future.exception.return_value = asyncio.CancelledError()

        sync_mod._on_sync_done(con, future)

        con.print.assert_not_called()

    def test_runtime_error_cannot_schedule_ignored(self):
        con = MagicMock()
        future = MagicMock(spec=Future)
        future.exception.return_value = RuntimeError("cannot schedule new futures after shutdown")

        sync_mod._on_sync_done(con, future)

        con.print.assert_not_called()

    def test_other_error_non_chat_mode(self):
        con = MagicMock()
        future = MagicMock(spec=Future)
        future.exception.return_value = ValueError("something broke")

        sync_mod._on_sync_done(con, future, chat_mode=False)

        con.print.assert_called_once()
        printed = con.print.call_args[0][0]
        assert theme.ERROR in printed
        assert "something broke" in printed

    def test_other_error_chat_mode(self, capsys):
        con = MagicMock()
        future = MagicMock(spec=Future)
        future.exception.return_value = ValueError("sync failed")

        sync_mod._on_sync_done(con, future, chat_mode=True)

        captured = capsys.readouterr()
        assert "sync failed" in captured.out
        con.print.assert_not_called()


class TestSyncStatus:
    def test_init_defaults(self):
        status = sync_mod.SyncStatus()
        assert status.text == ""
        assert status.pending == 0

    def test_clear(self):
        status = sync_mod.SyncStatus()
        status.text = "something"
        status.clear()
        assert status.text == ""


class TestChatSyncCallback:
    def test_file_start_event(self):
        status = sync_mod.SyncStatus()
        callback = sync_mod._chat_sync_callback(status)
        data = FileStartEvent(file="doc.txt", total_files=3, current_file=1)

        callback(EventType.FILE_START, data)

        assert "doc.txt" in status.text
        assert "1/3" in status.text

    def test_file_start_with_pending(self):
        status = sync_mod.SyncStatus()
        callback = sync_mod._chat_sync_callback(status)
        status.pending = 2
        data = FileStartEvent(file="doc.txt", total_files=3, current_file=1)

        callback(EventType.FILE_START, data)

        assert "+2 queued" in status.text

    def test_extract_event(self):
        status = sync_mod.SyncStatus()
        callback = sync_mod._chat_sync_callback(status)
        data = ExtractEvent(file="scan.pdf", page=2, total_pages=10)

        callback(EventType.EXTRACT, data)

        assert "Vision OCR" in status.text
        assert "2/10" in status.text
        assert "scan.pdf" in status.text

    def test_extract_event_with_pending(self):
        status = sync_mod.SyncStatus()
        callback = sync_mod._chat_sync_callback(status)
        status.pending = 1
        data = ExtractEvent(file="scan.pdf", page=1, total_pages=5)

        callback(EventType.EXTRACT, data)

        assert "+1 queued" in status.text

    def test_done_event_with_changes(self, capsys):
        status = sync_mod.SyncStatus()
        status.text = "syncing..."
        callback = sync_mod._chat_sync_callback(status)
        data = SyncDoneEvent(added=2, updated=1, removed=0, failed=0)

        callback(EventType.DONE, data)

        assert status.text == ""
        captured = capsys.readouterr()
        assert "2 added" in captured.out
        assert "1 updated" in captured.out

    def test_done_event_no_changes(self, capsys):
        status = sync_mod.SyncStatus()
        status.text = "syncing..."
        callback = sync_mod._chat_sync_callback(status)
        data = SyncDoneEvent(added=0, updated=0, removed=0, failed=0)

        callback(EventType.DONE, data)

        assert status.text == ""
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_clears_status_on_init(self):
        status = sync_mod.SyncStatus()
        status.text = "leftover"
        sync_mod._chat_sync_callback(status)
        assert status.text == ""

    def test_unhandled_event_ignored(self):
        status = sync_mod.SyncStatus()
        callback = sync_mod._chat_sync_callback(status)

        callback(EventType.EMBED, EmbedEvent(file="x", chunk=1, total_chunks=1))

        assert status.text == ""


class TestRunSyncBackground:
    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_non_chat_mode(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        result = sync_mod.run_sync_background(con)

        assert result is mock_future
        mock_executor.submit.assert_called_once()
        mock_future.add_done_callback.assert_called_once()

    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_chat_mode_increments_pending(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        status = sync_mod.SyncStatus()
        sync_mod.run_sync_background(con, chat_mode=True, sync_status=status)

        assert status.pending == 1

    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_submitted_callable_runs_sync(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        sync_mod.run_sync_background(con, force_vision=True)

        submitted_fn = mock_executor.submit.call_args[0][0]
        submitted_fn()

        mock_asyncio_run.assert_called_once()
        # Verify sync was called with correct kwargs (on_progress is the callback)
        mock_sync.assert_called_once()
        call_kwargs = mock_sync.call_args[1]
        assert call_kwargs["quiet"] is True
        assert call_kwargs["force_vision"] is True
        assert callable(call_kwargs["on_progress"])

    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_chat_mode_decrements_pending_on_run(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        status = sync_mod.SyncStatus()
        sync_mod.run_sync_background(con, chat_mode=True, sync_status=status)

        assert status.pending == 1
        submitted_fn = mock_executor.submit.call_args[0][0]
        submitted_fn()
        assert status.pending == 0

    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_non_chat_mode_no_pending_change(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        status = sync_mod.SyncStatus()
        sync_mod.run_sync_background(con, chat_mode=False, sync_status=status)

        # Non-chat mode doesn't touch pending
        assert status.pending == 0
        submitted_fn = mock_executor.submit.call_args[0][0]
        submitted_fn()
        assert status.pending == 0

    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_default_sync_status_created(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        result = sync_mod.run_sync_background(con)

        assert result is mock_future

    @patch("lilbee.cli.sync.asyncio.run", side_effect=lambda coro: coro.close())
    @patch("lilbee.cli.sync.sync", new_callable=AsyncMock)
    def test_done_callback_wired(self, mock_sync, mock_asyncio_run, monkeypatch):
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_future = MagicMock(spec=Future)
        mock_executor.submit.return_value = mock_future
        monkeypatch.setattr(sync_mod, "_bg_executor", mock_executor)

        con = MagicMock()
        sync_mod.run_sync_background(con, chat_mode=True)

        mock_future.add_done_callback.assert_called_once()
        done_cb = mock_future.add_done_callback.call_args[0][0]

        # Verify the callback calls _on_sync_done with chat_mode=True
        with patch.object(sync_mod, "_on_sync_done") as mock_osd:
            done_cb(mock_future)
            mock_osd.assert_called_once_with(con, mock_future, chat_mode=True)
