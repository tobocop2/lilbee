"""Tests for Ctrl+C cancellation of the CLI ingest path (lilbee add / sync)."""

from __future__ import annotations

import signal
import threading
from unittest import mock

import pytest

from lilbee.cli.commands import ingest_sync
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.progress import EventType, FileStartEvent


def test_cancellable_progress_raises_when_event_set() -> None:
    cancel = threading.Event()
    seen: list[object] = []
    cb = ingest_sync._cancellable_progress(cancel, lambda et, d: seen.append(et))
    event = FileStartEvent(file="x", total_files=1, current_file=1)
    cb(EventType.FILE_START, event)  # not cancelled: forwards
    assert seen == [EventType.FILE_START]
    cancel.set()
    with pytest.raises(TaskCancelledError):
        cb(EventType.FILE_START, event)


def test_cancellable_progress_forwards_to_chain_until_cancelled() -> None:
    cancel = threading.Event()
    seen: list[tuple] = []
    cb = ingest_sync._cancellable_progress(cancel, lambda et, d: seen.append((et, d)))
    e1 = FileStartEvent(file="a", total_files=2, current_file=1)
    e2 = FileStartEvent(file="b", total_files=2, current_file=2)
    cb(EventType.FILE_START, e1)
    cb(EventType.FILE_START, e2)
    assert seen == [(EventType.FILE_START, e1), (EventType.FILE_START, e2)]


def test_run_sync_with_signal_cancel_installs_and_restores_sigint(monkeypatch) -> None:
    async def _fake_sync(**kwargs):
        # The SIGINT handler must be installed while the sync runs.
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not original
        return "sync-result"

    original = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(ingest_sync, "cfg", mock.MagicMock(json_mode=False))
    monkeypatch.setattr("lilbee.data.ingest.sync", _fake_sync)

    result = ingest_sync._run_sync_with_signal_cancel()
    assert result == "sync-result"
    # The previous handler is restored after the run.
    assert signal.getsignal(signal.SIGINT) is original


def test_run_sync_with_signal_cancel_passes_cancel_event(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_sync(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(ingest_sync, "cfg", mock.MagicMock(json_mode=True))
    monkeypatch.setattr("lilbee.data.ingest.sync", _fake_sync)

    ingest_sync._run_sync_with_signal_cancel(force_rebuild=True, retry_skipped=True)
    assert isinstance(captured["cancel"], threading.Event)
    assert captured["force_rebuild"] is True
    assert captured["retry_skipped"] is True
    assert captured["quiet"] is True  # json_mode -> quiet bar


def test_rebuild_rejects_non_sync_result(monkeypatch) -> None:
    # The signal-cancel runner is typed to return ``object``; rebuild narrows it
    # back to a SyncResult and rejects anything else rather than reading .added
    # off an unexpected type.
    monkeypatch.setattr(ingest_sync, "apply_overrides", lambda **_k: None)
    monkeypatch.setattr(ingest_sync, "cfg", mock.MagicMock(json_mode=False))
    monkeypatch.setattr(ingest_sync, "_run_sync_with_signal_cancel", lambda **_k: "not-a-result")
    with pytest.raises(TypeError, match="Expected SyncResult"):
        ingest_sync.rebuild()


def test_sigint_during_sync_sets_cancel_event(monkeypatch) -> None:
    # The installed handler, when fired, must set the cancel event the sync polls.
    seen_cancel: dict[str, threading.Event] = {}

    async def _fake_sync(**kwargs):
        cancel = kwargs["cancel"]
        seen_cancel["event"] = cancel
        # Simulate a SIGINT arriving mid-sync by invoking the current handler.
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert cancel.is_set()
        return "cancelled-clean"

    monkeypatch.setattr(ingest_sync, "cfg", mock.MagicMock(json_mode=True))
    monkeypatch.setattr("lilbee.data.ingest.sync", _fake_sync)

    result = ingest_sync._run_sync_with_signal_cancel()
    assert result == "cancelled-clean"
    assert seen_cancel["event"].is_set()
