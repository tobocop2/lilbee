"""Tests for the CLI cold-start status helpers (Starting ... / ready stderr lines)."""

from __future__ import annotations

from unittest import mock

from rich.console import Console

from lilbee.cli import helpers
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress


def _services_with_role_ready(ready: bool) -> mock.MagicMock:
    services = mock.MagicMock()
    services.provider.role_ready.return_value = ready
    return services


def test_announce_cold_start_prints_when_role_cold(monkeypatch) -> None:
    monkeypatch.setattr(helpers.cfg, "json_mode", False)
    monkeypatch.setattr(
        "lilbee.app.services.get_services", lambda: _services_with_role_ready(False)
    )
    err = helpers.announce_cold_start(WorkerRole.CHAT, "org/repo/chat.gguf")
    assert isinstance(err, Console)
    assert err.stderr is True


def test_announce_cold_start_silent_when_role_warm(monkeypatch) -> None:
    monkeypatch.setattr(helpers.cfg, "json_mode", False)
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: _services_with_role_ready(True))
    assert helpers.announce_cold_start(WorkerRole.EMBED, "org/repo/embed.gguf") is None


def test_announce_cold_start_silent_in_json_mode(monkeypatch) -> None:
    monkeypatch.setattr(helpers.cfg, "json_mode", True)
    # role_ready must not even be consulted in JSON mode (no chatter on a parseable stream).
    services = _services_with_role_ready(False)
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: services)
    assert helpers.announce_cold_start(WorkerRole.CHAT, "m") is None
    services.provider.role_ready.assert_not_called()


def _services_with_warm(snapshot: object) -> mock.MagicMock:
    services = mock.MagicMock()
    services.provider.warm_progress.return_value = snapshot
    return services


def test_announce_ready_prints_through_returned_console(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: _services_with_warm(None))
    err = mock.MagicMock()
    helpers.announce_ready(err, WorkerRole.CHAT)
    assert err.print.call_count == 1
    assert "ready" in err.print.call_args.args[0].lower()


def test_announce_ready_noop_when_no_console() -> None:
    # When cold-start was never announced (role warm), announce_ready is a no-op.
    helpers.announce_ready(None, WorkerRole.CHAT)  # must not raise


def test_announce_ready_reports_the_engines_reason_when_the_model_failed_to_load(
    monkeypatch,
) -> None:
    # A chat model whose llama-server died on load must not be announced as ready. In
    # RAG mode a grounded refusal still streams, so a token is not evidence of a load.
    snapshot = WarmProgress(phase=WarmPhase.ERROR, error="unknown model architecture: qwen35moe")
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: _services_with_warm(snapshot))
    err = mock.MagicMock()
    helpers.announce_ready(err, WorkerRole.CHAT)
    printed = " ".join(str(call.args[0]) for call in err.print.call_args_list)
    assert "ready" not in printed.lower()
    assert "unknown model architecture: qwen35moe" in printed


def test_announce_ready_is_not_fooled_by_a_transient_not_running_probe(monkeypatch) -> None:
    # llama-swap can report a freshly loaded model as not running. Readiness must not be
    # re-probed here, or a healthy engine prints a spurious failure line.
    services = _services_with_warm(WarmProgress(phase=WarmPhase.READY))
    services.provider.role_ready.return_value = False
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: services)
    err = mock.MagicMock()
    helpers.announce_ready(err, WorkerRole.CHAT)
    assert "ready" in err.print.call_args.args[0].lower()
    services.provider.role_ready.assert_not_called()


def test_announce_ready_for_a_non_chat_role_ignores_the_chat_warm_tracker(monkeypatch) -> None:
    # The warm tracker only tracks chat; embed must not inherit a chat load failure.
    snapshot = WarmProgress(phase=WarmPhase.ERROR, error="chat blew up")
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: _services_with_warm(snapshot))
    err = mock.MagicMock()
    helpers.announce_ready(err, WorkerRole.EMBED)
    assert "ready" in err.print.call_args.args[0].lower()
