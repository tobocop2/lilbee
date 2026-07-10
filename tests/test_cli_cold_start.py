"""Tests for the CLI cold-start status helpers (Starting ... / ready stderr lines)."""

from __future__ import annotations

from unittest import mock

from rich.console import Console

from lilbee.cli import helpers
from lilbee.providers.roles import WorkerRole


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


def test_announce_ready_prints_through_returned_console(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.app.services.get_services", lambda: _services_with_role_ready(True))
    err = mock.MagicMock()
    helpers.announce_ready(err, WorkerRole.CHAT)
    assert err.print.call_count == 1
    assert "ready" in err.print.call_args.args[0].lower()


def test_announce_ready_noop_when_no_console() -> None:
    # When cold-start was never announced (role warm), announce_ready is a no-op.
    helpers.announce_ready(None, WorkerRole.CHAT)  # must not raise


def test_announce_ready_does_not_claim_ready_when_the_model_never_loaded(monkeypatch) -> None:
    # A chat model whose llama-server died on load leaves the role not ready. The RAG
    # path can still stream a grounded refusal, so a token arriving is not evidence the
    # engine came up: announcing "ready" there is how a load failure got masked.
    monkeypatch.setattr(
        "lilbee.app.services.get_services", lambda: _services_with_role_ready(False)
    )
    err = mock.MagicMock()
    helpers.announce_ready(err, WorkerRole.CHAT)
    printed = " ".join(str(call.args[0]) for call in err.print.call_args_list).lower()
    assert "ready" not in printed
    assert "did not load" in printed
