"""A pin record that cannot be read leaves the request answered and the log line saying why."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.app.status import serving_roles
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers import serving
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.providers.serving import UNKNOWN_VERSION
from tests.conformance.fake_adapter import BundledFakeAdapter

GGUF_REF = "org/repo/file.gguf"


class _BrokenPin(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    failure: Exception | None = OSError("unreadable")

    def pin_record(self) -> Mapping[str, Any]:
        if self.failure is None:
            return super().pin_record()
        raise self.failure


@pytest.fixture(autouse=True)
def broken_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _BrokenPin.failure = OSError("unreadable")
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _BrokenPin())  # type: ignore[arg-type]
    monkeypatch.setattr(
        runnable,
        "session_host",
        lambda: collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None),
    )
    monkeypatch.setattr(cfg, "chat_model", GGUF_REF)
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")


def _provider(monkeypatch: pytest.MonkeyPatch) -> RoutingProvider:
    provider = RoutingProvider()
    local = MagicMock()
    local.chat.return_value = "Rome"
    monkeypatch.setattr(provider, "_get_local", lambda: local)
    return provider


def _served(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("served by")]


def _unreadable(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Cannot read the pin" in r.getMessage()]


@pytest.mark.parametrize(
    "failure",
    [OSError("disk gone"), KeyError("version"), RuntimeError("host unread")],
    ids=["os-error", "key-error", "runtime-error"],
)
def test_an_unreadable_pin_still_answers_and_logs_the_reason_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: Exception
) -> None:
    _BrokenPin.failure = failure
    provider = _provider(monkeypatch)
    with caplog.at_level(logging.INFO):
        assert provider.chat([], model=None) == "Rome"
        assert provider.chat([], model=None) == "Rome"
    lines = _served(caplog)
    assert len(lines) == 2
    unknown = f"engine=llama.cpp version={UNKNOWN_VERSION} build={UNKNOWN_VERSION} "
    assert all(unknown in line for line in lines)
    reasons = _unreadable(caplog)
    assert len(reasons) == 1
    assert "llama.cpp" in reasons[0]
    assert f"{type(failure).__name__}: {failure}" in reasons[0]


def test_the_status_reports_the_engine_with_an_unknown_version_when_the_pin_fails() -> None:
    _BrokenPin.failure = KeyError("version")
    rows = {row.role: row for row in serving_roles()}
    assert rows[WorkerRole.CHAT].engine == "llama.cpp"
    assert rows[WorkerRole.CHAT].version == UNKNOWN_VERSION


def test_an_unreadable_pin_reports_the_engine_with_an_unknown_version_build_and_reason() -> None:
    _BrokenPin.failure = KeyError("version")
    served = serving.served_by(GGUF_REF, _BrokenPin(), native=True)
    assert served == serving.ServingEngine(
        "llama.cpp", UNKNOWN_VERSION, UNKNOWN_VERSION, "KeyError: 'version'"
    )


def test_a_readable_pin_carries_no_reason() -> None:
    _BrokenPin.failure = None
    assert serving.served_by(GGUF_REF, _BrokenPin(), native=True).unread == ""


def test_a_pin_that_fails_recovers_and_fails_again_is_logged_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider = _provider(monkeypatch)
    with caplog.at_level(logging.WARNING):
        _BrokenPin.failure = OSError("same")
        provider.chat([], model=None)
        provider.chat([], model=None)
        _BrokenPin.failure = None
        provider.chat([], model=None)
        _BrokenPin.failure = OSError("same")
        provider.chat([], model=None)
    assert len(_unreadable(caplog)) == 2


def test_a_different_reason_is_logged_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider = _provider(monkeypatch)
    with caplog.at_level(logging.WARNING):
        _BrokenPin.failure = OSError("first")
        provider.chat([], model=None)
        _BrokenPin.failure = OSError("second")
        provider.chat([], model=None)
    assert len(_unreadable(caplog)) == 2


def test_an_error_outside_the_lookup_set_propagates() -> None:
    """A guard: the base also lets this error through; it fails if the catch widens."""
    _BrokenPin.failure = ZeroDivisionError("bug")
    with pytest.raises(ZeroDivisionError, match="bug"):
        serving.served_by(GGUF_REF, _BrokenPin(), native=True)
