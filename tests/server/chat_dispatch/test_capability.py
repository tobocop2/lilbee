"""Tests for the tool-capability check that gates tool-bearing requests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.app.services import set_services
from lilbee.providers.base import ProviderError
from lilbee.server.chat_dispatch.capability import model_supports_tools

pytestmark = pytest.mark.usefixtures("_isolated_services")


@pytest.fixture
def _isolated_services(monkeypatch):
    """Install a fresh mock services container with a swappable provider."""
    from tests.conftest import make_mock_services

    provider = MagicMock()
    services = make_mock_services(provider=provider)
    set_services(services)
    yield provider
    set_services(None)


def test_model_supports_tools_delegates_to_provider(_isolated_services) -> None:
    _isolated_services.supports_tools.return_value = True
    assert model_supports_tools("bartowski/Qwen3-0.6B-GGUF::Q4_K_M") is True
    _isolated_services.supports_tools.assert_called_once_with("bartowski/Qwen3-0.6B-GGUF::Q4_K_M")


def test_model_supports_tools_returns_false_when_provider_says_no(
    _isolated_services,
) -> None:
    _isolated_services.supports_tools.return_value = False
    assert model_supports_tools("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF::Q8_0") is False


def test_model_supports_tools_returns_false_on_provider_error(
    _isolated_services,
) -> None:
    # ProviderError from the probe (model file unavailable, backend down) is
    # the safe-default-False path so callers see a clean 400, not a 500.
    # Other exceptions are intentionally NOT caught here; see capability.py.
    _isolated_services.supports_tools.side_effect = ProviderError(
        "backend down", provider="llama-cpp"
    )
    assert model_supports_tools("nonexistent/model::Q4") is False


def test_model_supports_tools_propagates_unexpected_exceptions(
    _isolated_services,
) -> None:
    # A non-ProviderError bug in the provider must surface as a 500, not be
    # silently downgraded to "no tools". Documents the intentional narrowing
    # of the except clause in capability.py.
    _isolated_services.supports_tools.side_effect = RuntimeError("genuine bug")
    with pytest.raises(RuntimeError, match="genuine bug"):
        model_supports_tools("nonexistent/model::Q4")


class _HeaderEngine:
    """An engine stand-in whose header read returns a fixed answer."""

    def __init__(self, meta: dict[str, str] | None) -> None:
        self.meta = meta

    def model_meta(self, path):
        return self.meta


def _tool_request():
    from lilbee.server.chat_dispatch.canonical import CanonicalChatRequest, CanonicalTool

    return CanonicalChatRequest(
        model="org/repo",
        messages=[],
        tools=[CanonicalTool(name="lookup", description="d", input_schema={"type": "object"})],
    )


def _serve_with_engine(monkeypatch, tmp_path, meta):
    """Route the tool check to the real fleet probe over a chat engine that reads *meta*."""
    from lilbee.providers.fleet import provider as fleet

    fleet._supports_tools_cached.cache_clear()
    set_services(None)
    from tests.conftest import make_mock_services

    set_services(make_mock_services(provider=fleet.FleetProvider()))
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _m: tmp_path)
    monkeypatch.setattr(fleet.FleetProvider, "_adopt_engine", lambda _s, _r: _HeaderEngine(meta))


def test_tool_request_reaches_an_engine_that_reads_no_header(monkeypatch, tmp_path) -> None:
    from lilbee.server.chat_dispatch.dispatch import _ensure_tool_capability

    _serve_with_engine(monkeypatch, tmp_path, None)
    _ensure_tool_capability(_tool_request(), "org/repo#safetensors")


def test_tool_request_for_a_gguf_without_a_tool_template_gets_the_400(
    monkeypatch, tmp_path
) -> None:
    from lilbee.server.chat_dispatch.dispatch import (
        ModelDoesNotSupportToolsError,
        _ensure_tool_capability,
    )

    _serve_with_engine(monkeypatch, tmp_path, {"chat_template": "{{ messages }}"})
    with pytest.raises(ModelDoesNotSupportToolsError, match=r"org/repo/plain\.gguf"):
        _ensure_tool_capability(_tool_request(), "org/repo/plain.gguf")
