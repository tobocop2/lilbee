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
