"""A path or model ref in a user-facing message appears exactly as the user typed it.

``repr`` doubles the backslashes of a Windows path, so a reader who copies the
path out of the message gets one that does not exist. The refs here are literal
Windows paths, so each arm fails on every platform, not only on a Windows runner.
"""

from __future__ import annotations

import logging

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from lilbee.retrieval.embedder import Embedder

WINDOWS_REF = "C:\\Users\\me\\models\\chat.gguf"
DOUBLED = "C:\\\\Users"


def _assert_verbatim(text: str) -> None:
    assert f"'{WINDOWS_REF}'" in text
    assert DOUBLED not in text


def _plan_with(monkeypatch: pytest.MonkeyPatch, error: ProviderError) -> None:
    def _estimate(role, ref, **_k):
        raise error

    monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
    monkeypatch.setattr(cfg, "chat_model", WINDOWS_REF)
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")


def test_not_installed_warning_shows_the_path_as_typed(monkeypatch, caplog) -> None:
    _plan_with(
        monkeypatch,
        ProviderError("gone", provider="llama-server", kind=ProviderErrorKind.NOT_FOUND),
    )
    with caplog.at_level(logging.WARNING):
        planning_mod._server_model_inputs()
    assert "is not installed" in caplog.text
    _assert_verbatim(caplog.text)


def test_could_not_size_warning_shows_the_path_as_typed(monkeypatch, caplog) -> None:
    _plan_with(
        monkeypatch,
        ProviderError("bad", provider="llama-server", kind=ProviderErrorKind.SERVER),
    )
    with caplog.at_level(logging.WARNING):
        planning_mod._server_model_inputs()
    assert "could not size model" in caplog.text
    _assert_verbatim(caplog.text)


def test_embedder_warning_shows_the_path_as_typed(monkeypatch, caplog) -> None:
    from unittest.mock import MagicMock

    provider = MagicMock()
    provider.list_models.return_value = []
    monkeypatch.setattr(cfg, "embedding_model", WINDOWS_REF)
    with caplog.at_level(logging.WARNING, logger="lilbee.retrieval.embedder"):
        Embedder(cfg, provider).validate_model()
    assert "is not available" in caplog.text
    _assert_verbatim(caplog.text)


def test_estimator_failure_shows_the_path_as_typed(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise OSError("no estimator")

    monkeypatch.setattr(vram_mod, "run_bounded", _boom)
    with pytest.raises(ProviderError) as excinfo:
        vram_mod._run_parser(["gguf-parser"], WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))


def test_unexpected_estimator_output_shows_the_path_as_typed() -> None:
    with pytest.raises(ProviderError) as excinfo:
        vram_mod._parse_estimate("{}", WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))


def test_pull_refusal_shows_the_ref_as_typed() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        FleetProvider().pull_model(WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))


def test_prompt_budget_error_shows_the_ref_as_typed() -> None:
    from tests.test_fleet_provider import _fake_client, _provider_with_clients

    provider = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    provider._chat_ctx = 64
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 9000}]
    with pytest.raises(ProviderError) as excinfo:
        provider._fit_chat_context(msgs, None, None, WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))
