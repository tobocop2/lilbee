"""A path or model ref in a user-facing message appears exactly as the user typed it.

``repr`` doubles the backslashes of a Windows path, so a reader who copies the
path out of the message gets one that does not exist. The refs here are literal
Windows paths, so each arm fails on every platform, not only on a Windows runner.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app import services as services_mod
from lilbee.cli.commands.search_chat import _swap_stale_models_to_installed
from lilbee.cli.tui import messages as tui_messages
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from lilbee.retrieval.embedder import Embedder
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
)
from tests.conftest import make_mock_services
from tests.test_fleet_provider import _fake_client, _provider_with_clients

WINDOWS_REF = "C:\\Users\\me\\models\\chat.gguf"
OTHER_REF = "D:\\Models\\fallback.gguf"
RELATIVE_REF = "models\\chat.gguf"


def _assert_verbatim(text: str, ref: str = WINDOWS_REF) -> None:
    assert f"'{ref}'" in text
    assert ref.replace("\\", "\\\\") not in text


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
    provider = _provider_with_clients({WorkerRole.CHAT: [_fake_client()]})
    provider._chat_ctx = 64
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 9000}]
    with pytest.raises(ProviderError) as excinfo:
        provider._fit_chat_context(msgs, None, None, WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))


@pytest.mark.parametrize(
    "template",
    [
        tui_messages.MODEL_FALLBACK_NOTICE,
        tui_messages.MODEL_FALLBACK_FAILED,
        tui_messages.MODEL_ENV_PIN_UNUSABLE,
        tui_messages.MODEL_CLI_PIN_UNUSABLE,
        tui_messages.MODEL_UNUSABLE_NO_FALLBACK,
    ],
)
def test_tui_model_notice_shows_the_original_as_typed(template: str) -> None:
    text = template.format(
        label="Chat", original=WINDOWS_REF, effective=OTHER_REF, reason="r", env_var="E"
    )
    _assert_verbatim(text)


@pytest.mark.parametrize(
    "template", [tui_messages.MODEL_FALLBACK_NOTICE, tui_messages.MODEL_FALLBACK_FAILED]
)
def test_tui_fallback_notice_shows_the_effective_ref_as_typed(template: str) -> None:
    text = template.format(
        label="Chat", original=WINDOWS_REF, effective=OTHER_REF, reason="r", env_var="E"
    )
    _assert_verbatim(text, OTHER_REF)


def test_tui_adopted_log_shows_the_effective_ref_as_typed() -> None:
    _assert_verbatim(tui_messages.MODEL_ADOPTED_LOG.format(label="Chat", effective=WINDOWS_REF))


def test_one_shot_swap_notice_shows_both_refs_as_typed(capsys) -> None:
    swapped = CanonicalRef(
        original=WINDOWS_REF,
        effective=OTHER_REF,
        status=ValidationResult.UNKNOWN,
        reason="it is unavailable",
    )
    untouched = CanonicalRef(original="", effective="", status=ValidationResult.UNKNOWN)
    with (
        patch("lilbee.modelhub.model_manager.canonicalize_chat_model", return_value=swapped),
        patch("lilbee.modelhub.model_manager.canonicalize_embedding_model", return_value=untouched),
        patch("lilbee.app.settings.apply_ephemeral_model_swap"),
    ):
        _swap_stale_models_to_installed()
    err = capsys.readouterr().err
    _assert_verbatim(err)
    _assert_verbatim(err, OTHER_REF)


def test_one_shot_adopt_notice_shows_the_installed_ref_as_typed(capsys) -> None:
    adopted = CanonicalRef(
        original="", effective=WINDOWS_REF, status=ValidationResult.UNKNOWN, reason="unset"
    )
    untouched = CanonicalRef(original="", effective="", status=ValidationResult.UNKNOWN)
    with (
        patch("lilbee.modelhub.model_manager.canonicalize_chat_model", return_value=adopted),
        patch("lilbee.modelhub.model_manager.canonicalize_embedding_model", return_value=untouched),
        patch("lilbee.app.settings.apply_ephemeral_model_swap"),
    ):
        _swap_stale_models_to_installed()
    _assert_verbatim(capsys.readouterr().err)


def test_unregistered_ref_error_shows_the_ref_as_typed() -> None:
    registry = MagicMock()
    registry.resolve.side_effect = KeyError("not in registry")
    services_mod.set_services(make_mock_services(registry=registry))
    try:
        with pytest.raises(ProviderError) as excinfo:
            engine_params.resolve_model_path(RELATIVE_REF)
    finally:
        services_mod.set_services(None)
    _assert_verbatim(str(excinfo.value), RELATIVE_REF)
    assert f"lilbee model pull {RELATIVE_REF}" in str(excinfo.value)


def test_chat_dispatch_not_found_shows_the_ref_as_typed() -> None:
    _assert_verbatim(str(ModelNotFoundError(WINDOWS_REF)))


def test_chat_dispatch_no_tools_shows_the_ref_as_typed() -> None:
    _assert_verbatim(str(ModelDoesNotSupportToolsError(WINDOWS_REF)))
