"""A path or model ref in a user-facing message appears exactly as the user typed it.

``repr`` doubles the backslashes of a Windows path, so a reader who copies the
path out of the message gets one that does not exist. The refs here are literal
Windows paths, so each arm fails on every platform, not only on a Windows runner.
"""

from __future__ import annotations

import logging
from pathlib import PureWindowsPath
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app import services as services_mod
from lilbee.app.analyze import server_directory
from lilbee.cli.commands.search_chat import _swap_stale_models_to_installed
from lilbee.cli.tui import messages as tui_messages
from lilbee.core.config import cfg
from lilbee.data.ingest import pipeline as pipeline_mod
from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult
from lilbee.modelhub.registry import parse_hf_ref
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import WorkerRole, configured_model_message
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.providers.sdk_llm_provider import SdkLLMProvider
from lilbee.retrieval.embedder import Embedder
from lilbee.server.chat_dispatch.dispatch import (
    ModelDoesNotSupportToolsError,
    ModelNotFoundError,
)
from lilbee.server.handlers import ingest as ingest_handlers
from lilbee.wiki.lint import _lint_model_changed
from tests.conftest import make_mock_services
from tests.providers.test_sdk_llm_provider import FakeBackend
from tests.test_fleet_provider import _fake_client, _provider_with_clients

WINDOWS_REF = "C:\\Users\\me\\models\\chat.gguf"
OTHER_REF = "D:\\Models\\fallback.gguf"
RELATIVE_REF = "models\\chat.gguf"
HOSTED_REF = "ollama/models\\chat"
UNSUFFIXED_REF = "C:\\Users\\me\\models\\chat"


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


def test_ingest_refusal_shows_the_embedding_ref_as_typed(monkeypatch) -> None:
    services = MagicMock()
    services.embedder.validate_model.return_value = False
    monkeypatch.setattr(pipeline_mod, "get_services", lambda: services)
    monkeypatch.setattr(cfg, "embedding_model", WINDOWS_REF)
    with pytest.raises(RuntimeError) as excinfo:
        pipeline_mod._require_embedding_model()
    _assert_verbatim(str(excinfo.value))


def test_configured_model_message_shows_the_requested_ref_as_typed() -> None:
    _assert_verbatim(configured_model_message(WorkerRole.CHAT, "configured", WINDOWS_REF))


def test_routing_pull_refusal_shows_the_ref_as_typed() -> None:
    provider = RoutingProvider()
    sdk = MagicMock()
    sdk.available.return_value = False
    provider._sdk_provider = sdk
    with pytest.raises(ProviderError) as excinfo:
        provider.pull_model(WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))


def test_routing_rerank_refusal_shows_the_ref_as_typed(monkeypatch) -> None:
    provider = RoutingProvider()
    sdk = MagicMock()
    sdk.supports_rerank.return_value = False
    provider._sdk_provider = sdk
    provider._local = MagicMock()
    tagged = f"{HOSTED_REF}:latest"
    monkeypatch.setattr(cfg, "reranker_model", tagged)
    with pytest.raises(ProviderError) as excinfo:
        provider.rerank("q", ["a"])
    _assert_verbatim(str(excinfo.value), tagged)


def test_sdk_tool_refusal_shows_the_ref_as_typed() -> None:
    provider = SdkLLMProvider(FakeBackend(supports_tools_result=False))
    with pytest.raises(ProviderError) as excinfo:
        provider.chat(
            [{"role": "user", "content": "x"}],
            model=HOSTED_REF,
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
    _assert_verbatim(str(excinfo.value), HOSTED_REF)


def test_sdk_pull_refusal_shows_the_ref_as_typed() -> None:
    provider = SdkLLMProvider(FakeBackend(pull_not_supported=True))
    with pytest.raises(ProviderError) as excinfo:
        provider.pull_model(HOSTED_REF)
    _assert_verbatim(str(excinfo.value), HOSTED_REF)


def test_sdk_pull_failure_shows_the_ref_as_typed() -> None:
    class _Failing(FakeBackend):
        def pull_model(self, model: str, *, base_url: str, on_progress: Any = None) -> None:
            raise RuntimeError("boom")

    with pytest.raises(ProviderError) as excinfo:
        SdkLLMProvider(_Failing()).pull_model(HOSTED_REF)
    _assert_verbatim(str(excinfo.value), HOSTED_REF)


def test_sdk_show_failure_shows_the_ref_as_typed() -> None:
    class _Failing(FakeBackend):
        def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
            raise RuntimeError("boom")

    with pytest.raises(ProviderError) as excinfo:
        SdkLLMProvider(_Failing()).show_model(HOSTED_REF)
    _assert_verbatim(str(excinfo.value), HOSTED_REF)


def test_wiki_lint_model_change_shows_both_refs_as_typed(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "chat_model", WINDOWS_REF)
    text = f"---\ngenerated_by: {OTHER_REF}\n---\n\n# Doc\n"
    issue = _lint_model_changed("wiki/summaries/doc.md", text, cfg)
    assert issue is not None
    _assert_verbatim(issue.message)
    _assert_verbatim(issue.message, OTHER_REF)


def test_add_paths_refusal_shows_the_path_as_typed(monkeypatch) -> None:
    # On Windows a bare drive root has no name; PureWindowsPath gives the same on every host.
    monkeypatch.setattr(ingest_handlers, "Path", PureWindowsPath)
    with pytest.raises(ValueError) as excinfo:
        ingest_handlers.validate_add_paths({"paths": ["C:\\"]})
    _assert_verbatim(str(excinfo.value), "C:\\")


def test_analyze_directory_refusal_shows_the_path_as_typed() -> None:
    with pytest.raises(ValueError) as excinfo:
        server_directory(RELATIVE_REF)
    _assert_verbatim(str(excinfo.value), RELATIVE_REF)


def test_model_ref_shape_error_shows_the_ref_as_typed() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_model_ref(UNSUFFIXED_REF)
    _assert_verbatim(str(excinfo.value), UNSUFFIXED_REF)


def test_registry_ref_shape_error_shows_the_ref_as_typed() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_hf_ref(WINDOWS_REF)
    _assert_verbatim(str(excinfo.value))
