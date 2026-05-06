"""Persisted-model validation: classify refs and pick a session fallback."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import (
    CanonicalRef,
    ValidationResult,
    canonicalize_chat_model,
    canonicalize_embedding_model,
    validate_persisted_model,
)


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.llm_api_key = ""
    cfg.openai_api_key = ""
    cfg.anthropic_api_key = ""
    cfg.gemini_api_key = ""
    cfg.chat_model = "placeholder/chat"
    cfg.embedding_model = "placeholder/embed"
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


def test_empty_ref_unknown():
    assert validate_persisted_model("") == ValidationResult.UNKNOWN


def test_local_ref_not_installed_when_registry_empty():
    """A local-style ref with no GGUF on disk classifies as not OK."""
    result = validate_persisted_model("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf")
    assert result != ValidationResult.OK


def test_local_ref_installed_classifies_ok():
    """When the registry reports the ref as installed, validation returns OK."""
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/local-model"
    fake_entry.hf_repo = "test/local-model"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        assert validate_persisted_model("test/local-model") == ValidationResult.OK


def test_canonicalize_chat_model_ok_passthrough():
    """An OK ref is returned unchanged with status OK."""
    cfg.chat_model = "test/installed-model"
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/installed-model"
    fake_entry.hf_repo = "test/installed-model"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        canon = canonicalize_chat_model()
    assert isinstance(canon, CanonicalRef)
    assert canon.original == "test/installed-model"
    assert canon.effective == "test/installed-model"
    assert canon.status == ValidationResult.OK


def test_canonicalize_chat_model_falls_back_to_local():
    """When the persisted ref is invalid and no API key is configured,
    the helper falls back to the first installed local model."""
    cfg.chat_model = "missing/model"
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/fallback-local"
    fake_entry.hf_repo = "test/fallback-local"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={},
        ),
    ):
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        canon = canonicalize_chat_model()
    assert canon.effective == "test/fallback-local"
    assert canon.status != ValidationResult.OK


def test_canonicalize_chat_model_returns_original_when_no_fallback():
    """With no API keys and no installed locals, the helper returns the
    original ref so the caller can surface a hard error if needed."""
    cfg.chat_model = "missing/model"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={},
        ),
    ):
        registry_cls.return_value.list_installed.return_value = []
        canon = canonicalize_chat_model()
    assert canon.original == "missing/model"
    assert canon.effective == "missing/model"
    assert canon.status != ValidationResult.OK


def test_canonicalize_embedding_model_local_only():
    """Embedding fallback chain is local-only (no API equivalent)."""
    cfg.embedding_model = "missing/embed"
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/fallback-embed"
    fake_entry.hf_repo = "test/fallback-embed"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        canon = canonicalize_embedding_model()
    assert canon.effective == "test/fallback-embed"


def test_validate_handles_parse_error():
    """Malformed refs that crash parse_model_ref classify as UNKNOWN."""
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.parse_model_ref",
        side_effect=ValueError("malformed"),
    ):
        assert validate_persisted_model("garbage://ref") == ValidationResult.UNKNOWN


def test_validate_unknown_provider_classifies_unknown():
    """A non-local ref whose provider is not a configured field returns UNKNOWN."""
    fake_parsed = mock.MagicMock()
    fake_parsed.provider = "nonexistent_provider"
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.parse_model_ref",
        return_value=fake_parsed,
    ):
        assert (
            validate_persisted_model("nonexistent_provider/some-model") == ValidationResult.UNKNOWN
        )


def test_validate_known_provider_no_key_returns_no_key():
    """A recognized provider whose key is empty returns NO_KEY."""
    cfg.openai_api_key = ""
    fake_parsed = mock.MagicMock()
    fake_parsed.provider = "openai"
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.parse_model_ref",
        return_value=fake_parsed,
    ):
        assert validate_persisted_model("openai/gpt-4") == ValidationResult.NO_KEY


def test_canonicalize_chat_falls_back_to_api_when_keyed():
    """When discover_api_models returns a model and no local is installed,
    canonicalize uses the first API entry as the effective ref.

    The SDK backend exposes bare model names (``gpt-4-test``); the helper
    must prefix them with the provider so the result round-trips through
    Config's model-ref validator (which rejects bare names).
    """
    cfg.chat_model = "missing/model"
    fake_remote = mock.MagicMock()
    fake_remote.name = "gpt-4-test"
    fake_remote.provider = "OpenAI"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={"OpenAI": [fake_remote]},
        ),
    ):
        registry_cls.return_value.list_installed.return_value = []
        canon = canonicalize_chat_model()
    assert canon.effective == "openai/gpt-4-test"


def test_canonicalize_chat_prefixes_bare_provider_name():
    """The SDK backend reports models as bare names (``chatgpt-4o-latest``).

    Without prefixing, ``setattr(cfg, 'chat_model', name)`` would crash on
    Config's model-ref validator at app startup, taking down the TUI.
    """
    cfg.chat_model = "missing/model"
    bare = mock.MagicMock()
    bare.name = "chatgpt-4o-latest"
    bare.provider = "OpenAI"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={"OpenAI": [bare]},
        ),
    ):
        registry_cls.return_value.list_installed.return_value = []
        canon = canonicalize_chat_model()
    # The canonicalized ref must round-trip through Config's validator.
    from lilbee.providers.model_ref import parse_model_ref

    parse_model_ref(canon.effective)  # would raise on a bare name
    assert canon.effective == "openai/chatgpt-4o-latest"


def test_canonicalize_chat_handles_discover_failure():
    """If discover_api_models throws, canonicalize falls back to local."""
    cfg.chat_model = "missing/model"
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/fallback-local"
    fake_entry.hf_repo = "test/fallback-local"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            side_effect=RuntimeError("network down"),
        ),
    ):
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        canon = canonicalize_chat_model()
    assert canon.effective == "test/fallback-local"


def test_canonicalize_handles_registry_failure():
    """If ModelRegistry construction throws, fallback chain still works."""
    cfg.chat_model = "missing/model"
    with (
        mock.patch(
            "lilbee.modelhub.model_manager.validation.ModelRegistry",
            side_effect=OSError("models dir gone"),
        ),
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={},
        ),
    ):
        canon = canonicalize_chat_model()
    # No API key, no local registry -> falls back to original.
    assert canon.effective == "missing/model"


def test_canonicalize_embedding_returns_original_when_no_fallback():
    """Embedding has no API path, so a missing local registry returns original."""
    cfg.embedding_model = "missing/embed"
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.ModelRegistry",
        side_effect=OSError("models dir gone"),
    ):
        canon = canonicalize_embedding_model()
    assert canon.effective == "missing/embed"


def test_canonicalize_embedding_model_ok_passthrough():
    """A valid persisted embedding ref is returned unchanged."""
    cfg.embedding_model = "test/installed-embed"
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/installed-embed"
    fake_entry.hf_repo = "test/installed-embed"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        canon = canonicalize_embedding_model()
    assert canon.effective == "test/installed-embed"
    assert canon.status == ValidationResult.OK
