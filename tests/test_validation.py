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
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.ModelRegistry"
    ) as registry_cls:
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        assert validate_persisted_model("test/local-model") == ValidationResult.OK


def test_canonicalize_chat_model_ok_passthrough():
    """An OK ref is returned unchanged with status OK."""
    cfg.chat_model = "test/installed-model"
    fake_entry = mock.MagicMock()
    fake_entry.ref = "test/installed-model"
    fake_entry.hf_repo = "test/installed-model"
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.ModelRegistry"
    ) as registry_cls:
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
        mock.patch(
            "lilbee.modelhub.model_manager.validation.ModelRegistry"
        ) as registry_cls,
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
        mock.patch(
            "lilbee.modelhub.model_manager.validation.ModelRegistry"
        ) as registry_cls,
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
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.ModelRegistry"
    ) as registry_cls:
        registry_cls.return_value.list_installed.return_value = [fake_entry]
        canon = canonicalize_embedding_model()
    assert canon.effective == "test/fallback-embed"
