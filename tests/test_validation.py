"""Persisted-model validation: classify refs and pick a session fallback."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lilbee.catalog.types import ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import (
    CanonicalRef,
    ValidationResult,
    canonicalize_chat_model,
    canonicalize_embedding_model,
    validate_persisted_model,
)
from lilbee.modelhub.registry import ModelManifest, ModelRegistry

_BLOB = b"GGUF-bytes"
_REPO = "Qwen/Qwen3-0.6B-GGUF"
_REF = f"{_REPO}/Qwen3-0.6B-Q4_K_M.gguf"
_SPLIT_REF = f"{_REPO}/Qwen3-0.6B-Q4_K_M-00001-of-00002.gguf"


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


def _installed(ref: str, task: ModelTask) -> mock.MagicMock:
    """Build a fake installed manifest with the given ref and task."""
    entry = mock.MagicMock()
    entry.ref = ref
    entry.hf_repo = ref
    entry.task = task
    return entry


def _holding(registry_cls: mock.MagicMock, entries: list[mock.MagicMock]) -> None:
    """Point a patched registry at *entries*, resolving those refs and nothing else."""
    refs = {entry.ref for entry in entries}
    registry_cls.return_value.list_installed.return_value = entries
    registry_cls.return_value.is_installed.side_effect = lambda ref: ref in refs


def _install(ref: str) -> None:
    """Install *ref* into ``cfg.models_dir`` the way a pull does: blob plus manifest."""
    hf_repo, filename = ref.rsplit("/", 1)
    source = cfg.models_dir / "source.gguf"
    source.write_bytes(_BLOB)
    ModelRegistry(cfg.models_dir).install(
        hf_repo,
        filename,
        source,
        ModelManifest(
            hf_repo=hf_repo,
            gguf_filename=filename,
            size_bytes=len(_BLOB),
            task=ModelTask.CHAT,
            downloaded_at="2026-04-25T00:00:00+00:00",
        ),
    )
    source.unlink()


def test_empty_ref_unknown():
    assert validate_persisted_model("") == ValidationResult.UNKNOWN


def test_local_ref_not_installed_when_registry_empty():
    """A local-style ref with no GGUF on disk classifies as not OK."""
    result = validate_persisted_model("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf")
    assert result != ValidationResult.OK


def test_local_ref_installed_classifies_ok():
    """A pulled model classifies as OK."""
    _install(_REF)
    assert validate_persisted_model(_REF) == ValidationResult.OK


def test_canonicalize_chat_model_ok_passthrough():
    """An OK ref is returned unchanged with status OK."""
    cfg.chat_model = "test/installed-model"
    fake_entry = _installed("test/installed-model", ModelTask.CHAT)
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        _holding(registry_cls, [fake_entry])
        canon = canonicalize_chat_model()
    assert isinstance(canon, CanonicalRef)
    assert canon.original == "test/installed-model"
    assert canon.effective == "test/installed-model"
    assert canon.status == ValidationResult.OK


def test_canonicalize_chat_model_falls_back_to_local():
    """When the persisted ref is invalid and no API key is configured,
    the helper falls back to the first installed local model."""
    from lilbee.catalog.types import ModelTask

    cfg.chat_model = "missing/model"
    fake_entry = _installed("test/fallback-local", ModelTask.CHAT)
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={},
        ),
    ):
        _holding(registry_cls, [fake_entry])
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
        _holding(registry_cls, [])
        canon = canonicalize_chat_model()
    assert canon.original == "missing/model"
    assert canon.effective == "missing/model"
    assert canon.status != ValidationResult.OK


def test_canonicalize_embedding_model_local_only():
    """Embedding fallback chain is local-only (no API equivalent)."""
    from lilbee.catalog.types import ModelTask

    cfg.embedding_model = "missing/embed"
    fake_entry = _installed("test/fallback-embed", ModelTask.EMBEDDING)
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        _holding(registry_cls, [fake_entry])
        canon = canonicalize_embedding_model()
    assert canon.effective == "test/fallback-embed"


def test_canonicalize_embedding_skips_installed_chat_model():
    """A chat model installed first must not become the embedding fallback (#307).

    The embedding slot is local-only, so the fallback walks the installed
    registry. Picking the first entry of any task hands a chat model to the
    embedding role; the role validator then rejects it. The fallback must be
    task-filtered.
    """
    from lilbee.catalog.types import ModelTask

    cfg.embedding_model = "missing/embed"
    chat_entry = mock.MagicMock()
    chat_entry.ref = "test/installed-chat"
    chat_entry.task = ModelTask.CHAT
    embed_entry = mock.MagicMock()
    embed_entry.ref = "test/installed-embed"
    embed_entry.task = ModelTask.EMBEDDING
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        _holding(registry_cls, [chat_entry, embed_entry])
        canon = canonicalize_embedding_model()
    assert canon.effective == "test/installed-embed"


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


def test_get_provider_api_key_unknown_provider_returns_none():
    """get_provider_api_key returns None for providers outside the dispatch map."""
    from lilbee.providers.sdk_backend import get_provider_api_key

    assert get_provider_api_key("nonexistent_provider") is None


def test_validate_known_provider_no_key_returns_no_key(monkeypatch):
    """A recognized provider with neither a config key nor an env key returns NO_KEY."""
    cfg.openai_api_key = ""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake_parsed = mock.MagicMock()
    fake_parsed.provider = "openai"
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.parse_model_ref",
        return_value=fake_parsed,
    ):
        assert validate_persisted_model("openai/gpt-4") == ValidationResult.NO_KEY


def test_validate_known_provider_with_env_key_returns_ok(monkeypatch):
    """An API ref whose key lives in the standard env var (not lilbee config) is OK.

    Regression guard: usability must honor ``OPENAI_API_KEY`` etc., not only
    the ``LILBEE_``-prefixed config field, or env-key users get sent to setup.
    """
    cfg.openai_api_key = ""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_parsed = mock.MagicMock()
    fake_parsed.provider = "openai"
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.parse_model_ref",
        return_value=fake_parsed,
    ):
        assert validate_persisted_model("openai/gpt-4") == ValidationResult.OK


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
        _holding(registry_cls, [])
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
        _holding(registry_cls, [])
        canon = canonicalize_chat_model()
    # The canonicalized ref must round-trip through Config's validator.
    from lilbee.providers.model_ref import parse_model_ref

    parse_model_ref(canon.effective)  # would raise on a bare name
    assert canon.effective == "openai/chatgpt-4o-latest"


def test_canonicalize_chat_handles_discover_failure():
    """If discover_api_models throws, canonicalize falls back to local."""
    from lilbee.catalog.types import ModelTask

    cfg.chat_model = "missing/model"
    fake_entry = _installed("test/fallback-local", ModelTask.CHAT)
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            side_effect=RuntimeError("network down"),
        ),
    ):
        _holding(registry_cls, [fake_entry])
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
        _holding(registry_cls, [fake_entry])
        canon = canonicalize_embedding_model()
    assert canon.effective == "test/installed-embed"


@pytest.fixture
def _litellm_absent():
    """Pretend the litellm extra is not installed (the reported crash scenario)."""
    with mock.patch(
        "lilbee.modelhub.model_manager.validation.litellm_available", return_value=False
    ):
        yield


def test_embedding_fallback_skips_chat_model(_litellm_absent):
    """A stale embedding ref must not fall back to an installed *chat* model.

    Regression: an unusable ``ollama/...`` embedder used to fall back to
    the first installed model of any task (a chat model), which the role
    validator then rejected and crashed startup.
    """
    cfg.embedding_model = "ollama/nomic-embed-text:latest"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        _holding(
            registry_cls,
            [_installed("owner/Phi-4-mini-instruct-GGUF/Phi-4.Q4_K_M.gguf", ModelTask.CHAT)],
        )
        canon = canonicalize_embedding_model()
    # No installed embedding model, so the original is kept (no bad swap).
    assert canon.effective == "ollama/nomic-embed-text:latest"
    assert canon.status != ValidationResult.OK


def test_embedding_fallback_picks_installed_embedding_model(_litellm_absent):
    """With both a chat and an embedding model installed, the embedding
    role falls back to the embedding model, never the chat model."""
    cfg.embedding_model = "ollama/nomic-embed-text:latest"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        _holding(
            registry_cls,
            [
                _installed("owner/Phi-4-mini-instruct-GGUF/Phi-4.Q4_K_M.gguf", ModelTask.CHAT),
                _installed("owner/nomic-embed-GGUF/nomic.Q8_0.gguf", ModelTask.EMBEDDING),
            ],
        )
        canon = canonicalize_embedding_model()
    assert canon.effective == "owner/nomic-embed-GGUF/nomic.Q8_0.gguf"
    assert canon.status != ValidationResult.OK


def test_ollama_ref_unusable_when_litellm_missing(_litellm_absent):
    """An ollama ref with the litellm extra absent is unusable, reason names litellm."""
    cfg.embedding_model = "ollama/nomic-embed-text:latest"
    with mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls:
        _holding(registry_cls, [])
        canon = canonicalize_embedding_model()
    assert canon.status != ValidationResult.OK
    assert canon.reason is not None and "litellm" in canon.reason


def test_ollama_ref_unusable_when_server_unreachable():
    """litellm present but the server is down: unusable, reason names the server."""
    cfg.embedding_model = "ollama/nomic-embed-text:latest"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.litellm_available", return_value=True),
        mock.patch(
            "lilbee.modelhub.model_manager.validation.classify_remote_models",
            return_value=[],
        ),
        mock.patch("lilbee.modelhub.model_manager.validation.ModelRegistry") as registry_cls,
    ):
        _holding(registry_cls, [])
        canon = canonicalize_embedding_model()
    assert canon.status != ValidationResult.OK
    assert canon.reason is not None and "reachable" in canon.reason


def test_ollama_ref_kept_when_server_live():
    """litellm present and the server lists models: keep the user's ollama ref."""
    cfg.embedding_model = "ollama/nomic-embed-text:latest"
    with (
        mock.patch("lilbee.modelhub.model_manager.validation.litellm_available", return_value=True),
        mock.patch(
            "lilbee.modelhub.model_manager.validation.classify_remote_models",
            return_value=[mock.MagicMock()],
        ),
    ):
        canon = canonicalize_embedding_model()
    assert canon.effective == "ollama/nomic-embed-text:latest"
    assert canon.status == ValidationResult.OK


class TestOneDefinitionOfInstalled:
    """Persisted-ref validation reads the install state the serving path reads.

    A second predicate over the manifest list answered the same question and
    disagreed in both directions: it called a loose GGUF path uninstalled, and
    it called a split set with a shard missing installed.
    """

    def test_a_loose_gguf_keeps_the_persisted_chat_ref(self, tmp_path: Path) -> None:
        """A GGUF path outside the registry loads, so nothing may swap it away."""
        gguf = tmp_path / "MiniMax.gguf"
        gguf.write_bytes(_BLOB)
        cfg.chat_model = str(gguf)

        assert validate_persisted_model(str(gguf)) == ValidationResult.OK
        assert canonicalize_chat_model().effective == str(gguf)

    def test_a_loose_gguf_keeps_the_persisted_chat_ref_over_an_api_model(
        self, tmp_path: Path
    ) -> None:
        """A configured API key must not displace a ref that already loads."""
        gguf = tmp_path / "MiniMax.gguf"
        gguf.write_bytes(_BLOB)
        cfg.chat_model = str(gguf)
        remote = mock.MagicMock()
        remote.name = "gpt-4-test"
        remote.provider = "OpenAI"
        with mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={"OpenAI": [remote]},
        ):
            canon = canonicalize_chat_model()

        assert canon.effective == str(gguf)
        assert canon.status == ValidationResult.OK

    def test_a_bare_repo_ref_is_installed(self) -> None:
        """Older builds persisted ``<org>/<repo>``; the registry still resolves it."""
        _install(_REF)

        assert validate_persisted_model(_REPO) == ValidationResult.OK

    def test_a_split_set_missing_a_shard_is_not_installed(self) -> None:
        """A manifest is not enough: the engine needs every shard of a split set."""
        _install(_SPLIT_REF)

        assert validate_persisted_model(_SPLIT_REF) == ValidationResult.NOT_INSTALLED
