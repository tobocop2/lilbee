"""LiteLLM provider integration tests — real Ollama server, no mocks.

Requires litellm installed and Ollama running at OLLAMA_HOST (default localhost:11434)
with the qwen3:0.6b model pulled.
"""

from __future__ import annotations

import os

import pytest

litellm = pytest.importorskip("litellm")

from lilbee.config import cfg  # noqa: E402
from lilbee.providers.litellm_provider import LiteLLMProvider  # noqa: E402

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = "qwen3:0.6b"


def _ollama_reachable() -> bool:
    try:
        import httpx

        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not running"),
]


@pytest.fixture(autouse=True)
def _isolate_cfg():
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    yield
    for name, val in snapshot.items():
        setattr(cfg, name, val)


class TestLiteLLMEmbed:
    def test_embed_returns_vectors(self) -> None:
        """Real embedding via Ollama returns float vectors."""
        cfg.embedding_model = "nomic-embed-text"
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        result = provider.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) > 0
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_batch(self) -> None:
        """Batch embedding returns one vector per input."""
        cfg.embedding_model = "nomic-embed-text"
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        texts = ["hello", "world", "test"]
        result = provider.embed(texts)

        assert len(result) == 3
        assert all(len(v) > 0 for v in result)


class TestLiteLLMChat:
    def test_chat_returns_response(self) -> None:
        """Real chat completion via Ollama returns non-empty text."""
        cfg.chat_model = OLLAMA_MODEL
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        result = provider.chat(
            [{"role": "user", "content": "Say hello in exactly one word."}],
            options={"temperature": 0},
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_stream_yields_tokens(self) -> None:
        """Streaming chat yields string tokens."""
        cfg.chat_model = OLLAMA_MODEL
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        result = provider.chat(
            [{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            options={"temperature": 0},
        )

        tokens = list(result)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        full_text = "".join(tokens)
        assert len(full_text) > 0

    def test_chat_with_model_override(self) -> None:
        """Model override in chat() works."""
        cfg.chat_model = "should-not-be-used"
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        result = provider.chat(
            [{"role": "user", "content": "Say yes."}],
            model=OLLAMA_MODEL,
            options={"temperature": 0},
        )

        assert isinstance(result, str)
        assert len(result) > 0


class TestLiteLLMModelManagement:
    def test_list_models(self) -> None:
        """list_models returns models from Ollama."""
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        models = provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert any("qwen3" in m for m in models)

    def test_show_model(self) -> None:
        """show_model returns model info dict."""
        provider = LiteLLMProvider(base_url=OLLAMA_HOST)
        info = provider.show_model(OLLAMA_MODEL)

        assert info is not None
        assert isinstance(info, dict)


class TestLiteLLMFactory:
    def test_create_litellm_provider(self) -> None:
        """Factory creates LiteLLMProvider for 'litellm' provider."""
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "litellm"
        cfg.litellm_base_url = OLLAMA_HOST
        provider = create_provider(cfg)

        assert isinstance(provider, LiteLLMProvider)

    def test_ollama_alias(self) -> None:
        """Factory creates LiteLLMProvider for 'ollama' provider alias."""
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "ollama"
        cfg.litellm_base_url = OLLAMA_HOST
        provider = create_provider(cfg)

        assert isinstance(provider, LiteLLMProvider)
