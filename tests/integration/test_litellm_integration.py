"""SDK-backed provider integration tests: real Ollama server, no mocks.

Requires litellm installed and Ollama running at OLLAMA_HOST (default
localhost:11434) with the qwen3:0.6b and nomic-embed-text models pulled.
"""

from __future__ import annotations

import os

import pytest

litellm = pytest.importorskip("litellm")

from lilbee.core.config import cfg  # noqa: E402
from lilbee.providers.base import ChatResult, ToolCallDelta  # noqa: E402
from lilbee.providers.litellm_sdk import LitellmSdkBackend  # noqa: E402
from lilbee.providers.sdk_llm_provider import SdkLLMProvider  # noqa: E402

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
# Ollama keeps its own ``name:tag`` shape; lilbee's config layer requires
# the ``ollama/`` prefix so its routing knows where to send the request.
OLLAMA_MODEL = "ollama/qwen3:0.6b"
OLLAMA_EMBED_MODEL = "ollama/nomic-embed-text"


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


class TestSdkEmbed:
    def test_embed_returns_vectors(self) -> None:
        """Real embedding via Ollama returns float vectors."""
        cfg.embedding_model = OLLAMA_EMBED_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        result = provider.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) > 0
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_batch(self) -> None:
        """Batch embedding returns one vector per input."""
        cfg.embedding_model = OLLAMA_EMBED_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        texts = ["hello", "world", "test"]
        result = provider.embed(texts)

        assert len(result) == 3
        assert all(len(v) > 0 for v in result)


class TestSdkChat:
    def test_chat_returns_response(self) -> None:
        """Real chat completion via Ollama returns non-empty text."""
        cfg.chat_model = OLLAMA_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        result = provider.chat(
            [{"role": "user", "content": "Say hello in exactly one word."}],
            options={"temperature": 0},
        )

        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    def test_chat_stream_yields_tokens(self) -> None:
        """Streaming chat yields text and (optionally) tool-call deltas."""
        cfg.chat_model = OLLAMA_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        result = provider.chat(
            [{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            options={"temperature": 0},
        )

        items = list(result)
        assert len(items) > 0
        assert all(isinstance(t, (str, ToolCallDelta)) for t in items)
        full_text = "".join(t for t in items if isinstance(t, str))
        assert len(full_text) > 0

    def test_chat_with_model_override(self) -> None:
        """Model override in chat() works."""
        cfg.chat_model = OLLAMA_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        result = provider.chat(
            [{"role": "user", "content": "Say yes."}],
            model=OLLAMA_MODEL,
            options={"temperature": 0},
        )

        assert isinstance(result, ChatResult)
        assert len(result.text) > 0


class TestSdkModelManagement:
    def test_list_models(self) -> None:
        """list_models returns models from Ollama."""
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        models = provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert any("qwen3" in m for m in models)

    def test_show_model(self) -> None:
        """show_model returns model info dict."""
        provider = SdkLLMProvider(LitellmSdkBackend(), base_url=OLLAMA_HOST)
        info = provider.show_model(OLLAMA_MODEL)

        assert info is not None
        assert isinstance(info, dict)


class TestSdkFactory:
    def test_create_sdk_provider_for_litellm_config(self) -> None:
        """Factory wraps the SDK backend in SdkLLMProvider when cfg.llm_provider == "remote"."""
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "remote"
        cfg.remote_base_url = OLLAMA_HOST
        provider = create_provider(cfg)

        assert isinstance(provider, SdkLLMProvider)

    def test_ollama_alias_rejected(self) -> None:
        """'ollama' is not a valid llm_provider value (use 'remote' for Ollama).

        Now a validated ``LlmProvider`` enum, it is rejected at the config
        boundary on assignment rather than later in ``create_provider``.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cfg.llm_provider = "ollama"
