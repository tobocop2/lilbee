"""SDK-backed provider integration tests against an Ollama-shaped stub.

Spawns ``_ollama_stub.py`` over real HTTP: no daemon, no model pulls,
deterministic. Drift against a real Ollama daemon is covered by the
scheduled ollama-pypi lane in qa-matrix.yml.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest

litellm = pytest.importorskip("litellm")

from lilbee.core.config import cfg  # noqa: E402
from lilbee.providers import litellm_sdk as _litellm_sdk_mod  # noqa: E402
from lilbee.providers.base import (  # noqa: E402
    ChatResult,
    StreamFinish,
    TokenUsage,
    ToolCallDelta,
)
from lilbee.providers.litellm_sdk import LitellmSdkBackend  # noqa: E402
from lilbee.providers.local_servers import OLLAMA  # noqa: E402
from lilbee.providers.local_servers.spec import LocalServerSpec  # noqa: E402
from lilbee.providers.sdk_llm_provider import SdkLLMProvider  # noqa: E402

_STUB = Path(__file__).parent / "_ollama_stub.py"
# Ollama keeps its own ``name:tag`` shape; lilbee's config layer requires
# the ``ollama/`` prefix so its routing knows where to send the request.
OLLAMA_MODEL = "ollama/qwen3:0.6b"
OLLAMA_EMBED_MODEL = "ollama/nomic-embed-text"


pytestmark = [pytest.mark.slow]


def _pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def ollama_stub_base() -> Iterator[str]:
    """Base URL of a freshly spawned Ollama stub server."""
    port = _pick_free_port()
    proc = subprocess.Popen([sys.executable, str(_STUB), "--port", str(port)])
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/api/tags", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise RuntimeError("ollama stub did not become ready")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10.0)


@pytest.fixture(autouse=True)
def _isolate_cfg(ollama_stub_base: str):
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    # ollama/ refs resolve their api_base from this field, so point it at the
    # stub for the duration of each test.
    cfg.ollama_base_url = ollama_stub_base
    yield
    for name, val in snapshot.items():
        setattr(cfg, name, val)


@pytest.fixture(autouse=True)
def _stub_detects_as_ollama(monkeypatch: pytest.MonkeyPatch, ollama_stub_base: str):
    """Map the stub URL to the Ollama spec; its random port matches no URL pattern."""
    real_detect = _litellm_sdk_mod.detect_local_server

    def _detect(base_url: str) -> LocalServerSpec | None:
        if base_url.rstrip("/") == ollama_stub_base:
            return OLLAMA
        return real_detect(base_url)

    monkeypatch.setattr(_litellm_sdk_mod, "detect_local_server", _detect)


class TestSdkEmbed:
    def test_embed_returns_vectors(self) -> None:
        """Embedding via the stub returns float vectors."""
        cfg.embedding_model = OLLAMA_EMBED_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend())
        result = provider.embed(["hello world"])

        assert len(result) == 1
        assert len(result[0]) > 0
        assert result[0].dtype == np.float32

    def test_embed_batch(self) -> None:
        """Batch embedding returns one vector per input."""
        cfg.embedding_model = OLLAMA_EMBED_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend())
        texts = ["hello", "world", "test"]
        result = provider.embed(texts)

        assert len(result) == 3
        assert all(len(v) > 0 for v in result)


class TestSdkChat:
    def test_chat_returns_response(self) -> None:
        """Chat completion via the stub returns non-empty text."""
        cfg.chat_model = OLLAMA_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend())
        result = provider.chat(
            [{"role": "user", "content": "Say hello in exactly one word."}],
            options={"temperature": 0},
        )

        assert isinstance(result, ChatResult)
        assert len(result.text) > 0

    def test_chat_stream_yields_tokens(self) -> None:
        """Streaming chat yields text, tool-call deltas, and a closing finish frame."""
        cfg.chat_model = OLLAMA_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend())
        result = provider.chat(
            [{"role": "user", "content": "Count from 1 to 3."}],
            stream=True,
            options={"temperature": 0},
        )

        items = list(result)
        assert len(items) > 0
        assert all(isinstance(t, (str, ToolCallDelta, TokenUsage, StreamFinish)) for t in items)
        full_text = "".join(t for t in items if isinstance(t, str))
        assert len(full_text) > 0
        assert any(isinstance(t, StreamFinish) for t in items)

    def test_chat_with_model_override(self) -> None:
        """Model override in chat() works."""
        cfg.chat_model = OLLAMA_MODEL
        provider = SdkLLMProvider(LitellmSdkBackend())
        result = provider.chat(
            [{"role": "user", "content": "Say yes."}],
            model=OLLAMA_MODEL,
            options={"temperature": 0},
        )

        assert isinstance(result, ChatResult)
        assert len(result.text) > 0


class TestSdkModelManagement:
    def test_list_models(self) -> None:
        """list_models returns the stub's models."""
        provider = SdkLLMProvider(LitellmSdkBackend())
        models = provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert any("qwen3" in m for m in models)

    def test_show_model(self) -> None:
        """show_model returns model info dict."""
        provider = SdkLLMProvider(LitellmSdkBackend())
        info = provider.show_model(OLLAMA_MODEL)

        assert info is not None
        assert isinstance(info, dict)


class TestSdkFactory:
    def test_create_sdk_provider_for_litellm_config(self) -> None:
        """Factory wraps the SDK backend in SdkLLMProvider when cfg.llm_provider == "remote"."""
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "remote"
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
