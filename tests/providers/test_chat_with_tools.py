"""Provider Protocol returns ``ChatResult`` and exposes ``supports_tools``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.providers.base import LLMProvider, ProviderError
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
    StreamChunk,
)
from lilbee.providers.worker.transport import (
    ChatResult,
    FinishReason,
    ToolCall,
    ToolCallDelta,
)


@dataclass
class _StubBackend:
    """Minimal ``LlmSdkBackend`` impl for chat / tool-support assertions."""

    complete_content: str = "hi"
    complete_finish: str = "stop"
    tools_supported: bool = False
    supports_tools_calls: list[str] = field(default_factory=list)
    provider_name: str = "fake"

    def available(self) -> bool:
        return True

    def configure_logging(self, *, suppress_debug: bool) -> None:
        return None

    def active_backend_name(self, base_url: str) -> str:
        return "Fake"

    def complete(self, request: CompletionRequest) -> CompletionResult:
        return CompletionResult(
            content=self.complete_content,
            finish_reason=self.complete_finish,
            model="m",
        )

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        yield StreamChunk(content=self.complete_content, finish_reason=self.complete_finish)

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(vectors=[])

    def rerank(self, request: RerankRequest) -> RerankResult:
        return RerankResult(scores=[])

    def list_models(self, *, base_url: str, api_key: str) -> list[str]:
        return []

    def list_chat_models(self, provider: str) -> list[str]:
        return []

    def pull_model(self, model: str, *, base_url: str, on_progress: Any = None) -> None:
        return None

    def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
        return None

    def supports_tools(self, model_ref: str) -> bool:
        self.supports_tools_calls.append(model_ref)
        return self.tools_supported


def _stub_pool(provider: Any, *, accessor: MagicMock, runtime: MagicMock) -> None:
    """Bypass pool registration so chat() can be exercised without a worker."""
    provider._get_pool_accessor = lambda *args, **kwargs: accessor  # type: ignore[method-assign]
    provider._pool_runtime = lambda: runtime  # type: ignore[method-assign]


def test_llama_cpp_chat_returns_chat_result_unchanged() -> None:
    """``LlamaCppProvider.chat`` returns the worker's ``ChatResult`` directly."""
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    accessor = MagicMock()
    runtime = MagicMock()
    expected = ChatResult(
        text="hi",
        tool_calls=(ToolCall(id="c1", name="f", arguments="{}"),),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    runtime.run_sync.return_value = expected
    _stub_pool(provider, accessor=accessor, runtime=runtime)

    result = provider.chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="auto",
    )
    assert isinstance(result, ChatResult)
    assert result.tool_calls[0].name == "f"
    assert result.finish_reason == FinishReason.TOOL_CALLS


def test_llama_cpp_chat_request_carries_tools() -> None:
    """The pool worker request includes the tool definitions."""
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider
    from lilbee.providers.worker.transport import ChatRequest

    provider = LlamaCppProvider()
    accessor = MagicMock()
    runtime = MagicMock()
    runtime.run_sync.return_value = ChatResult(
        text="ok", tool_calls=(), finish_reason=FinishReason.STOP
    )
    _stub_pool(provider, accessor=accessor, runtime=runtime)

    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    provider.chat([{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto")
    accessor.call.assert_called_once()
    sent_request = accessor.call.call_args.args[1]
    assert isinstance(sent_request, ChatRequest)
    assert sent_request.tools == tools
    assert sent_request.tool_choice == "auto"


def test_llama_cpp_supports_tools_when_template_mentions_tools(monkeypatch) -> None:
    """``supports_tools`` returns True iff the GGUF chat template references tools."""
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: {
            "chat_template": "{% if tools %}{{ tools }}{% endif %}",
        },
    )
    assert provider.supports_tools("any/model::Q4_K_M") is True


def test_llama_cpp_supports_tools_false_when_template_text_only(monkeypatch) -> None:
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: {"chat_template": "{{ messages }}"},
    )
    assert provider.supports_tools("any/model::Q4_K_M") is False


def test_llama_cpp_supports_tools_false_when_resolve_fails(monkeypatch) -> None:
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    def _boom(model: str) -> Path:
        raise ProviderError("missing", provider="llama-cpp")

    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.resolve_model_path", _boom)
    assert provider.supports_tools("unknown") is False


def test_llama_cpp_supports_tools_false_when_metadata_unreadable(monkeypatch) -> None:
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )

    def _boom(path: Path) -> dict[str, str]:
        raise OSError("nope")

    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.read_gguf_metadata", _boom)
    assert provider.supports_tools("any/model") is False


def test_llama_cpp_supports_tools_false_when_metadata_none(monkeypatch) -> None:
    """``read_gguf_metadata`` returns ``None`` for unreadable files; treat as no tools."""
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: None,
    )
    assert provider.supports_tools("any/model") is False


def test_llama_cpp_supports_tools_false_when_template_missing(monkeypatch) -> None:
    """Metadata without a ``chat_template`` key conservatively reports no tool support."""
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: {"architecture": "llama"},
    )
    assert provider.supports_tools("any/model") is False


def test_sdk_provider_chat_returns_chat_result() -> None:
    """``SdkLLMProvider.chat`` wraps the backend ``CompletionResult`` as ``ChatResult``."""
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    provider = SdkLLMProvider(_StubBackend(), base_url="http://localhost:11434")
    result = provider.chat([{"role": "user", "content": "hi"}])
    assert isinstance(result, ChatResult)
    assert result.text == "hi"
    assert result.finish_reason == FinishReason.STOP


def test_sdk_provider_rejects_tools_for_non_supporting_backend() -> None:
    """Backends without tool support raise ``ProviderError`` when ``tools`` is passed."""
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    backend = _StubBackend(tools_supported=False)
    provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
    with pytest.raises(ProviderError) as excinfo:
        provider.chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            model="ollama/llama3",
        )
    assert "does not support" in str(excinfo.value)
    assert backend.supports_tools_calls == ["ollama/llama3"]


def test_sdk_provider_supports_tools_delegates_to_backend() -> None:
    """``SdkLLMProvider.supports_tools`` forwards the call to the backend."""
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    backend = _StubBackend(tools_supported=True)
    provider = SdkLLMProvider(backend, base_url="http://x")
    assert provider.supports_tools("any") is True
    assert backend.supports_tools_calls == ["any"]


def test_sdk_provider_forwards_tools_and_tool_choice_when_supported() -> None:
    """When the backend supports tools, ``chat`` passes them through as options."""
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    backend = _StubBackend(tools_supported=True)
    provider = SdkLLMProvider(backend, base_url="http://x")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    captured: list[Any] = []

    original_complete = backend.complete

    def _capture(request: CompletionRequest) -> CompletionResult:
        captured.append(request)
        return original_complete(request)

    backend.complete = _capture  # type: ignore[method-assign]

    provider.chat(
        [{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "f"}},
        model="openai/gpt-4o",
    )
    assert captured and captured[0].options["tools"] == tools
    assert captured[0].options["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_litellm_supports_tools_true() -> None:
    """The litellm backend reports tool support for all cloud routes."""
    from lilbee.providers.litellm_sdk import LitellmSdkBackend

    backend = LitellmSdkBackend()
    assert backend.supports_tools("openai/gpt-4o") is True


def test_routing_provider_chat_returns_chat_result() -> None:
    """``RoutingProvider`` returns the routed backend's ``ChatResult`` unchanged."""
    from lilbee.providers.routing_provider import RoutingProvider

    provider = RoutingProvider()
    fake_backend = MagicMock(spec=LLMProvider)
    fake_backend.chat.return_value = ChatResult(
        text="ok", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider._llama_cpp = fake_backend

    result = provider.chat(
        [{"role": "user", "content": "hi"}],
        model="org/repo/file.gguf",
    )
    assert isinstance(result, ChatResult)
    assert result.text == "ok"


def test_routing_provider_supports_tools_delegates() -> None:
    from lilbee.providers.routing_provider import RoutingProvider

    provider = RoutingProvider()
    fake_backend = MagicMock(spec=LLMProvider)
    fake_backend.supports_tools.return_value = True
    provider._llama_cpp = fake_backend

    assert provider.supports_tools("org/repo/file.gguf") is True
    fake_backend.supports_tools.assert_called_once_with("org/repo/file.gguf")


def test_chat_stream_item_includes_tool_call_delta() -> None:
    """``ChatStreamItem`` is exported and covers both text + tool deltas."""
    from lilbee.providers.worker.transport import ChatStreamItem

    text: ChatStreamItem = "abc"
    delta: ChatStreamItem = ToolCallDelta(index=0, id="c1", name=None, arguments_delta=None)
    assert isinstance(text, str)
    assert isinstance(delta, ToolCallDelta)
