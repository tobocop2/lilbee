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


def test_llama_cpp_supports_tools_memoizes_metadata_reads(monkeypatch, tmp_path) -> None:
    """Repeated calls for the same (path, mtime) read GGUF metadata only once.

    A tool-bearing chat request hits supports_tools on every call; without
    the lru_cache, every request would re-read the GGUF header from disk.
    """
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    real_file = tmp_path / "real.gguf"
    real_file.write_bytes(b"x")
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: real_file,
    )
    calls: list[Path] = []

    def _reader(path: Path) -> dict[str, str]:
        calls.append(path)
        return {"chat_template": "{% if tools %}{{ tools }}{% endif %}"}

    monkeypatch.setattr("lilbee.providers.llama_cpp.provider.read_gguf_metadata", _reader)
    assert provider.supports_tools("any/model::Q4_K_M") is True
    assert provider.supports_tools("any/model::Q4_K_M") is True
    assert provider.supports_tools("any/model::Q4_K_M") is True
    assert len(calls) == 1


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


def test_llama_cpp_supports_tools_true_when_chat_format_override_applies(monkeypatch) -> None:
    """Hermes-3 community GGUFs ship a stripped template but the chat_format
    override fills in the tool blocks at load time, so the model IS
    tool-capable from the route layer's perspective.
    """
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/hermes-3.gguf"),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: {
            "name": "Hermes 3 Llama 3.1 8B",
            # Stripped: no tools mention anywhere.
            "chat_template": "{{ messages }}",
        },
    )
    assert provider.supports_tools("any/model::Q4_K_M") is True


def test_llama_cpp_supports_tools_true_when_family_profile_has_schema(monkeypatch) -> None:
    """ERNIE/LFM2 use native tool-call wrappers their embedded chat template
    references in ways the generic Jinja probe doesn't match. The matching
    family profile (plus a registered response schema) is enough to declare
    the model tool-capable.
    """
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()

    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/ernie.gguf"),
    )
    # ERNIE marker triggers the ERNIE family profile, which has NO
    # chat_format_override but DOES have a registered response schema.
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: {
            "name": "ERNIE-4.5",
            "chat_template": "<|begin_of_sentence|>{{ messages }}<|end_of_sentence|>",
        },
    )
    assert provider.supports_tools("baidu/ERNIE-4.5") is True


def test_llama_cpp_supports_tools_false_when_tool_words_only_in_prose(monkeypatch) -> None:
    """A template that mentions tool words in literal text, not inside Jinja
    delimiters, must not report tool support. Matching anywhere in the string
    would route real tool requests to a model that can't render them.
    """
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
        lambda path: {"chat_template": "{{ messages }} (this template doesn't render tools)"},
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
    """``GGUFReader`` failures route through ``read_gguf_metadata`` and surface as ``False``."""
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider

    provider = LlamaCppProvider()
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda model: Path("/fake/path.gguf"),
    )

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("nope")

    monkeypatch.setattr("lilbee.providers.llama_cpp.gguf_meta.GGUFReader", _boom)
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


def test_litellm_supports_tools_true_optimistic() -> None:
    """The litellm backend reports tool support for any SDK-routed ref.

    litellm forwards tool definitions to the configured backend; lilbee
    extracts the response-side tool calls via ``_LitellmResponseView``.
    A specific model that lacks a tool template just returns an empty
    tool_calls list, which the dispatch handles as a normal end-of-turn.
    A strict per-model probe would falsely block every Ollama-routed
    chat since litellm has no Ollama-tag-level tool metadata.
    """
    from lilbee.providers.litellm_sdk import LitellmSdkBackend

    backend = LitellmSdkBackend()
    assert backend.supports_tools("openai/gpt-4o") is True
    assert backend.supports_tools("ollama/gemma4:26b") is True


def test_litellm_response_view_tool_calls_empty_on_missing_pieces() -> None:
    """Defensive null-paths in the litellm response view.

    A truncated or pre-init litellm response object can carry no choices,
    a choice with no message, or a stream chunk with no delta. The
    extractors must return an empty tuple in each case rather than
    raising; otherwise a transient SDK shape change would crash chat.
    """
    from types import SimpleNamespace

    from lilbee.providers.litellm_sdk import _LitellmResponseView

    # choices=None / empty list -> no tool calls
    no_choices = _LitellmResponseView(SimpleNamespace(choices=None))
    assert no_choices.tool_calls == ()
    assert no_choices.delta_tool_calls == ()
    # choice without a message -> empty
    no_message = _LitellmResponseView(SimpleNamespace(choices=[SimpleNamespace(message=None)]))
    assert no_message.tool_calls == ()
    # streaming chunk without a delta -> empty
    no_delta = _LitellmResponseView(
        SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason=None)])
    )
    assert no_delta.delta_tool_calls == ()


def test_litellm_complete_extracts_tool_calls(monkeypatch) -> None:
    """``LitellmSdkBackend.complete`` surfaces tool calls from the response."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from lilbee.providers.litellm_sdk import LitellmSdkBackend
    from lilbee.providers.model_ref import parse_model_ref
    from lilbee.providers.sdk_backend import CompletionRequest

    response = SimpleNamespace(
        model="ollama/gemma4:26b",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_42",
                            function=SimpleNamespace(
                                name="lilbee_search",
                                arguments='{"query": "chat worker"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )
    fake_litellm = MagicMock()
    fake_litellm.completion.return_value = response
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LitellmSdkBackend()
    request = CompletionRequest(
        ref=parse_model_ref("ollama/gemma4:26b"),
        messages=[{"role": "user", "content": "find the chat worker"}],
        api_base="http://localhost:11434",
    )
    result = backend.complete(request)
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "call_42"
    assert call.name == "lilbee_search"
    assert call.arguments == '{"query": "chat worker"}'
    assert result.finish_reason == "tool_calls"


def test_litellm_stream_extracts_tool_call_deltas(monkeypatch) -> None:
    """Streaming chunks surface ``tool_call_deltas`` so the dispatch can
    rebuild a complete ``ToolUseBlock`` from the accumulated frames.
    """
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from lilbee.providers.litellm_sdk import LitellmSdkBackend
    from lilbee.providers.model_ref import parse_model_ref
    from lilbee.providers.sdk_backend import CompletionRequest

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_7",
                                function=SimpleNamespace(name="lilbee_search", arguments=""),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments='{"q":"x"}'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="", tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]
    fake_litellm = MagicMock()
    fake_litellm.completion.return_value = iter(chunks)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LitellmSdkBackend()
    request = CompletionRequest(
        ref=parse_model_ref("ollama/gemma4:26b"),
        messages=[{"role": "user", "content": "search"}],
        api_base="http://localhost:11434",
    )
    collected = list(backend.complete_stream(request))
    # First chunk opens the tool call with id + name.
    opener_deltas = collected[0].tool_call_deltas
    assert opener_deltas[0].id == "call_7"
    assert opener_deltas[0].name == "lilbee_search"
    # Second chunk continues with argument bytes.
    args_deltas = collected[1].tool_call_deltas
    assert args_deltas[0].arguments_delta == '{"q":"x"}'
    # Final chunk carries the finish reason.
    assert collected[-1].finish_reason == "tool_calls"


def test_sdk_provider_propagates_tool_calls_to_chat_result(monkeypatch) -> None:
    """``SdkLLMProvider.chat`` lifts SDK tool calls into ``ChatResult.tool_calls``."""
    from lilbee.providers.sdk_backend import CompletionResult, SdkToolCall
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    backend = _StubBackend(tools_supported=True)
    backend.complete = MagicMock(  # type: ignore[method-assign]
        return_value=CompletionResult(
            content="",
            finish_reason="tool_calls",
            model="ollama/gemma4:26b",
            tool_calls=(SdkToolCall(id="c1", name="lilbee_search", arguments='{"q":"x"}'),),
        )
    )
    provider = SdkLLMProvider(backend=backend, base_url="http://localhost:11434")
    result = provider.chat(
        [{"role": "user", "content": "go"}],
        tools=[{"type": "function", "function": {"name": "lilbee_search"}}],
        model="ollama/gemma4:26b",
    )
    assert isinstance(result, ChatResult)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "c1"
    assert result.tool_calls[0].name == "lilbee_search"


def test_sdk_provider_stream_yields_tool_call_deltas(monkeypatch) -> None:
    """The streaming path yields ``ToolCallDelta`` items between content tokens."""
    from lilbee.providers.sdk_backend import SdkToolCallDelta, StreamChunk
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

    backend = _StubBackend(tools_supported=True)

    def _stream(_req):
        yield StreamChunk(
            content="",
            tool_call_deltas=(SdkToolCallDelta(index=0, id="c1", name="lilbee_search"),),
        )
        yield StreamChunk(
            content="",
            tool_call_deltas=(SdkToolCallDelta(index=0, arguments_delta='{"q":"x"}'),),
        )
        yield StreamChunk(content="", finish_reason="tool_calls")

    backend.complete_stream = MagicMock(side_effect=_stream)  # type: ignore[method-assign]
    provider = SdkLLMProvider(backend=backend, base_url="http://localhost:11434")
    frames = list(
        provider.chat(
            [{"role": "user", "content": "go"}],
            stream=True,
            tools=[{"type": "function", "function": {"name": "lilbee_search"}}],
            model="ollama/gemma4:26b",
        )
    )
    deltas = [f for f in frames if isinstance(f, ToolCallDelta)]
    assert len(deltas) == 2
    assert deltas[0].id == "c1"
    assert deltas[1].arguments_delta == '{"q":"x"}'


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
