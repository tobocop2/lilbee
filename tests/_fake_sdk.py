"""A stand-in for the ``liter_llm`` module, injected where the extra is not installed.

It mirrors the parts of liter-llm 2.2.4 that ``lilbee.providers.remote_sdk``
calls: the async ``chat`` / ``chat_stream`` / ``embed`` / ``rerank`` functions,
the request and options types, and the typed exception hierarchy. Each call
records the fields it received so a test can assert what went on the wire.
"""

from __future__ import annotations

import json
import types
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from lilbee.providers.remote_sdk import SDK_MODULE
from tests._sys_modules import inject_modules

# liter-llm's exception classes: every one subclasses LiterLlmError.
ERROR_NAMES = (
    "AuthenticationError",
    "BadRequestError",
    "ContentPolicyError",
    "ContextWindowExceededError",
    "EndpointNotSupportedError",
    "InternalError",
    "LiterLlmTimeoutError",
    "NetworkError",
    "NotFoundError",
    "ProviderQuotaExceededError",
    "RateLimitedError",
    "ServerError",
    "ServiceUnavailableError",
    "StreamingError",
)


def _ns(**kwargs: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**kwargs)


class AssistantContent:
    """Mirrors ``liter_llm.AssistantContent``: not a ``str``; ``str()`` gives the text."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return repr(self._text)


class FinishReason:
    """Mirrors ``liter_llm.FinishReason``: not a ``str``; ``str()`` gives the wire value."""

    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return self._value


def text_response(content: str, *, model: str = "fake-model", finish: str = "stop") -> Any:
    """A chat response whose first choice carries *content*, typed like the real client's."""
    message = _ns(content=AssistantContent(content), tool_calls=None)
    return _ns(model=model, choices=[_ns(message=message, finish_reason=FinishReason(finish))])


def tool_call_response(call_id: str, name: str, arguments: str) -> Any:
    """A chat response whose first choice asks for one tool call."""
    call = _ns(id=call_id, function=_ns(name=name, arguments=arguments))
    message = _ns(content=None, tool_calls=[call])
    return _ns(
        model="fake-model",
        choices=[_ns(message=message, finish_reason=FinishReason("tool_calls"))],
    )


def chunk(
    content: str | None = None,
    *,
    finish: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    """A streaming chunk with one choice."""
    delta = _ns(content=content, tool_calls=tool_calls)
    reason = FinishReason(finish) if finish is not None else None
    return _ns(choices=[_ns(delta=delta, finish_reason=reason)])


def tool_call_delta(
    index: int, *, call_id: str | None = None, name: str | None = None, arguments: str | None = None
) -> Any:
    """One streaming tool-call slot."""
    return _ns(index=index, id=call_id, function=_ns(name=name, arguments=arguments))


def embedding_response(vectors: list[tuple[int, list[float]]], model: str = "fake") -> Any:
    """An embedding response carrying ``(index, vector)`` items in the given order."""
    return _ns(data=[_ns(index=index, embedding=vector) for index, vector in vectors], model=model)


def rerank_response(scores: list[tuple[int, float]]) -> Any:
    """A rerank response carrying ``(index, score)`` results in the given order."""
    return _ns(results=[_ns(index=i, relevance_score=s) for i, s in scores])


class _Fields:
    """A request type that keeps the keyword fields it was built with."""

    public_name = ""

    def __init__(self, **kwargs: Any) -> None:
        self.fields = kwargs


class _ClientOptions(_Fields):
    public_name = "ClientOptions"


class _ChatRequest(_Fields):
    public_name = "ChatCompletionRequest"


class _EmbedRequest(_Fields):
    public_name = "EmbeddingRequest"


class _RerankRequest(_Fields):
    public_name = "RerankRequest"


class _Tool:
    public_name = "ChatCompletionTool"

    def __init__(self, parsed: Any) -> None:
        self.parsed = parsed

    @staticmethod
    def from_json(text: str) -> _Tool:
        return _Tool(json.loads(text))


@dataclass
class FakeSdk:
    """Scripted liter-llm: set the result or error a call returns, read what it received."""

    chat_result: Any = None
    chat_error: BaseException | None = None
    require_key: bool = False
    stream_chunks: list[Any] = field(default_factory=list)
    stream_error: BaseException | None = None
    stream_open_error: BaseException | None = None
    embed_result: Any = None
    embed_error: BaseException | None = None
    rerank_result: Any = None
    rerank_error: BaseException | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    clients: list[dict[str, Any]] = field(default_factory=list)
    _module: types.ModuleType | None = field(default=None, repr=False)

    def last(self, name: str) -> dict[str, Any]:
        """The fields of the most recent call to *name*."""
        return [fields for call, fields in self.calls if call == name][-1]

    def module(self) -> types.ModuleType:
        """Build the module object to inject as the SDK module."""
        mod = types.ModuleType(SDK_MODULE)
        base = type("LiterLlmError", (Exception,), {})
        mod.LiterLlmError = base  # type: ignore[attr-defined]
        for name in ERROR_NAMES:
            setattr(mod, name, type(name, (base,), {}))
        for cls in (_ClientOptions, _ChatRequest, _EmbedRequest, _RerankRequest, _Tool):
            setattr(mod, cls.public_name, cls)
        mod.create_client_with_options = self._create_client  # type: ignore[attr-defined]
        mod.chat = self._chat  # type: ignore[attr-defined]
        mod.chat_stream = self._chat_stream  # type: ignore[attr-defined]
        mod.embed = self._embed  # type: ignore[attr-defined]
        mod.rerank = self._rerank  # type: ignore[attr-defined]
        return mod

    def _create_client(self, options: Any) -> Any:
        self.clients.append(options.fields)
        return _ns(options=options.fields)

    def _check_key(self, client: Any) -> None:
        """Refuse a hosted call with an empty key, like the real client with no env key."""
        options = client.options
        if self.require_key and not options["api_key"] and options["base_url"] is None:
            raise self.error("AuthenticationError", "no API key provided")

    async def _chat(self, client: Any, req: Any) -> Any:
        self._check_key(client)
        self.calls.append(("chat", req.fields))
        if self.chat_error is not None:
            raise self.chat_error
        return self.chat_result

    async def _chat_stream(self, client: Any, req: Any) -> AsyncIterator[Any]:
        self.calls.append(("chat_stream", req.fields))
        if self.stream_open_error is not None:
            raise self.stream_open_error
        for item in self.stream_chunks:
            yield item
        if self.stream_error is not None:
            raise self.stream_error

    async def _embed(self, client: Any, req: Any) -> Any:
        self.calls.append(("embed", req.fields))
        if self.embed_error is not None:
            raise self.embed_error
        return self.embed_result

    async def _rerank(self, client: Any, req: Any) -> Any:
        self.calls.append(("rerank", req.fields))
        if self.rerank_error is not None:
            raise self.rerank_error
        return self.rerank_result

    def error(self, name: str, message: str = "boom") -> BaseException:
        """Build an instance of the fake module's exception class *name*."""
        error_class: type[BaseException] = getattr(self.module_cached(), name)
        return error_class(message)

    def module_cached(self) -> types.ModuleType:
        """The module injected by ``installed``; built once per fake."""
        if self._module is None:
            self._module = self.module()
        return self._module


@contextmanager
def installed(fake: FakeSdk) -> Iterator[FakeSdk]:
    """Inject *fake* as ``liter_llm`` and start with an empty client cache."""
    from lilbee.providers import remote_sdk

    remote_sdk._client.cache_clear()
    with inject_modules({SDK_MODULE: fake.module_cached()}):
        try:
            yield fake
        finally:
            remote_sdk._client.cache_clear()
