"""A stand-in for the ``liter_llm`` module, injected where the extra is not installed.

It models the declared surface of liter-llm 2.4.0 (its type stub), never the adapter's
reading of it: reply objects carry the documented accessors and nothing the stub omits.
It has the async ``chat`` / ``chat_stream`` / ``embed`` / ``rerank`` functions, the
request and options types, and the typed exception hierarchy. Each call
records the fields it received so a test can assert what went on the wire.
"""

from __future__ import annotations

import json
import types
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lilbee.providers.factory import sdk_library
from tests._sys_modules import inject_modules

SDK_MODULE = sdk_library().module

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
    """Mirrors ``liter_llm.AssistantContent``: a text reply or a parts reply.

    ``text`` is a ``str`` for a text reply and ``None`` for a parts reply; ``parts`` is the
    reverse. Like the library's stub, it declares no ``__str__`` and is not a ``str``.
    """

    def __init__(self, text: str | None, parts: list[Any] | None) -> None:
        self.text = text
        self.parts = parts

    @staticmethod
    def from_text(value: str) -> AssistantContent:
        return AssistantContent(value, None)

    @staticmethod
    def from_parts(value: list[tuple[str, str]]) -> AssistantContent:
        return AssistantContent(None, [_ns(type=kind, text=text) for kind, text in value])


class _Enum:
    """A library enum member: compared and hashed by identity, with no ``__str__``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{type(self).__name__}.{self.name}>"


class FinishReason(_Enum):
    """Mirrors ``liter_llm.FinishReason``: the members the stub lists and nothing else."""

    STOP: FinishReason
    LENGTH: FinishReason
    TOOL_CALLS: FinishReason
    CONTENT_FILTER: FinishReason
    FUNCTION_CALL: FinishReason
    OTHER: FinishReason


FINISH_WIRE = {
    "STOP": "stop",
    "LENGTH": "length",
    "TOOL_CALLS": "tool_calls",
    "CONTENT_FILTER": "content_filter",
    "FUNCTION_CALL": "function_call",
    "OTHER": "other",
}
for _member in FINISH_WIRE:
    setattr(FinishReason, _member, FinishReason(_member))


def finish_member(wire: str) -> FinishReason:
    """The ``FinishReason`` member whose wire string is *wire*."""
    return getattr(FinishReason, wire.upper())


class ReasoningEffort(_Enum):
    """Mirrors ``liter_llm.ReasoningEffort`` 2.4.0."""

    LOW: ReasoningEffort
    MEDIUM: ReasoningEffort
    HIGH: ReasoningEffort
    MINIMAL: ReasoningEffort
    MAX: ReasoningEffort
    NONE: ReasoningEffort
    _members: ClassVar[dict[str, ReasoningEffort]] = {}

    def __init__(self, value: str) -> None:
        """Keep the member's own name; ``__new__`` already picked the member."""

    def __new__(cls, value: str) -> ReasoningEffort:
        try:
            return cls._members[value.upper()]
        except KeyError:
            raise ValueError(
                f"invalid value for ReasoningEffort: {value!r}. "
                "Expected variant name (str) or discriminant (int)"
            ) from None


def _make_efforts() -> None:
    for name in ("LOW", "MEDIUM", "HIGH", "MINIMAL", "MAX", "NONE"):
        member = object.__new__(ReasoningEffort)
        member.name = name
        ReasoningEffort._members[name] = member
        setattr(ReasoningEffort, name, member)


_make_efforts()


class Usage:
    """Mirrors ``liter_llm.Usage``."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


def text_response(
    content: str | None,
    *,
    model: str = "fake-model",
    finish: str = "stop",
    usage: Usage | None = None,
    parts: list[tuple[str, str]] | None = None,
) -> Any:
    """A chat response whose first choice carries *content* (``None`` for no content).

    With *parts*, the reply is a parts reply and *content* is ignored.
    """
    if parts is not None:
        body: Any = AssistantContent.from_parts(parts)
    else:
        body = None if content is None else AssistantContent.from_text(content)
    message = _ns(content=body, tool_calls=None)
    return _ns(
        model=model,
        choices=[_ns(message=message, finish_reason=finish_member(finish))],
        usage=usage,
    )


def tool_call_response(call_id: str, name: str, arguments: str) -> Any:
    """A chat response whose first choice asks for one tool call."""
    call = _ns(id=call_id, function=_ns(name=name, arguments=arguments))
    message = _ns(content=None, tool_calls=[call])
    return _ns(
        model="fake-model",
        choices=[_ns(message=message, finish_reason=FinishReason.TOOL_CALLS)],
        usage=None,
    )


def chunk(
    content: str | None = None,
    *,
    finish: str | None = None,
    tool_calls: list[Any] | None = None,
) -> Any:
    """A streaming chunk with one choice."""
    delta = _ns(content=content, tool_calls=tool_calls)
    reason = finish_member(finish) if finish is not None else None
    return _ns(choices=[_ns(delta=delta, finish_reason=reason)], usage=None)


def usage_chunk(usage: Usage) -> Any:
    """The closing stream chunk: no choices, the token totals."""
    return _ns(choices=[], usage=usage)


def tool_call_delta(
    index: int, *, call_id: str | None = None, name: str | None = None, arguments: str | None = None
) -> Any:
    """One streaming tool-call slot."""
    return _ns(index=index, id=call_id, function=_ns(name=name, arguments=arguments))


def embedding_response(
    vectors: list[tuple[int, list[float]]], model: str = "fake", *, as_given: bool = False
) -> Any:
    """An embedding response carrying ``(index, vector)`` items.

    The items arrive in reverse order, as a gateway may send them; ``as_given`` keeps the
    order of *vectors* for a test that scripts its own arrival order.
    """
    items = vectors if as_given else vectors[::-1]
    return _ns(data=[_ns(index=index, embedding=vector) for index, vector in items], model=model)


def rerank_response(scores: list[tuple[int, float]]) -> Any:
    """A rerank response carrying ``(index, score)`` results sorted by score, best first."""
    ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
    return _ns(results=[_ns(index=i, relevance_score=s) for i, s in ranked])


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
        mod.FinishReason = FinishReason  # type: ignore[attr-defined]
        mod.ReasoningEffort = ReasoningEffort  # type: ignore[attr-defined]
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
    from lilbee.providers.backends.liter_llm import client as sdk_client

    sdk_client._client.cache_clear()
    with inject_modules({SDK_MODULE: fake.module_cached()}):
        try:
            yield fake
        finally:
            sdk_client._client.cache_clear()
