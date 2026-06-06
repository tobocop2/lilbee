"""litellm implementation of the ``LlmSdkBackend`` Protocol.

This is the ONLY file in lilbee that imports ``litellm``. When migrating
to a different SDK (e.g. ``liter-llm``), add a sibling module alongside
this one and flip the single import in ``providers/factory.py``.

All knowledge of the litellm wire format (``ollama/`` prefix, OpenAI
content-parts schema for images) lives here. The semantic layer in
``sdk_llm_provider`` never touches SDK-specific conventions.
"""

from __future__ import annotations

import base64
import functools
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from lilbee.core.config import DEFAULT_HTTP_TIMEOUT
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.local_servers import (
    OLLAMA,
    detect_local_server,
    local_server_for_key,
    openai_models_url,
)
from lilbee.providers.model_ref import ProviderModelRef
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
    SdkToolCall,
    SdkToolCallDelta,
    StreamChunk,
    detect_backend_name,
)

log = logging.getLogger(__name__)

_PROVIDER_NAME = "remote"

# Substrings dropped from the "LiteLLM" logger before they reach the user's
# terminal. Two classes of noise: (1) the model-cost-map fetch failure that
# LiteLLM logs at WARNING on every offline chat call, and (2) AWS-flavored
# advisories from sagemaker / bedrock / boto3 / botocore. lilbee's litellm
# extra deliberately excludes boto3, so the AWS warnings aren't actionable.
# Compared case-insensitively to catch the mixed-case variants LiteLLM emits.
_LITELLM_SUPPRESS_SUBSTRINGS = (
    "failed to fetch remote model cost map",
    "boto3",
    "botocore",
    "sagemaker",
    "bedrock",
)


class _LitellmSubstringFilter(logging.Filter):
    """Drop ``LiteLLM`` log records whose message contains a suppressed substring."""

    def __init__(self, needles: tuple[str, ...]) -> None:
        super().__init__()
        self._needles = tuple(n.lower() for n in needles)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().lower()
        return not any(n in msg for n in self._needles)


def install_litellm_log_filter() -> None:
    """Attach the ``LiteLLM`` substring filter to the package logger.

    Called automatically when this module is imported (see the module-top
    invocation below) so the filter is in place before any litellm call
    can emit a warning. Exposed as a function so tests can re-apply after
    clearing the logger.
    """
    logging.getLogger("LiteLLM").addFilter(_LitellmSubstringFilter(_LITELLM_SUPPRESS_SUBSTRINGS))


# Install the filter at module import. lilbee never touches litellm before
# importing this module, so installing here always beats litellm's first
# warning to the punch.
install_litellm_log_filter()


def _sdk_attr(obj: object, name: str) -> Any:
    """Read an optional attribute off a litellm response/chunk object (absent -> None).

    The single dynamic-read boundary for the SDK's loosely-typed objects, whose tool-call
    fields are absent (not just ``None``) across litellm chunk shapes.
    """
    return getattr(obj, name, None)


class _LitellmResponseView:
    """Typed read-only view over a litellm completion-response object.

    The litellm response shape is not in the SDK's type stubs. This
    adapter is the one place that knows how to pull ``model``, ``choices``,
    ``message_content`` and the streaming chunk fields out; SDK drift
    breaks here rather than across every caller.
    """

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def model(self) -> str | None:
        """The model name the SDK echoed back, if any."""
        value = getattr(self._response, "model", None)
        return str(value) if value is not None else None

    def _first_choice(self) -> Any:
        """First entry of the response's ``choices`` list, or ``None``."""
        choices = getattr(self._response, "choices", None) or []
        return choices[0] if choices else None

    @property
    def message_content(self) -> str:
        """Content text of the first choice's message (non-stream path)."""
        choice = self._first_choice()
        if choice is None:
            return ""
        message = getattr(choice, "message", None)
        if message is None:
            return ""
        return getattr(message, "content", "") or ""

    @property
    def delta_content(self) -> str:
        """Content delta of the first choice (stream-path chunk)."""
        choice = self._first_choice()
        if choice is None:
            return ""
        delta = getattr(choice, "delta", None)
        if delta is None:
            return ""
        return getattr(delta, "content", "") or ""

    @property
    def finish_reason(self) -> str | None:
        """``finish_reason`` of the first choice, if the SDK populated it."""
        choice = self._first_choice()
        return getattr(choice, "finish_reason", None) if choice is not None else None

    @property
    def tool_calls(self) -> tuple[SdkToolCall, ...]:
        """Tool calls from the first choice's message (non-stream path)."""
        choice = self._first_choice()
        if choice is None:
            return ()
        message = _sdk_attr(choice, "message")
        if message is None:
            return ()
        raw_calls = _sdk_attr(message, "tool_calls") or []
        return tuple(_extract_tool_call(call) for call in raw_calls)

    @property
    def delta_tool_calls(self) -> tuple[SdkToolCallDelta, ...]:
        """Tool-call deltas from the first choice's streaming delta."""
        choice = self._first_choice()
        if choice is None:
            return ()
        delta = _sdk_attr(choice, "delta")
        if delta is None:
            return ()
        raw_calls = _sdk_attr(delta, "tool_calls") or []
        return tuple(
            _extract_tool_call_delta(call, fallback_index=i) for i, call in enumerate(raw_calls)
        )


def _extract_tool_call(call: Any) -> SdkToolCall:
    """Pull one ``SdkToolCall`` out of a litellm tool-call object."""
    call_id = str(_sdk_attr(call, "id") or "")
    function = _sdk_attr(call, "function")
    name = str(_sdk_attr(function, "name") or "") if function is not None else ""
    arguments = str(_sdk_attr(function, "arguments") or "") if function is not None else ""
    return SdkToolCall(id=call_id, name=name, arguments=arguments)


def _extract_tool_call_delta(call: Any, *, fallback_index: int) -> SdkToolCallDelta:
    """Pull one ``SdkToolCallDelta`` out of a streaming chunk's tool-call slot.

    Empty-string ``name`` / ``arguments`` are normalised to ``None`` so the
    SDK stream shape matches the native worker's deltas (the dispatch's
    ``_StreamState`` gates on ``is not None``; emitting ``""`` produces a
    spurious empty ContentBlockDelta on every opener).
    """
    raw_index = _sdk_attr(call, "index")
    index = int(raw_index) if isinstance(raw_index, int) else fallback_index
    call_id = _sdk_attr(call, "id")
    function = _sdk_attr(call, "function")
    raw_name = _sdk_attr(function, "name") if function is not None else None
    raw_args = _sdk_attr(function, "arguments") if function is not None else None
    return SdkToolCallDelta(
        index=index,
        id=str(call_id) if call_id else None,
        name=str(raw_name) if raw_name else None,
        arguments_delta=str(raw_args) if raw_args else None,
    )


@functools.cache
def litellm_available() -> bool:
    """Return True if the ``litellm`` package is installed.

    Uses ``importlib.util.find_spec`` rather than ``import litellm`` so the
    check stays fast on the UI thread. Executing ``litellm`` on Windows
    with Defender real-time scanning takes seconds (the package loads a
    long list of provider plugins on first import); the Settings screen
    builds synchronously and calls this in ``_FEATURE_GATED_GROUPS``, so
    a real import here blocks the entire TUI on the first Settings open.
    ``find_spec`` just walks ``sys.path`` to locate the package; the
    heavy import runs later, in worker threads or remote-call paths
    where the cost is expected.
    """
    import importlib.util

    return importlib.util.find_spec("litellm") is not None


_LITELLM_MISSING_MSG = (
    "Remote and API models need the lilbee[remote] extra. "
    "Reinstall with: uv tool install --prerelease=allow 'lilbee[remote]'"
)


def _require_litellm() -> Any:
    """Import ``litellm`` or raise a user-facing ProviderError with install steps."""
    try:
        import litellm
    except ImportError as exc:
        raise ProviderError(_LITELLM_MISSING_MSG, provider=_PROVIDER_NAME) from exc
    return litellm


def _cache_ollama_defaults(model: str, params_text: str) -> None:
    """Parse Ollama parameters and store in the model defaults cache."""
    from lilbee.providers.model_defaults import parse_kv_parameters, set_defaults

    defaults = parse_kv_parameters(params_text)
    set_defaults(model, defaults)


def _route_model(ref: ProviderModelRef, api_base: str | None) -> str:
    """Format *ref* for litellm using the OpenAI ``provider/model`` convention.

    API and local-server refs already carry their canonical prefix. A bare
    ``local`` ref forced through the SDK (``llm_provider=remote``) gets the
    prefix of whichever local server its ``api_base`` points at.
    """
    if ref.is_api or local_server_for_key(ref.provider) is not None:
        return ref.for_openai_prefix()
    if api_base and (spec := detect_local_server(api_base)) is not None:
        return spec.qualify(ref.name)
    return ref.name


def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert messages with inline image bytes into OpenAI content parts.

    litellm routes to OpenAI-compatible endpoints that expect the
    ``{"type": "image_url", "image_url": {...}}`` content-parts schema
    for multimodal input. Messages without ``images`` pass through
    untouched.
    """
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        if "images" in msg:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": msg.get("content", "")}]
            for img in msg["images"]:
                if isinstance(img, bytes):
                    b64 = base64.b64encode(img).decode()
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
            formatted.append({"role": msg["role"], "content": content_parts})
        else:
            formatted.append(msg)
    return formatted


# User-facing message per recognised error kind. Each names the problem against
# {model} and makes clear the cause sits with the user's provider account or
# network, not with lilbee. UNKNOWN has no entry and falls back to the raw error.
_KIND_MESSAGES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.RATE_LIMIT: (
        "{model} is rate-limited or out of quota. That's a limit on your provider "
        "API key, not a lilbee problem. Check your plan and billing with the "
        "provider, or pick a different model."
    ),
    ProviderErrorKind.AUTH: (
        "{model} rejected your API key. Check that the key is set correctly and has "
        "access to this model. That's between your key and the provider, not a lilbee problem."
    ),
    ProviderErrorKind.NOT_FOUND: (
        "The provider doesn't offer {model} on your account. "
        "Pick a different model or check the name."
    ),
    ProviderErrorKind.CONTEXT_OVERFLOW: (
        "This conversation is too long for {model}'s context window. "
        "Start a new chat or pick a model with a larger context."
    ),
    ProviderErrorKind.BAD_REQUEST: (
        "The provider rejected the request for {model}. Check the model name and your settings."
    ),
    ProviderErrorKind.CONNECTION: (
        "Couldn't reach the provider for {model}, or it timed out. Check your "
        "connection and base URL, then try again or pick a different model."
    ),
    ProviderErrorKind.SERVER: (
        "The provider for {model} is unavailable right now. That's on the provider's "
        "side, not a lilbee problem. Try again shortly or pick a different model."
    ),
}

# Operation labels prefixed onto the fallback message for an unrecognised error.
_CHAT_FAILED = "Chat failed"
_EMBED_FAILED = "Embedding failed"
_RERANK_FAILED = "Rerank failed"


def _cause_chain(exc: BaseException) -> list[BaseException]:
    """Return *exc* and its causes, root cause first.

    litellm's mid-stream fallback keeps the real cause in ``original_exception``;
    walking root-first stops a 503 wrapper from masking the 429 it carries.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        nxt = getattr(cur, "original_exception", None)
        if not isinstance(nxt, BaseException):
            nxt = cur.__cause__
        cur = nxt if isinstance(nxt, BaseException) else None
    chain.reverse()
    return chain


def _classify_litellm_error(exc: BaseException) -> ProviderErrorKind:
    """Map a litellm exception to a ``ProviderErrorKind`` by type, never by message.

    litellm normalises every backend's failures into one exception hierarchy, so
    the same mapping covers all providers. The MRO walk picks the most specific
    kind (``ContextWindowExceededError`` over its ``BadRequestError`` base).
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover - unreachable after a real litellm call
        return ProviderErrorKind.UNKNOWN
    table: dict[type, ProviderErrorKind] = {
        litellm.AuthenticationError: ProviderErrorKind.AUTH,
        litellm.PermissionDeniedError: ProviderErrorKind.AUTH,
        litellm.NotFoundError: ProviderErrorKind.NOT_FOUND,
        litellm.RateLimitError: ProviderErrorKind.RATE_LIMIT,
        litellm.ContextWindowExceededError: ProviderErrorKind.CONTEXT_OVERFLOW,
        litellm.BadRequestError: ProviderErrorKind.BAD_REQUEST,
        litellm.Timeout: ProviderErrorKind.CONNECTION,
        litellm.APIConnectionError: ProviderErrorKind.CONNECTION,
        litellm.ServiceUnavailableError: ProviderErrorKind.SERVER,
        litellm.InternalServerError: ProviderErrorKind.SERVER,
    }
    for err in _cause_chain(exc):
        for cls in type(err).__mro__:
            kind = table.get(cls)
            if kind is not None:
                return kind
    return ProviderErrorKind.UNKNOWN


def _provider_error(fallback: str, exc: Exception, model: str) -> ProviderError:
    """Wrap a litellm failure as a ``ProviderError`` classified by type.

    Recognised kinds get a blob-free, user-facing message; unrecognised ones
    keep the raw ``{fallback}: {exc}`` shape so nothing is lost when debugging.
    """
    kind = _classify_litellm_error(exc)
    template = _KIND_MESSAGES.get(kind)
    message = template.format(model=model) if template is not None else f"{fallback}: {exc}"
    return ProviderError(message, provider=_PROVIDER_NAME, kind=kind)


class LitellmSdkBackend:
    """``LlmSdkBackend`` adapter backed by the ``litellm`` SDK."""

    @property
    def provider_name(self) -> str:
        """Stable identifier used when wrapping errors in ``ProviderError``."""
        return _PROVIDER_NAME

    def active_backend_name(self, base_url: str) -> str:
        """Return the display name of the backend ``base_url`` points at."""
        return detect_backend_name(base_url)

    def available(self) -> bool:
        """Return True if the underlying SDK is installed."""
        return litellm_available()

    def supports_tools(self, _model_ref: str) -> bool:
        """Optimistic: all SDK-routed refs report tool support.

        A model that lacks a tool template just returns an empty
        ``tool_calls`` array, which the dispatch handles as a normal
        end-of-turn.
        """
        return True

    def configure_logging(self, *, suppress_debug: bool) -> None:
        """Apply litellm's debug-info suppression toggle when requested."""
        if not suppress_debug:
            return
        try:
            import litellm

            litellm.suppress_debug_info = True
        except ImportError:
            pass  # debug-suppression is best-effort when the litellm extra is absent

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run a single-shot completion through ``litellm.completion``."""
        litellm = _require_litellm()
        kwargs = self._completion_kwargs(request, stream=False)
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, request.ref.for_display()) from exc
        view = _LitellmResponseView(response)
        return CompletionResult(
            content=view.message_content,
            finish_reason=view.finish_reason,
            model=view.model,
            tool_calls=view.tool_calls,
        )

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion through ``litellm.completion(stream=True)``."""
        litellm = _require_litellm()
        kwargs = self._completion_kwargs(request, stream=True)
        model = request.ref.for_display()
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc
        return self._stream_chunks(response, model)

    @staticmethod
    def _stream_chunks(response: Any, model: str) -> Iterator[StreamChunk]:
        """Yield ``StreamChunk`` values from a litellm streaming response.

        Exceptions raised mid-iteration are classified into ``ProviderError``
        so the semantic layer sees a consistent error type regardless of
        where the SDK failed.
        """
        try:
            for chunk in response:
                view = _LitellmResponseView(chunk)
                content = view.delta_content
                finish_reason = view.finish_reason
                tool_call_deltas = view.delta_tool_calls
                if content or finish_reason or tool_call_deltas:
                    yield StreamChunk(
                        content=content,
                        finish_reason=finish_reason,
                        tool_call_deltas=tool_call_deltas,
                    )
        except ProviderError:
            raise
        except Exception as exc:
            raise _provider_error(_CHAT_FAILED, exc, model) from exc

    @staticmethod
    def _completion_kwargs(request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        """Translate a ``CompletionRequest`` into litellm kwargs."""
        kwargs: dict[str, Any] = {
            "model": _route_model(request.ref, request.api_base),
            "messages": _format_messages(request.messages),
            "stream": stream,
        }
        if request.api_base:
            kwargs["api_base"] = request.api_base
        if request.api_key:
            kwargs["api_key"] = request.api_key
        if request.options:
            kwargs.update(request.options)
        return kwargs

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed inputs through ``litellm.embedding``."""
        litellm = _require_litellm()
        kwargs: dict[str, Any] = {
            "model": _route_model(request.ref, request.api_base),
            "input": request.inputs,
        }
        if request.api_base:
            kwargs["api_base"] = request.api_base
        if request.api_key:
            kwargs["api_key"] = request.api_key
        try:
            response = litellm.embedding(**kwargs)
        except Exception as exc:
            raise _provider_error(_EMBED_FAILED, exc, request.ref.for_display()) from exc
        data = response["data"] if isinstance(response, dict) else response.data
        vectors = [item["embedding"] for item in data]
        if isinstance(response, dict):
            model = response.get("model")
        else:
            model = getattr(response, "model", None)
        return EmbeddingResult(vectors=vectors, model=model)

    def rerank(self, request: RerankRequest) -> RerankResult:
        """Rerank documents via ``litellm.rerank`` (Cohere, Voyage, Jina, Together, HF TEI).

        The SDK returns results sorted by relevance; we restore input
        order via each result's ``index`` so scores line up with the
        caller's ``candidates`` list.
        """
        if not request.candidates:
            return RerankResult(scores=[])
        litellm = _require_litellm()
        kwargs: dict[str, Any] = {
            "model": _route_model(request.ref, request.api_base),
            "query": request.query,
            "documents": request.candidates,
        }
        if request.api_base:
            kwargs["api_base"] = request.api_base
        if request.api_key:
            kwargs["api_key"] = request.api_key
        try:
            response = litellm.rerank(**kwargs)
        except Exception as exc:
            raise _provider_error(_RERANK_FAILED, exc, request.ref.for_display()) from exc
        results = response["results"] if isinstance(response, dict) else response.results
        scores = [0.0] * len(request.candidates)
        for item in results:
            idx = item["index"] if isinstance(item, dict) else item.index
            score = item["relevance_score"] if isinstance(item, dict) else item.relevance_score
            scores[idx] = float(score)
        if isinstance(response, dict):
            model = response.get("model")
        else:
            model = getattr(response, "model", None)
        return RerankResult(scores=scores, model=model)

    def list_models(self, *, base_url: str, api_key: str) -> list[str]:
        """List models from Ollama (``/api/tags``) or an OpenAI-compatible ``/v1/models``."""
        clean_base = base_url.rstrip("/")
        spec = detect_local_server(clean_base)
        if spec is OLLAMA:
            return self._list_ollama_models(clean_base)
        return self._list_openai_models(clean_base, api_key)

    def list_chat_models(self, provider: str) -> list[str]:
        """Return chat-mode model ids from litellm's static catalog.

        Returns whatever litellm exposes for *provider*, alphabetically.
        Empty list when litellm is not installed or the provider has no
        chat-mode entries.
        """
        try:
            import litellm
        except ImportError:
            return []
        return self._all_chat_models_for(provider, litellm)

    @staticmethod
    def _all_chat_models_for(provider: str, litellm: Any) -> list[str]:
        """Filter litellm's catalog down to chat-mode entries for ``provider``.

        litellm's catalog stores some providers' models bare (``gpt-4o``)
        and others prefixed (``mistral/codestral-latest``,
        ``openrouter/anthropic/claude-3.5-sonnet``). Strip any leading
        ``{provider}/`` so callers see uniformly bare names; the canonical
        ``provider/name`` form is reapplied at the routing layer via
        :meth:`ProviderModelRef.for_openai_prefix`.
        """
        models = litellm.models_by_provider.get(provider, set())
        prefix = f"{provider}/"
        bare: set[str] = set()
        for model_name in models:
            info = litellm.model_cost.get(model_name, {})
            if info.get("mode") != "chat":
                continue
            bare.add(model_name.removeprefix(prefix))
        return sorted(bare)

    @staticmethod
    def _list_ollama_models(base_url: str) -> list[str]:
        """List models via the Ollama ``/api/tags`` endpoint."""
        try:
            resp = httpx.get(f"{base_url}/api/tags", timeout=DEFAULT_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot list models: {exc}", provider=_PROVIDER_NAME) from exc

    @staticmethod
    def _list_openai_models(base_url: str, api_key: str) -> list[str]:
        """List models via an OpenAI-compatible ``/v1/models`` endpoint."""
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = httpx.get(
                openai_models_url(base_url), headers=headers, timeout=DEFAULT_HTTP_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.HTTPError:
            log.debug("Failed to list models via /v1/models", exc_info=True)
            return []

    def pull_model(
        self,
        model: str,
        *,
        base_url: str,
        on_progress: Callable[..., Any] | None = None,
    ) -> None:
        """Refuse to pull: local servers (Ollama, LM Studio) are read-only.

        Their models are managed in their own app and surface here once
        present, so lilbee never downloads them over the network.
        """
        spec = detect_local_server(base_url.rstrip("/"))
        server = spec.display_name if spec is not None else "This server"
        raise ProviderError(
            f"{server} doesn't download models over the network. "
            f"Add the model in its own app, then pick it here.",
            provider=_PROVIDER_NAME,
        )

    def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
        """Get model info via the Ollama ``/api/show`` endpoint.

        Parses and caches per-model generation defaults from the
        ``parameters`` field. Also extracts the ``capabilities`` list
        (newer Ollama versions) so callers can check for vision support.
        Returns ``None`` for servers without a metadata endpoint (LM Studio).
        """
        clean_base = base_url.rstrip("/")
        spec = detect_local_server(clean_base)
        if spec is None or not spec.supports_show:
            return None
        # Ollama's API uses bare model names; the routing-layer prefix has
        # to come off before the request goes out.
        ollama_name = model.removeprefix(OLLAMA.wire_prefix)
        try:
            resp = httpx.post(
                f"{clean_base}/api/show",
                json={"name": ollama_name},
                timeout=DEFAULT_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return None

        result: dict[str, Any] = {}

        params = data.get("parameters", "")
        if isinstance(params, str) and params:
            _cache_ollama_defaults(model, params)
            result["parameters"] = params
        elif params:
            _cache_ollama_defaults(model, str(params))
            result["parameters"] = str(params)

        capabilities = data.get("capabilities")
        if isinstance(capabilities, list):
            result["capabilities"] = capabilities

        return result or None
