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
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from lilbee.core.config import DEFAULT_HTTP_TIMEOUT
from lilbee.providers.base import ProviderError
from lilbee.providers.model_ref import OLLAMA_PREFIX, ProviderModelRef
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
    StreamChunk,
    detect_backend_name,
)

log = logging.getLogger(__name__)

_PROVIDER_NAME = "litellm"
_OLLAMA_URL_PATTERNS = ("localhost:11434", "127.0.0.1:11434", "ollama")


def _is_ollama(base_url: str) -> bool:
    """Return True if *base_url* looks like an Ollama instance."""
    url_lower = base_url.lower()
    return any(p in url_lower for p in _OLLAMA_URL_PATTERNS)


def litellm_available() -> bool:
    """Return True if ``litellm`` can be imported."""
    try:
        import litellm  # noqa: F401
    except ImportError:
        return False
    return True


_LITELLM_MISSING_MSG = (
    "Remote and API models need the lilbee[litellm] extra. "
    "Reinstall with: uv tool install --prerelease=allow 'lilbee[litellm]'"
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
    from lilbee.modelhub.model_defaults import parse_kv_parameters, set_defaults

    defaults = parse_kv_parameters(params_text)
    set_defaults(model, defaults)


def _route_model(ref: ProviderModelRef, api_base: str | None) -> str:
    """Format *ref* for litellm using the OpenAI ``provider/model`` convention."""
    if ref.is_api:
        return ref.for_openai_prefix()
    if api_base and _is_ollama(api_base):
        return f"{OLLAMA_PREFIX}{ref.name}"
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

    def configure_logging(self, *, suppress_debug: bool) -> None:
        """Apply litellm's debug-info suppression toggle when requested."""
        if not suppress_debug:
            return
        try:
            import litellm

            litellm.suppress_debug_info = True
        except ImportError:
            pass

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run a single-shot completion through ``litellm.completion``."""
        litellm = _require_litellm()
        kwargs = self._completion_kwargs(request, stream=False)
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Chat failed: {exc}", provider=_PROVIDER_NAME) from exc
        choices = getattr(response, "choices", None) or []
        message = choices[0].message if choices else None
        content = getattr(message, "content", "") if message is not None else ""
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        return CompletionResult(
            content=content or "",
            finish_reason=finish_reason,
            model=getattr(response, "model", None),
        )

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion through ``litellm.completion(stream=True)``."""
        litellm = _require_litellm()
        kwargs = self._completion_kwargs(request, stream=True)
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            raise ProviderError(f"Chat failed: {exc}", provider=_PROVIDER_NAME) from exc
        return self._stream_chunks(response)

    @staticmethod
    def _stream_chunks(response: Any) -> Iterator[StreamChunk]:
        """Yield ``StreamChunk`` values from a litellm streaming response.

        Exceptions raised mid-iteration are wrapped in ``ProviderError``
        so the semantic layer sees a consistent error type regardless of
        where the SDK failed.
        """
        try:
            for chunk in response:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", "") if delta is not None else ""
                finish_reason = getattr(choices[0], "finish_reason", None)
                if content or finish_reason:
                    yield StreamChunk(content=content or "", finish_reason=finish_reason)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Chat failed: {exc}", provider=_PROVIDER_NAME) from exc

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
            raise ProviderError(f"Embedding failed: {exc}", provider=_PROVIDER_NAME) from exc
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
            raise ProviderError(f"Rerank failed: {exc}", provider=_PROVIDER_NAME) from exc
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
        """List models from Ollama or an OpenAI-compatible server."""
        clean_base = base_url.rstrip("/")
        if _is_ollama(clean_base):
            return self._list_ollama_models(clean_base)
        return self._list_openai_models(clean_base, api_key)

    def list_chat_models(self, provider: str, *, mode: str = "curated") -> list[str]:
        """Return chat-mode model ids from litellm's static catalog.

        ``mode="curated"`` returns the curated short list per
        :mod:`lilbee.providers.curated_models`, falling back to an
        alphabetical top-N from the upstream catalog when no curated
        entry exists. ``mode="all"`` returns the full upstream catalog.

        Empty list when litellm is not installed or the provider has no
        chat-mode entries. Callers that care about the litellm debug
        banner should invoke ``configure_logging`` first (the semantic
        layer's ``SdkLLMProvider`` does this in ``list_chat_models``).
        """
        try:
            import litellm
        except ImportError:
            return []
        all_chat = self._all_chat_models_for(provider, litellm)
        if mode == "all":
            return all_chat
        return self._curated_chat_models_for(provider, all_chat)

    @staticmethod
    def _all_chat_models_for(provider: str, litellm: Any) -> list[str]:
        """Filter litellm's catalog down to chat-mode entries for ``provider``."""
        models = litellm.models_by_provider.get(provider, set())
        chat_models: list[str] = []
        for model_name in sorted(models):
            info = litellm.model_cost.get(model_name, {})
            if info.get("mode") != "chat":
                continue
            chat_models.append(model_name)
        return chat_models

    @staticmethod
    def _curated_chat_models_for(provider: str, all_chat: list[str]) -> list[str]:
        """Pick the curated short list, or top-N alphabetical when uncurated.

        Filters the curated set down to ids actually present in the
        upstream catalog so a provider rename doesn't surface a dead
        entry. New providers without a curated entry get the alphabetical
        top-N so they auto-graduate without flooding the picker.
        """
        from lilbee.providers.curated_models import (
            CURATED_CHAT_MODELS,
            TOP_N_FALLBACK,
            curated_ids,
        )

        if provider in CURATED_CHAT_MODELS:
            available = set(all_chat)
            return [mid for mid in curated_ids(provider) if mid in available]
        return all_chat[:TOP_N_FALLBACK]

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
            resp = httpx.get(f"{base_url}/v1/models", headers=headers, timeout=DEFAULT_HTTP_TIMEOUT)
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
        """Pull a model via the Ollama ``/api/pull`` endpoint."""
        clean_base = base_url.rstrip("/")
        try:
            with (
                # Streaming Ollama /api/pull; unbounded read is intentional
                # since model downloads can exceed any wall-clock timeout.
                httpx.Client(timeout=None) as client,  # noqa: S113
                client.stream(
                    "POST",
                    f"{clean_base}/api/pull",
                    json={"name": model, "stream": True},
                ) as resp,
            ):
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if on_progress:
                        on_progress(event)
                    if event.get("status") == "success":
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Cannot pull model {model!r}: {exc}", provider=_PROVIDER_NAME
            ) from exc

    def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
        """Get model info via the Ollama ``/api/show`` endpoint.

        Parses and caches per-model generation defaults from the
        ``parameters`` field. Also extracts the ``capabilities`` list
        (newer Ollama versions) so callers can check for vision support.
        """
        clean_base = base_url.rstrip("/")
        # Ollama's API uses bare model names; the routing-layer prefix has
        # to come off before the request goes out.
        ollama_name = model[len(OLLAMA_PREFIX) :] if model.startswith(OLLAMA_PREFIX) else model
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
