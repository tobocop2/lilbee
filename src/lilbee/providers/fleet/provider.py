"""FleetProvider: the local llama-server engine for every role.

On first use it plans GPU placement and starts one llama-swap process that fronts
a llama-server per configured role (chat/embed/rerank/vision) co-resident behind a
single OpenAI endpoint; each call routes to that endpoint by role id. There is no
in-process fallback, so a missing role surfaces a user-facing ``ProviderError``.
Model management (list/show/capabilities) reads the registry and GGUF headers
directly and needs no running server.
"""

from __future__ import annotations

import functools
import logging
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from lilbee.providers.fleet import planning
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.swap_manager import SwapManager
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lilbee.providers.base import (
        ChatMessage,
        ChatResult,
        ChatStreamItem,
        ChatToolResult,
        ClosableIterator,
        OcrBackend,
        PageText,
    )
    from lilbee.providers.fleet.fleet import InstanceLaunch

# User-facing name for this engine in error messages.
_PROVIDER_NAME = "llama-server"
# Minimal input used to pre-load a role's upstream during warm-up (llama-swap
# starts an upstream on its first request, so warming issues one cheap call).
_WARM_PROMPT = "warm"
_WARM_MAX_TOKENS = 1
# Jinja chat templates flag tool support by referencing one of these names as an
# identifier inside a ``{% ... %}`` / ``{{ ... }}`` block (not free-text prose).
# The server parses tool calls natively via ``--jinja``; this probe only decides
# whether to offer tools to a given model at all.
_TOOL_TEMPLATE_PATTERN = re.compile(r"\{[%{][^}]*\b(?:tools|tool_calls|functions|function_calls)\b")


def _least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Pick the healthy client with the fewest in-flight requests."""
    return min(clients, key=lambda c: c.in_flight)


def _warm_role(role: WorkerRole, client: LlamaServerClient) -> None:
    """Send the cheapest request that loads *role*'s upstream behind llama-swap.

    Vision is skipped (its load is heavy and it warms on the first OCR); chat,
    embed, and rerank each issue a minimal call to trigger the upstream start.
    """
    if role is WorkerRole.CHAT:
        client.chat(
            [{"role": "user", "content": _WARM_PROMPT}],
            options={"max_tokens": _WARM_MAX_TOKENS},
            stream=False,
        )
    elif role is WorkerRole.EMBED:
        client.embed([_WARM_PROMPT])
    elif role is WorkerRole.RERANK:
        client.rerank(_WARM_PROMPT, [_WARM_PROMPT])


@functools.lru_cache(maxsize=32)
def _supports_tools_cached(path_str: str, _mtime_ns: int) -> bool:
    """Memoised tool-template probe keyed on the GGUF's path + mtime.

    The mtime arg participates in the cache key only; a re-quantised file at the
    same path invalidates automatically because its mtime changes.
    """
    from lilbee.providers.gguf_meta import read_gguf_metadata

    meta = read_gguf_metadata(Path(path_str))
    if not isinstance(meta, dict):
        return False
    template = meta.get("chat_template")
    if not isinstance(template, str):
        return False
    return _TOOL_TEMPLATE_PATTERN.search(template) is not None


def _vision_call(
    client: LlamaServerClient, messages: Sequence[Mapping[str, Any]], timeout: float | None
) -> str:
    """Run a vision chat on *client*, enforcing *timeout* like the in-process OCR.

    Caps generation at ``cfg.vision_ocr_max_tokens`` so a runaway repetition loop
    on one page (seen looping to tens of thousands of chars) can't dominate a
    scan's OCR time; a real page stays well under the cap.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.base import ProviderError

    options = {"max_tokens": cfg.vision_ocr_max_tokens}
    if timeout and timeout > 0:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.chat, messages, options=options, stream=False).result(
                timeout=timeout
            )
    else:
        result = client.chat(messages, options=options, stream=False)
    if not isinstance(result, str):
        raise ProviderError(
            f"Vision server returned {type(result).__name__}, expected text.",
            provider=_PROVIDER_NAME,
        )
    return result


def _pdf_drain_budget(total_pages: int, per_page_timeout_s: float | None) -> float | None:
    """Total OCR wall-clock budget = pages*per_page + load grace, or None for no cap.

    Mirrors the in-process drain budget: one document-wide deadline rather than a
    per-page cap, so a slow page borrows from fast ones and the vision model's cold
    first-inference is absorbed by the grace instead of tripping a fixed page limit.
    """
    from lilbee.core.config import cfg

    if not per_page_timeout_s or per_page_timeout_s <= 0:
        return None
    return total_pages * per_page_timeout_s + cfg.vision_load_budget_s


class FleetProvider:
    """Routes every role to the managed llama-server fleet (a fleet-of-one on one box)."""

    def __init__(self) -> None:
        self._swap: SwapManager | None = None
        # One OpenAI client per placed role, all pointed at the llama-swap endpoint
        # and routed by model id; rebuilt whenever the swap process (re)starts.
        self._clients: dict[WorkerRole, LlamaServerClient] = {}
        # Chat batching slots and per-slot context from the chat launch, surfaced to
        # the concurrency gate and clients; defaults until the swap is up.
        self._chat_slots = 1
        self._chat_ctx: int | None = None
        # Single-flight guard: the HTTP/MCP servers route concurrently, so two
        # first-requests must not each start a swap (double GPU allocation) or
        # tear one down mid-route. Reentrant: invalidate_load_cache nests calls.
        self._lock = threading.RLock()
        # Serializes the slow startup (GPU probe + GGUF parse + llama-swap spawn)
        # across concurrent callers, so the off-thread warm-up and an on-demand call
        # can't start two swaps. Held only during startup, NOT while routing.
        self._build_lock = threading.Lock()
        # Spawn-lifecycle listeners (set by the TUI via add_spawn_listener). Stored
        # so warm-up can report per-role progress as it pre-loads each upstream.
        self._on_spawning: Callable[[WorkerRole], None] | None = None
        self._on_spawned: Callable[[WorkerRole], None] | None = None
        # Single-flight guard for the off-thread warm-up: True from the moment a
        # warm thread is dispatched until it finishes, so a second warm_up_pool
        # never starts a second swap and double-allocates GPU memory.
        self._warming = False

    def _ensure_swap(self) -> SwapManager:
        """Start the llama-swap process exactly once across concurrent callers.

        The startup runs under ``_build_lock`` (not the routing lock), so the
        off-thread warm-up and an on-demand call can't start two swaps -- which
        would double-allocate GPU and parse the same GGUF twice. A second caller
        blocks on the build lock and reuses the swap the first one started.
        """
        with self._lock:
            if self._swap is not None:
                return self._swap
        with self._build_lock:
            with self._lock:
                if self._swap is not None:
                    return self._swap
            from lilbee.core.config import cfg

            launches = planning.plan_all_launches()
            swap = SwapManager(cfg.data_dir)
            swap.start(launches)
            with self._lock:
                self._adopt_swap(swap, launches)
            return swap

    def _adopt_swap(self, swap: SwapManager, launches: list[InstanceLaunch]) -> None:
        """Record a freshly started swap and build a client per placed role.

        Caller holds ``self._lock``. ``launches`` carries each role's slots/ctx so
        the chat capacity and served context come from the launch, not a probe.
        """
        self._swap = swap
        endpoint = swap.endpoint()
        self._clients = {
            launch.role: LlamaServerClient(endpoint, launch.role.value) for launch in launches
        }
        chat = next((launch for launch in launches if launch.role is WorkerRole.CHAT), None)
        self._chat_slots = chat.slots if chat is not None else 1
        self._chat_ctx = chat.ctx if chat is not None else None

    def _require_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        """The client for *role*, or a user-facing error when it has no server.

        A configured, placeable role gets a client; its absence means the role is
        unconfigured or did not fit memory. llama-swap loads the upstream on the
        first request, so a returned client may still be cold. No in-process
        fallback, so a missing client is a hard error.
        """
        from lilbee.providers.base import ProviderError

        self._ensure_swap()
        with self._lock:
            client = self._clients.get(role)
        if client is None:
            raise ProviderError(
                f"No {role.value} model server is running. Make sure a {role.value} "
                "model is installed and configured, then try again.",
                provider=_PROVIDER_NAME,
            )
        return [client]

    def role_ready(self, role: WorkerRole) -> bool:
        """Whether *role*'s upstream is loaded and ready, without starting the swap.

        A read-only probe for surfaces (HTTP status, SSE warming event) that want
        to report cold-start state without triggering a load. False before the swap
        is up or while the role's upstream is still loading.
        """
        with self._lock:
            swap = self._swap
        return swap is not None and swap.role_ready(role)

    def max_concurrent_chats(self) -> int:
        """The chat server's batching-slot capacity, so the gate admits that many.

        Falls back to ``1`` before the swap is up, so chat is serialized until the
        slot count is known (the launcher warms the engine before a client
        connects, so the real capacity is in effect by the first chat).
        """
        with self._lock:
            if self._swap is None:
                return 1
            return self._chat_slots

    def served_chat_ctx(self) -> int | None:
        """Per-slot context the chat server runs with, or None if not up."""
        with self._lock:
            return self._chat_ctx if self._swap is not None else None

    def _shutdown_swap(self) -> None:
        with self._lock:
            swap = self._swap
            clients = list(self._clients.values())
            self._swap = None
            self._clients = {}
            self._chat_slots = 1
            self._chat_ctx = None
        for client in clients:
            client.close()
        if swap is not None:
            swap.shutdown()

    def _require_configured_model(self, model: str | None, configured: str, role: str) -> None:
        """Reject a per-call model that differs from the server's configured one.

        The fleet serves the configured model for each role; switching models is
        a config change that respawns the server (via ``invalidate_load_cache``),
        not a per-call override. An empty/None ``model`` means "use the configured
        one" and is always accepted.
        """
        if model and model != configured:
            from lilbee.providers.base import ProviderError

            raise ProviderError(
                f"This engine serves the configured {role} model ({configured}). "
                f"To use {model!r}, set it as the {role} model and reload.",
                provider=_PROVIDER_NAME,
            )

    @overload
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: Literal[False] = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult: ...

    @overload
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: Literal[True],
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ClosableIterator[ChatStreamItem]: ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult | ClosableIterator[ChatStreamItem]:
        """Route a chat turn to the least-busy chat server.

        Non-streaming returns a :class:`ChatResult` (text, tool calls, finish
        reason); streaming yields :data:`ChatStreamItem` frames. ``--jinja`` on
        the server parses native tool calls, so tool support needs no per-family
        parser here.
        """
        from lilbee.core.config import cfg
        from lilbee.providers.engine_params import chat_options_to_kwargs

        self._require_configured_model(model, str(cfg.chat_model), "chat")
        client = _least_in_flight(self._require_clients(WorkerRole.CHAT))
        # Translate options exactly as the in-process path did (validate via
        # LLMOptions, num_predict -> max_tokens, drop num_ctx) so the server
        # honors the same generation settings; a raw passthrough would drop
        # num_predict and leak the load-only num_ctx.
        server_options = chat_options_to_kwargs(options) or None
        if stream:
            # generator satisfies ClosableIterator; close() releases the request.
            return client.chat_stream_items(  # type: ignore[return-value]
                messages, tools=tools, tool_choice=tool_choice, options=server_options
            )
        return client.chat_result(
            messages, tools=tools, tool_choice=tool_choice, options=server_options
        )

    def chat_with_tools(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ChatToolResult:
        """Route a tool-enabled chat turn to the least-busy chat server."""
        from lilbee.core.config import cfg
        from lilbee.providers.engine_params import chat_options_to_kwargs

        self._require_configured_model(model, str(cfg.chat_model), "chat")
        clients = self._require_clients(WorkerRole.CHAT)
        server_options = chat_options_to_kwargs(options) or None
        return _least_in_flight(clients).chat_tools(
            messages, tools=tools, tool_choice=tool_choice, options=server_options
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _least_in_flight(self._require_clients(WorkerRole.EMBED)).embed(texts)

    def vision_ocr(
        self, png_bytes: bytes, model: str, prompt: str = "", *, timeout: float | None = None
    ) -> str:
        from lilbee.core.config import cfg
        from lilbee.vision import OCR_PROMPT, build_vision_messages

        self._require_configured_model(model, str(cfg.vision_model), "vision")
        clients = self._require_clients(WorkerRole.VISION)
        messages = build_vision_messages(prompt or OCR_PROMPT, png_bytes)
        return _vision_call(_least_in_flight(clients), messages, timeout)

    def pdf_ocr(
        self,
        path: Path,
        *,
        backend: OcrBackend,
        model: str = "",
        per_page_timeout_s: float | None = None,
        quiet: bool = True,
        on_progress: Callable[..., None] | None = None,
    ) -> list[PageText]:
        """OCR each rasterized PDF page through the vision server.

        ``backend`` is ``Literal["vision"]`` (tesseract is run inline by the
        ingest caller, never here). ``per_page_timeout_s`` caps each page's
        request; ``quiet`` is accepted for protocol parity (the server emits no
        Rich progress to suppress). Pages are numbered 1-based to match
        ``PageText`` / ``ExtractEvent`` everywhere else in lilbee.
        """
        from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

        from lilbee.core.config import cfg
        from lilbee.runtime.progress import EventType, ExtractEvent
        from lilbee.vision import (
            OCR_PROMPT,
            PageText,
            build_vision_messages,
            pdf_page_count,
            rasterize_pdf,
        )

        del quiet  # protocol parity; no server-side Rich progress to suppress.
        self._require_configured_model(model, str(cfg.vision_model), "vision")
        clients = self._require_clients(WorkerRole.VISION)
        total = pdf_page_count(path)
        # One document-wide deadline (pages*per_page + load grace), not a per-page
        # cap: each page gets whatever budget remains, so a slow page borrows from
        # fast ones and the cold first-inference is covered, matching in-process OCR.
        budget = _pdf_drain_budget(total, per_page_timeout_s)
        deadline = (time.monotonic() + budget) if budget is not None else None

        def _ocr(idx: int, png: bytes) -> tuple[int, str]:
            messages = build_vision_messages(OCR_PROMPT, png)
            remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else None
            return idx, _vision_call(_least_in_flight(clients), messages, remaining)

        # OCR pages concurrently (a single-page decode underuses the GPU; the vision
        # server runs cfg.vision_ocr_concurrency batching slots). A bounded sliding
        # window keeps that many pages in flight without rasterizing the whole PDF
        # into memory; results are reassembled in page order.
        concurrency = max(1, cfg.vision_ocr_concurrency)
        raster = rasterize_pdf(path)
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending: set[Future[tuple[int, str]]] = set()

            def _submit_next() -> bool:
                page = next(raster, None)
                if page is None:
                    return False
                idx, png_bytes = page
                pending.add(pool.submit(_ocr, idx, bytes(png_bytes)))
                return True

            for _ in range(concurrency):
                if not _submit_next():
                    break
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for done in completed:
                    page_idx, text = done.result()
                    results[page_idx] = text
                    if on_progress is not None:
                        on_progress(
                            EventType.EXTRACT,
                            ExtractEvent(file=path.name, page=page_idx + 1, total_pages=total),
                        )
                    _submit_next()
        return [PageText(idx + 1, results[idx]) for idx in sorted(results)]

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        return _least_in_flight(self._require_clients(WorkerRole.RERANK)).rerank(query, candidates)

    # --- model management: registry / GGUF reads, no running server needed ---

    def supports_rerank(self) -> bool:
        """llama-server can always rerank a cross-encoder GGUF via ``--pooling rank``."""
        return True

    def list_models(self) -> list[str]:
        """List installed models from the registry."""
        from lilbee.app.services import get_services

        registry = get_services().registry
        return sorted(m.ref for m in registry.list_installed())

    def list_chat_models(self, provider: str) -> list[str]:
        """The local engine has no frontier-provider catalog; always ``[]``."""
        del provider
        return []

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        """Not supported directly: ``lilbee.catalog`` handles GGUF downloads."""
        del on_progress
        raise NotImplementedError(
            f"The local engine cannot pull model {model!r}. "
            "Download GGUF files through the catalog or 'lilbee model pull'."
        )

    def show_model(self, model: str) -> dict[str, Any] | None:
        """Return model metadata from GGUF headers, or ``None`` if unresolved."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.engine_params import resolve_model_path
        from lilbee.providers.gguf_meta import read_gguf_metadata

        try:
            path = resolve_model_path(model)
        except ProviderError:
            return None
        return read_gguf_metadata(path)

    def get_capabilities(self, model: str) -> list[str]:
        """Detect capabilities from the local GGUF files.

        Cross-encoder rerank GGUFs report ``["rerank"]`` (they cannot generate);
        other models report ``"completion"`` plus ``"vision"`` when an mmproj
        sidecar is present.
        """
        from lilbee.catalog import is_rerank_ref
        from lilbee.providers.base import ProviderError
        from lilbee.providers.engine_params import resolve_model_path
        from lilbee.providers.gguf_meta import find_mmproj_for_model

        if model and is_rerank_ref(model):
            return ["rerank"]
        caps = ["completion"]
        try:
            path = resolve_model_path(model)
        except ProviderError:
            return caps
        try:
            find_mmproj_for_model(path)
            caps.append("vision")
        except ProviderError:
            pass
        return caps

    def supports_tools(self, model_ref: str) -> bool:
        """True iff *model_ref*'s GGUF chat template references tool tokens.

        The server parses native tool calls via ``--jinja``; a template that
        declares tools is the signal that the model was trained to emit them.
        Cached on ``(path, mtime)`` so a tool-bearing chat doesn't re-read the
        GGUF header each request; a re-quantised file at the same path
        invalidates because its mtime changes.
        """
        from lilbee.providers.base import ProviderError
        from lilbee.providers.engine_params import resolve_model_path

        try:
            path = resolve_model_path(model_ref)
        except (ProviderError, OSError):
            log.debug("supports_tools: resolve_model_path failed for %s", model_ref, exc_info=True)
            return False
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        return _supports_tools_cached(str(path), mtime_ns)

    def warm_up_pool(self) -> None:
        """Pre-load every configured role off the caller's thread (idempotent).

        Starting the swap and loading each role's model (seconds on a cold large
        model) runs on a background thread and this returns at once: the eager-start
        at TUI mount must not freeze the UI. The spawn listeners fire per role as it
        loads, so the UI shows progress. A second call while warm-up is in flight
        (or once the swap is up) is a no-op.
        """
        with self._lock:
            if self._swap is not None or self._warming:
                return
            self._warming = True
        threading.Thread(
            target=self._warm_up_blocking,
            name="fleet-warm-up",
            daemon=True,
        ).start()

    def _warm_up_blocking(self) -> None:
        """Start the swap and pre-load every role on a background thread.

        Runs on a daemon thread with no caller to catch failures, so a startup
        error is logged and swallowed: a role that can't load surfaces a
        user-facing ProviderError on the next call, not a thread traceback.
        """
        try:
            self._ensure_swap()
            self._preload_roles()
        except Exception:
            log.warning("Engine warm-up failed; roles will load on first use.", exc_info=True)
        finally:
            with self._lock:
                self._warming = False

    def _preload_roles(self) -> None:
        """Issue one cheap request per role so llama-swap loads its upstream now.

        llama-swap starts an upstream on its first request, so warming sends a
        minimal call per role (firing the spawn listeners around each). A per-role
        failure is logged and skipped; the role still loads on its first real use.
        """
        with self._lock:
            clients = dict(self._clients)
            on_spawning, on_spawned = self._on_spawning, self._on_spawned
        for role, client in clients.items():
            if on_spawning is not None:
                on_spawning(role)
            try:
                _warm_role(role, client)
            except Exception:
                log.debug("Warm-up request for %s failed.", role.value, exc_info=True)
            if on_spawned is not None:
                on_spawned(role)

    def cancel_inference(self) -> None:
        """No-op: a llama-server stops generating when its client disconnects.

        The caller (the TUI chat worker) triggers that disconnect by closing the
        active stream, so there is no in-process abort flag to flip here.
        """
        return

    def reload_role(self, role: WorkerRole) -> None:
        """Apply a model/settings change for *role* with current cfg.

        Dispatched to a background thread because the slow restart (rewrite config +
        respawn + wait-ready) must not block the settings/model-picker callback.
        llama-swap reloads the whole proxy, so every role is re-planned. If the swap
        isn't up yet, the next use starts it with current cfg.
        """
        with self._lock:
            if self._swap is None:
                return
        threading.Thread(
            target=self._reload_blocking,
            name=f"fleet-reload-{role.value}",
            daemon=True,
        ).start()

    def _reload_blocking(self) -> None:
        """Re-plan all roles and restart llama-swap; runs off the caller's thread."""
        launches = planning.plan_all_launches()
        with self._lock:
            swap = self._swap
        if swap is None:
            return
        swap.reload(launches)
        with self._lock:
            self._adopt_swap(swap, launches)

    def add_spawn_listener(
        self,
        *,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        """Store spawn-lifecycle callbacks; warm-up fires them as each role loads."""
        with self._lock:
            self._on_spawning = on_spawning
            self._on_spawned = on_spawned

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """A model or settings change restarts the engine: drop the swap."""
        del model_path  # the whole engine restarts on next use; no per-model scope.
        self._shutdown_swap()

    def drop_loaded_models_async(self) -> None:
        """Drop the swap off the caller's thread; next use restarts with current cfg.

        ``_shutdown_swap`` stops llama-swap and waits on its process group, so a
        role-agnostic load-key change (num_ctx, kv_cache_type) routes here rather
        than blocking the settings callback. A no-op when no swap is up.
        """
        with self._lock:
            if self._swap is None:
                return
        threading.Thread(
            target=self._shutdown_swap,
            name="fleet-drop",
            daemon=True,
        ).start()

    def shutdown(self) -> None:
        self._shutdown_swap()
