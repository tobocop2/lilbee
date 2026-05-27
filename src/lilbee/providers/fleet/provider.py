"""FleetProvider: the local llama-server engine for every role.

On first use it plans GPU placement and spawns one llama-server per configured
role (chat/embed/rerank/vision), then routes each call to the least-busy healthy
server for that role. A single machine is a fleet-of-one; there is no in-process
fallback, so a missing or unhealthy server surfaces a user-facing
``ProviderError``. Model management (list/show/capabilities) reads the registry
and GGUF headers directly and needs no running server.
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
from lilbee.providers.fleet.binary import resolve_llama_server_binary
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.fleet import Fleet
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

# User-facing name for this engine in error messages.
_PROVIDER_NAME = "llama-server"
# Jinja chat templates flag tool support by referencing one of these names as an
# identifier inside a ``{% ... %}`` / ``{{ ... }}`` block (not free-text prose).
# The server parses tool calls natively via ``--jinja``; this probe only decides
# whether to offer tools to a given model at all.
_TOOL_TEMPLATE_PATTERN = re.compile(r"\{[%{][^}]*\b(?:tools|tool_calls|functions|function_calls)\b")


def _least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Pick the healthy client with the fewest in-flight requests."""
    return min(clients, key=lambda c: c.in_flight)


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
    """Run a vision chat on *client*, enforcing *timeout* like the in-process OCR."""
    from lilbee.providers.base import ProviderError

    if timeout and timeout > 0:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.chat, messages, stream=False).result(timeout=timeout)
    else:
        result = client.chat(messages, stream=False)
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
        self._fleet: Fleet | None = None
        # Single-flight guard: the HTTP/MCP servers route concurrently, so two
        # first-requests must not each build a fleet (double GPU allocation) or
        # tear one down mid-route. Reentrant: invalidate_load_cache nests calls.
        self._lock = threading.RLock()
        # Serializes the slow fleet build (GPU probe + GGUF metadata + spawn) across
        # concurrent callers -- the off-thread warm-up and an on-demand build must
        # not run build_fleet at once (double GPU alloc + concurrent GGUF parsing
        # thrash). Held only during the build, NOT while routing, so role_ready and
        # existing-fleet routing stay responsive during a cold start.
        self._build_lock = threading.Lock()
        # Spawn-lifecycle listeners (set by the TUI via add_spawn_listener). Stored
        # so they survive a fleet rebuild and attach to every fleet we construct.
        self._on_spawning: Callable[[WorkerRole], None] | None = None
        self._on_spawned: Callable[[WorkerRole], None] | None = None
        # Single-flight guard for the off-thread warm-up: True from the moment a
        # build thread is dispatched until it finishes, so a second warm_up_pool
        # (re-entry, or a call landing during the spawn) never starts a second
        # build and double-allocates GPU memory.
        self._warming = False

    def _server_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        with self._lock:
            fleet = self._fleet
        if fleet is None:
            fleet = self._build_fleet_once()
        return fleet.healthy_clients(role)

    def _build_fleet_once(self) -> Fleet:
        """Build the fleet exactly once across concurrent callers; return it.

        The build runs under ``_build_lock`` (not the routing lock), so the
        off-thread warm-up and an on-demand call can't run ``build_fleet``
        concurrently -- which would double-allocate GPU and make two threads
        parse the same GGUF metadata at once. A second caller blocks on the
        build lock and reuses the fleet the first one built.
        """
        with self._build_lock:
            with self._lock:
                if self._fleet is not None:
                    return self._fleet
            fleet = planning.build_fleet(self._on_spawning, self._on_spawned)
            with self._lock:
                self._fleet = fleet
            return fleet

    def _require_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        """Healthy clients for *role*, or a user-facing error when none are up.

        A configured role always gets a server; an empty result means the role
        is unconfigured or its server failed to start. There is no in-process
        fallback, so this is a hard error.
        """
        from lilbee.providers.base import ProviderError

        clients = self._server_clients(role)
        if not clients:
            raise ProviderError(
                f"No {role.value} model server is running. Make sure a {role.value} "
                "model is installed and configured, then try again.",
                provider=_PROVIDER_NAME,
            )
        return clients

    def role_ready(self, role: WorkerRole) -> bool:
        """Whether *role* has a healthy server right now, without building the fleet.

        A read-only probe for surfaces (HTTP status, SSE warming event) that want
        to report cold-start state without triggering a spawn. False while the
        fleet is still warming up or the role's server is mid-(re)start.
        """
        with self._lock:
            if self._fleet is None:
                return False
            return bool(self._fleet.healthy_clients(role))

    def _shutdown_fleet(self) -> None:
        with self._lock:
            if self._fleet is not None:
                self._fleet.shutdown()
                self._fleet = None

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
        pages: list[PageText] = []
        for idx, png_bytes in rasterize_pdf(path):
            messages = build_vision_messages(OCR_PROMPT, bytes(png_bytes))
            remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else None
            text = _vision_call(_least_in_flight(clients), messages, remaining)
            page_no = idx + 1
            pages.append(PageText(page_no, text))
            if on_progress is not None:
                on_progress(
                    EventType.EXTRACT,
                    ExtractEvent(file=path.name, page=page_no, total_pages=total),
                )
        return pages

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
        """Spawn the configured role servers off the caller's thread (idempotent).

        Building the fleet loads every role's model (seconds on a cold large
        model), so it runs on a background thread and this returns at once: the
        eager-start at TUI mount must not freeze the UI. The spawn listeners
        still fire during the build, so the UI shows per-role progress. A second
        call while a build is in flight (or once the fleet is up) is a no-op.
        """
        with self._lock:
            if self._fleet is not None or self._warming:
                return
            self._warming = True
        threading.Thread(
            target=self._warm_up_blocking,
            name="fleet-warm-up",
            daemon=True,
        ).start()

    def _warm_up_blocking(self) -> None:
        """Build the fleet on a background thread; clears the warming guard when done.

        Runs on a daemon thread with no caller to catch failures, so a build
        error is logged and swallowed: a role that can't spawn surfaces a
        user-facing ProviderError on the next call, not a thread traceback.
        """
        try:
            self._build_fleet_once()
        except Exception:
            log.warning("Fleet warm-up failed; roles will spawn on first use.", exc_info=True)
        finally:
            with self._lock:
                self._warming = False

    def cancel_inference(self) -> None:
        """No-op: a llama-server stops generating when its client disconnects.

        The caller (the TUI chat worker) triggers that disconnect by closing the
        active stream, so there is no in-process abort flag to flip here.
        """
        return

    def reload_role(self, role: WorkerRole) -> None:
        """Respawn just *role*'s server(s) with current cfg; other roles keep running.

        Dispatched to a background thread because the slow respawn (stop + spawn +
        wait-ready) must not block the settings/model-picker callback that calls
        this. If the fleet isn't built yet, the next use builds it with current cfg.
        """
        with self._lock:
            if self._fleet is None:
                return
        threading.Thread(
            target=self._reload_role_blocking,
            args=(role,),
            name=f"fleet-reload-{role.value}",
            daemon=True,
        ).start()

    def _reload_role_blocking(self, role: WorkerRole) -> None:
        """Re-plan and respawn one role's server(s); runs off the caller's thread."""
        binary = resolve_llama_server_binary()
        devices = planning.resolve_devices(binary)
        by_index = {d.index: d for d in devices}
        launches = planning.plan_launches((role,), binary, by_index, devices)
        with self._lock:
            fleet = self._fleet
        if fleet is not None:
            fleet.restart_role(role, launches)

    def add_spawn_listener(
        self,
        *,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        """Store spawn-lifecycle callbacks and attach them to the running fleet."""
        with self._lock:
            self._on_spawning = on_spawning
            self._on_spawned = on_spawned
            if self._fleet is not None:
                self._fleet.set_listener(on_spawning=on_spawning, on_spawned=on_spawned)

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        """A model or settings change respawns the affected servers: drop the fleet."""
        del model_path  # the whole fleet respawns on next use; no per-model scope.
        self._shutdown_fleet()

    def drop_loaded_models_async(self) -> None:
        """Drop the whole fleet off the caller's thread; next use rebuilds with current cfg.

        ``_shutdown_fleet`` stops every server and waits on each process group,
        so a role-agnostic load-key change (num_ctx, kv_cache_type) routes here
        rather than blocking the settings callback. A no-op when no fleet is up.
        """
        with self._lock:
            if self._fleet is None:
                return
        threading.Thread(
            target=self._shutdown_fleet,
            name="fleet-drop",
            daemon=True,
        ).start()

    def shutdown(self) -> None:
        self._shutdown_fleet()
