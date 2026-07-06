"""FleetProvider: the local llama-server engine for every role.

On first use it plans GPU placement and starts one llama-swap process per
configured role (chat/embed/rerank/vision), each fronting that role's
llama-server(s); each call routes to its role's proxy by replica model id.
Per-role processes let a reload restart only the roles whose launches changed,
so a placement or model change never unloads an untouched role's model. There
is no in-process fallback, so a missing role surfaces a user-facing
``ProviderError``. Model management (list/show/capabilities) reads the registry
and GGUF headers directly and needs no running server.
"""

from __future__ import annotations

import functools
import logging
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.client import (
    ChatDeadlineError,
    LlamaServerClient,
    is_connection_failure,
    retry_on_busy,
)
from lilbee.providers.fleet.replicas import (
    gpu_device_count,
    resolve_replica_count,
)
from lilbee.providers.fleet.swap_config import cold_load_timeout_s
from lilbee.providers.fleet.swap_manager import SwapManager, reap_stale, sweep_owned
from lilbee.providers.fleet.windowing import window_messages
from lilbee.providers.roles import WorkerRole, configured_model_message
from lilbee.providers.warm_progress import WarmProgress, WarmProgressTracker

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lilbee.providers.base import (
        ChatMessage,
        ChatResult,
        ChatStreamItem,
        ChatToolResult,
        ClosableIterator,
        OcrBackend,
        PageText,
    )
    from lilbee.providers.fleet.launch import InstanceLaunch

# User-facing name for this engine in error messages.
_PROVIDER_NAME = "llama-server"
# Tokens held back from the served context for the model's own generation when the
# request does not cap it, plus a margin for chat-template overhead and estimate drift.
_DEFAULT_GENERATION_RESERVE = 1024
_CONTEXT_WINDOW_MARGIN = 128
# Minimal input used to pre-load a role's upstream during warm-up (llama-swap
# starts an upstream on its first request, so warming issues one cheap call).
_WARM_PROMPT = "warm"
_WARM_MAX_TOKENS = 1
# Read size for paging chat shards into the page cache during warm; large enough
# to keep sequential reads efficient without holding much resident at once.
_PREWARM_CHUNK_BYTES = 8 * 1024 * 1024
# Shards fully paged in this boot, keyed on (path, size, mtime_ns); a fleet
# rebuild (e.g. a placement change) skips re-reading a hot cache. Module-level so
# it survives reset_services() replacing the provider instance.
_PREWARMED_SHARDS: set[tuple[str, int, int]] = set()
# Per-role client request budget: the first request covers the lazy cold load plus
# generation, so the weights-scaled cold-load budget plus the margin raises this floor.
_REQUEST_TIMEOUT_FLOOR_S = 900.0
_REQUEST_TIMEOUT_GENERATION_MARGIN_S = 120.0
# Jinja chat templates flag tool support by referencing one of these names as an
# identifier inside a ``{% ... %}`` / ``{{ ... }}`` block (not free-text prose).
# The server parses tool calls natively via ``--jinja``; this probe only decides
# whether to offer tools to a given model at all.
_TOOL_TEMPLATE_PATTERN = re.compile(r"\{[%{][^}]*\b(?:tools|tool_calls|functions|function_calls)\b")
# Ingest OCR is background work: back off and re-request a 429'd page (a cold
# vision start loading weights plus mmproj, or a transient slot-contention burst)
# rather than dropping it from the index. Capped backoff keeps the total wait
# near ~110s; a page still busy after that fails like any extraction error.
_VISION_BUSY_RETRIES = 18
_T = TypeVar("_T")


def _prewarm_key(shard: Path) -> tuple[str, int, int]:
    """The prewarm identity of *shard*: same path, size, and mtime -> same pages."""
    stat = shard.stat()
    return (str(shard), stat.st_size, stat.st_mtime_ns)


def _request_timeout_s(weights_bytes: int) -> float:
    """Per-client request budget: the floor, or the cold-load budget plus margin."""
    return max(
        _REQUEST_TIMEOUT_FLOOR_S,
        cold_load_timeout_s(weights_bytes) + _REQUEST_TIMEOUT_GENERATION_MARGIN_S,
    )


def _launches_by_role(
    launches: list[InstanceLaunch],
) -> dict[WorkerRole, tuple[InstanceLaunch, ...]]:
    """Group a plan's launches by role, replica order preserved within each role."""
    grouped: dict[WorkerRole, list[InstanceLaunch]] = {}
    for launch in launches:
        grouped.setdefault(launch.role, []).append(launch)
    return {role: tuple(role_launches) for role, role_launches in grouped.items()}


def _least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Pick the healthy client with the fewest in-flight requests.

    Falls back to the full pool when every client is marked unhealthy, so a
    fully-dead pool still gets a call (which surfaces the error and lets a
    recovered replica mark itself healthy again).
    """
    healthy = [client for client in clients if client.healthy]
    return min(healthy or clients, key=lambda c: c.in_flight)


def _call_with_failover(
    clients: list[LlamaServerClient],
    call: Callable[[LlamaServerClient], _T],
) -> _T:
    """Run *call* on the least-busy healthy client, retrying once on another replica.

    A connection-level failure marks the client unhealthy and retries once on a
    different replica; with no other replica the failure surfaces.
    """
    client = _least_in_flight(clients)
    try:
        result = call(client)
    except Exception as exc:
        if not is_connection_failure(exc):
            raise
        client.mark_unhealthy()
        return _retry_on_other_replica(clients, client, call, exc)
    client.mark_healthy()
    return result


def _retry_on_other_replica(
    clients: list[LlamaServerClient],
    failed: LlamaServerClient,
    call: Callable[[LlamaServerClient], _T],
    cause: Exception,
) -> _T:
    """Retry *call* once on a replica other than *failed*, marking its health."""
    others = [c for c in clients if c is not failed]
    if not others:
        raise _no_healthy_replica_error() from cause
    retry = _least_in_flight(others)
    try:
        retry_result = call(retry)
    except Exception as retry_exc:
        if is_connection_failure(retry_exc):
            retry.mark_unhealthy()
        raise
    retry.mark_healthy()
    return retry_result


def _no_healthy_replica_error() -> ProviderError:
    """User-facing error for a call with no healthy replica left to retry on."""
    return ProviderError(
        "The model server is not responding and no healthy replica is available. "
        "It may be restarting; try again in a moment.",
        provider=_PROVIDER_NAME,
        kind=ProviderErrorKind.CONNECTION,
    )


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


class _VisionRequestGate:
    """Process-wide cap on concurrent vision-server requests at the fleet's OCR slots.

    The ingest file fan-out runs many files at once and each file's ``pdf_ocr`` opens
    its own ``vision_ocr_concurrency`` page pool, so without a shared cap the aggregate
    over-subscribes a single-replica vision server into a 429 storm. *capacity* is
    the fleet's real slot count (see ``_vision_gate_capacity``); the semaphore is
    rebuilt to it only while the gate is idle, so a capacity change never doubles
    the live cap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capacity = 0
        self._in_flight = 0
        self._semaphore: threading.BoundedSemaphore | None = None

    @contextmanager
    def slot(self, capacity: int) -> Iterator[None]:
        """Hold one of *capacity* vision-request slots for the duration of the block.

        The acquired semaphore is captured and released by this same call, so a
        concurrent capacity change cannot release the wrong object.
        """
        sem = self._checkout(capacity)
        # _checkout incremented _in_flight; decrement on every exit path,
        # including one where acquire() raises, or a leaked count pins the gate
        # non-idle and defers every later capacity resize forever.
        try:
            sem.acquire()
            try:
                yield
            finally:
                sem.release()
        finally:
            with self._lock:
                self._in_flight -= 1

    def _checkout(self, capacity: int) -> threading.BoundedSemaphore:
        with self._lock:
            # Resize only when idle: rebuilding while old-semaphore holders are
            # in flight would briefly double the real cap, so a capacity change
            # waits for the current batch to drain.
            if self._semaphore is None or (self._capacity != capacity and self._in_flight == 0):
                self._capacity = capacity
                self._semaphore = threading.BoundedSemaphore(capacity)
            self._in_flight += 1
            return self._semaphore


_VISION_GATE = _VisionRequestGate()


def _vision_call(
    client: LlamaServerClient, messages: Sequence[Mapping[str, Any]], timeout: float | None
) -> str:
    """Run a vision chat on *client*, enforcing *timeout* like the in-process OCR.

    Caps generation at ``cfg.vision_ocr_max_tokens`` so a runaway repetition loop
    on one page (seen looping to tens of thousands of chars) can't dominate a
    scan's OCR time; a real page stays well under the cap. A timeout surfaces as
    a ``ProviderError`` so the page-level OCR caller can fail just that page.
    Callers hold ``_VISION_GATE`` so queue time isn't billed against the timeout.
    """
    from lilbee.core.config import cfg

    options = {"max_tokens": cfg.vision_ocr_max_tokens}
    if timeout and timeout > 0:
        return _bounded_vision_chat(client, messages, options, timeout)
    return client.chat(messages, options=options, stream=False)


def _bounded_vision_chat(
    client: LlamaServerClient,
    messages: Sequence[Mapping[str, Any]],
    options: dict[str, Any],
    timeout: float,
) -> str:
    """One vision chat streamed under a total *timeout*, released promptly on expiry.

    ``chat_bounded`` streams the response in this thread and closes it (freeing the
    in-flight slot) once the deadline passes, so a trickling upstream can't outlive
    the caller. Its deadline signal is re-worded as the vision OCR timeout.
    """
    try:
        return client.chat_bounded(messages, options=options, deadline_s=timeout)
    except ChatDeadlineError:
        raise ProviderError(
            f"Vision OCR timed out after {timeout:.0f}s.",
            provider=_PROVIDER_NAME,
        ) from None


def _ocr_pdf_page(
    idx: int,
    png: bytes,
    *,
    clients: list[LlamaServerClient],
    ocr_prompt: str,
    deadline: float | None,
    page_path: Path,
    gate_capacity: int,
) -> tuple[int, str]:
    """OCR one rasterized page through the gated vision server; empty text on failure.

    Acquires the gate before reading the clock so queue time isn't billed against the
    page's share of the document-wide deadline; an exhausted budget skips the page
    rather than running it un-timed. A busy fleet (429) is retried with backoff
    inside the held slot before the page is given up on.
    """
    from lilbee.vision import build_vision_messages

    messages = build_vision_messages(ocr_prompt, png)
    with _VISION_GATE.slot(gate_capacity):
        remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else None
        if remaining == 0.0:
            log.warning(
                "Vision OCR budget exhausted before page %d of %s; skipping.",
                idx + 1,
                page_path.name,
            )
            return idx, ""
        try:
            return idx, retry_on_busy(
                lambda: _call_with_failover(
                    clients, lambda client: _vision_call(client, messages, remaining)
                ),
                retries=_VISION_BUSY_RETRIES,
            )
        except ProviderError:
            # One failed/timed-out page yields empty text; siblings continue.
            log.warning(
                "Vision OCR failed for page %d of %s; skipping that page.",
                idx + 1,
                page_path.name,
                exc_info=True,
            )
            return idx, ""


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
        # One llama-swap per placed role, so restarting one role's servers (a
        # placement or per-role model change) never unloads another role's.
        self._swaps: dict[WorkerRole, SwapManager] = {}
        # The launches each running group was started with, kept so a reload can
        # diff the fresh plan against what is running and restart only the roles
        # whose launches actually changed. Launch argv is port-free (ports are
        # injected at config render), so the comparison is stable across starts.
        self._launches: dict[WorkerRole, tuple[InstanceLaunch, ...]] = {}
        # Latched once shutdown runs. A discarded provider (reset_services swaps
        # in a new one) can still have an in-flight warm-up or reload daemon
        # thread; without this latch that thread could start a llama-swap after
        # shutdown already ran, leaving a process no live provider owns.
        # _ensure_fleet checks it under the build lock so a post-shutdown build is
        # refused (the swap_manager reaper is the backstop if one slips through).
        self._shut_down = False
        # A pool of OpenAI clients per placed role (one per data-parallel replica),
        # all pointed at the llama-swap endpoint and routed by replica model id;
        # rebuilt whenever the swap process (re)starts. Requests round-robin the pool.
        self._clients: dict[WorkerRole, list[LlamaServerClient]] = {}
        # Clients retired by a reload, awaiting close. A reload's old clients may
        # still be held by an in-flight reader, so they are closed at the *next*
        # reload (by when those readers have finished) or at shutdown, never while
        # potentially in use. See _retire_clients.
        self._retiring_clients: list[LlamaServerClient] = []
        # Chat batching slots and per-slot context from the chat launch, surfaced to
        # the concurrency gate and clients; defaults until the chat group is up.
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
        # Granular cold-load progress for the chat role, streamed to a launcher so
        # the user sees real read/engine-load progress instead of a frozen spinner.
        self._warm_tracker = WarmProgressTracker()
        # Single-flight guard for the off-thread warm-up: True from the moment a
        # warm thread is dispatched until it finishes, so a second warm_up_pool
        # never starts a second swap and double-allocates GPU memory.
        self._warming = False
        # Single-flight guard for the off-thread reload: a second reload_role
        # while one is in flight sets the pending flag instead of dispatching,
        # and the in-flight thread re-runs the plan loop once per pending flag.
        self._reloading = False
        # Set when a reload arrives mid-reload: the in-flight pass may have
        # already snapshotted its plan, so the change must be re-applied.
        self._reload_pending = False
        # Notified when ``_reloading`` clears, so a ``reload_role(wait=True)`` caller
        # can block until the reload it requested (or the in-flight one that will
        # run its pending pass) has finished.
        self._reload_done = threading.Condition(self._lock)

    def _ensure_fleet(self) -> bool:
        """Start one llama-swap per placed role exactly once across concurrent callers.

        Returns whether any role group is running afterwards; ``False`` when no
        role is configured and installed (nothing to serve), leaving no process
        spawned. The startup runs under ``_build_lock`` (not the routing lock),
        so the off-thread warm-up and an on-demand call can't start two fleets --
        which would double-allocate GPU and parse the same GGUF twice. A second
        caller blocks on the build lock and reuses the groups the first one
        started. A group failing to start tears down the groups already started
        in this build, so a partial fleet never leaks past the failure.
        """
        with self._lock:
            if self._swaps:
                return True
        with self._build_lock:
            with self._lock:
                if self._swaps:
                    return True
                if self._shut_down:
                    # Provider was shut down (and likely discarded by reset_services)
                    # while this warm-up/reload thread was in flight; do not spawn a
                    # llama-swap no live provider would ever reap.
                    return False
            from lilbee.core.config import cfg

            # A dead owner's surviving llama-swap holds VRAM; reap it before the
            # snapshot so the cards are actually free for this fleet (and the
            # context sizer reads true clean-box memory).
            reap_stale(cfg.data_dir)
            try:
                # Snapshot the clean box; this plan and every later reload size
                # ctx, slots, and budgets against it (a live probe under a loaded
                # fleet would report our own residency as unavailable). Inside the
                # try: capturing resolves the engine binary, and a binary-less
                # host must serve nothing, not raise.
                planning.capture_plan_probe()
                launches = planning.plan_all_launches()
            except ProviderError:
                log.debug("Engine binary unavailable; no swap started")
                return False
            if not launches:
                return False  # no installed/configured model -> serve nothing, spawn nothing
            by_role = _launches_by_role(launches)
            started: dict[WorkerRole, SwapManager] = {}
            try:
                for role, role_launches in by_role.items():
                    swap = SwapManager(cfg.data_dir, role.value)
                    swap.start(list(role_launches))
                    started[role] = swap
            except BaseException:
                for swap in started.values():
                    swap.shutdown()
                raise
            with self._lock:
                for role, swap in started.items():
                    self._adopt_role(role, swap, list(by_role[role]))
            return True

    def _adopt_role(
        self, role: WorkerRole, swap: SwapManager, launches: list[InstanceLaunch]
    ) -> None:
        """Record *role*'s freshly started swap and build its client pool.

        Caller holds ``self._lock``. Each launch (one per replica) becomes a client
        keyed by its replica model id against this group's own proxy endpoint;
        the chat launch carries the slots/ctx so the capacity and served context
        come from the launch, not a probe.
        """
        # Retire the role's previous clients (a reload re-adopts over an existing
        # pool): closing them now would error a reader still mid-call on an old
        # client snapshot, and never closing leaks an httpx pool per replica.
        old_clients = list(self._clients.get(role, []))
        self._swaps[role] = swap
        self._launches[role] = tuple(launches)
        endpoint = swap.endpoint()
        # token_cap truncates oversize embed/rerank inputs to the per-slot context
        # (the in-process backstop); the longer timeout covers a cold upstream load.
        self._clients[role] = [
            LlamaServerClient(
                endpoint,
                launch.model_id,
                token_cap=launch.token_cap,
                timeout=_request_timeout_s(launch.weights_bytes),
                rerank_mode=launch.rerank_mode,
            )
            for launch in launches
        ]
        if role is WorkerRole.CHAT:
            chat = launches[0]
            self._chat_slots = chat.slots
            self._chat_ctx = chat.ctx
        self._retire_clients(old_clients)

    def _drop_role(self, role: WorkerRole) -> SwapManager | None:
        """Forget *role*'s swap/launches/clients; return the swap for teardown.

        Caller holds ``self._lock``. The role's clients are retired (closed at a
        later reload or shutdown, never while a reader could still hold one) and
        the chat capacity falls back to its defaults when chat itself is dropped.
        """
        swap = self._swaps.pop(role, None)
        self._launches.pop(role, None)
        self._retire_clients(self._clients.pop(role, []))
        if role is WorkerRole.CHAT:
            self._chat_slots = 1
            self._chat_ctx = None
        return swap

    def _retire_clients(self, old_clients: list[LlamaServerClient]) -> None:
        """Close the previously-retired clients, then retire *old_clients*.

        Caller holds ``self._lock``. Retired clients are never handed to new
        readers (they are out of ``self._clients``), so by this reload any reader
        that held one from a prior reload has finished; an ``in_flight == 0``
        check confirms it before close, and any still-busy client stays retired
        for the next reload. This closes idle reloaded-away pools without ever
        closing one a reader could still use. Shutdown closes whatever remains.
        """
        still_busy: list[LlamaServerClient] = []
        for client in self._retiring_clients:
            if client.in_flight == 0:
                client.close()
            else:
                still_busy.append(client)
        self._retiring_clients = still_busy + old_clients

    def _require_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        """The client pool for *role*, or a user-facing error when it has no server.

        A configured, placeable role gets one or more replica clients; their absence
        means the role is unconfigured or did not fit memory. llama-swap loads each
        upstream on its first request, so a returned client may still be cold. No
        in-process fallback, so a missing pool is a hard error.

        When the pool is empty but a swap was previously built and its process has
        since exited (detected via ``is_live()``), a one-shot rebuild is attempted
        before raising so a transient llama-swap restart recovers transparently.
        """
        self._ensure_fleet()
        with self._lock:
            clients = self._clients.get(role)
            swap = self._swaps.get(role)
        if not clients and swap is not None and not swap.is_live():
            self._rebuild_role(role)
            with self._lock:
                clients = self._clients.get(role)
        if not clients:
            raise ProviderError(
                f"No {role.value} model server is running. Make sure a {role.value} "
                "model is installed and configured, then try again.",
                provider=_PROVIDER_NAME,
            )
        return list(clients)

    def _rebuild_role(self, role: WorkerRole) -> None:
        """Restart just *role*'s dead group (new port) from a fresh plan.

        Other roles' groups keep serving; only the dead group is torn down and
        respawned. Runs the same diff-driven pass as a reload, forcing *role*
        into the restart set so an unchanged plan still replaces its dead swap.
        """
        self._reload_pass(force=frozenset((role,)))

    def role_ready(self, role: WorkerRole) -> bool:
        """Whether *role*'s upstream is loaded and ready, without starting the swap.

        A read-only probe for surfaces (HTTP status, SSE warming event) that want
        to report cold-start state without triggering a load. False before the swap
        is up or while the role's upstream is still loading.
        """
        with self._lock:
            swap = self._swaps.get(role)
        return swap is not None and swap.role_ready(role)

    def max_concurrent_chats(self) -> int:
        """The chat server's batching-slot capacity, so the gate admits that many.

        Falls back to ``1`` before the chat group is up, so chat is serialized
        until the slot count is known (the launcher warms the engine before a
        client connects, so the real capacity is in effect by the first chat).
        """
        with self._lock:
            if WorkerRole.CHAT not in self._swaps:
                return 1
            return self._chat_slots

    def served_chat_ctx(self) -> int | None:
        """Per-slot context the chat server runs with, or None if not up."""
        with self._lock:
            return self._chat_ctx if WorkerRole.CHAT in self._swaps else None

    def warm_progress(self) -> WarmProgress | None:
        """Live cold-load progress for the chat role, or None before warm begins."""
        return self._warm_tracker.snapshot()

    def _shutdown_swap(self, *, latch: bool = True) -> None:
        """Stop the engine; ``latch=False`` keeps the provider reusable.

        Terminal ``shutdown()`` latches ``_shut_down`` so a discarded provider's
        in-flight warm/reload thread can't spawn an orphan swap. The cache-drop
        paths (``invalidate_load_cache``, ``drop_loaded_models_async``) pass
        ``latch=False``: the provider is retained and the next use must rebuild
        with the current cfg.
        """
        # The build lock serializes shutdown against a concurrent reload/build:
        # both mutate self._swaps and the llama-swap processes, so an unserialized
        # loser would overwrite the winner's state and leak a live llama-swap.
        with self._build_lock:
            with self._lock:
                swaps = dict(self._swaps)
                if latch:
                    self._shut_down = True
            self._drop_swap_refs()
            for swap in swaps.values():
                swap.shutdown()
            # Always sweep even when this provider holds no tracked swaps: an
            # in-flight build may have started groups this thread never adopted,
            # and the sweep stops every llama-swap this process spawned (keyed
            # on the per-group config paths), not just tracked handles.
            if not swaps:
                from lilbee.core.config import cfg

                sweep_owned(cfg.data_dir)

    def _drop_swap_refs(self) -> None:
        """Clear every group's swap/clients and the chat capacity so the next call rebuilds."""
        with self._lock:
            # Close the live pools and any clients still awaiting retirement.
            clients = [client for pool in self._clients.values() for client in pool]
            clients.extend(self._retiring_clients)
            self._swaps = {}
            self._launches = {}
            self._clients = {}
            self._retiring_clients = []
            self._chat_slots = 1
            self._chat_ctx = None
        # Full teardown: the next build starts from a clean box, so it must
        # re-snapshot memory rather than plan against this boot's probe.
        planning.clear_plan_probe()
        for client in clients:
            client.close()

    def _drop_dead_swaps(self) -> None:
        """Drop the refs of groups whose process is gone so the next call rebuilds them.

        A no-op for groups still running (e.g. the failure was in planning), so
        a live engine is never abandoned unstopped.
        """
        with self._build_lock, self._lock:
            for role in [r for r, swap in self._swaps.items() if not swap.running]:
                self._drop_role(role)

    def _require_configured_model(
        self, model: str | None, configured: str, role: WorkerRole
    ) -> None:
        """Reject a per-call model that differs from the server's configured one.

        The fleet serves the configured model for each role; switching models is
        a config change that respawns the server (via ``invalidate_load_cache``),
        not a per-call override. An empty/None ``model`` means "use the configured
        one" and is always accepted.
        """
        if model and model != configured:
            raise ProviderError(
                configured_model_message(role, configured, model),
                provider=_PROVIDER_NAME,
                kind=ProviderErrorKind.BAD_REQUEST,
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

        self._require_configured_model(model, str(cfg.chat_model), WorkerRole.CHAT)
        clients = self._require_clients(WorkerRole.CHAT)
        messages = self._fit_chat_context(messages, tools, options, model or str(cfg.chat_model))
        client = _least_in_flight(clients)
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

        self._require_configured_model(model, str(cfg.chat_model), WorkerRole.CHAT)
        clients = self._require_clients(WorkerRole.CHAT)
        messages = self._fit_chat_context(messages, tools, options, model or str(cfg.chat_model))
        server_options = chat_options_to_kwargs(options) or None
        return _least_in_flight(clients).chat_tools(
            messages, tools=tools, tool_choice=tool_choice, options=server_options
        )

    def _fit_chat_context(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any] | None,
        model: str,
    ) -> list[ChatMessage]:
        """Drop oldest turns so the prompt fits the served context.

        Raises ``ProviderError(CONTEXT_OVERFLOW)`` when even the system messages,
        tools, and the final turn exceed the window; the chat-completions route
        maps that to a 400 ``context_length_exceeded`` rather than a 500.
        """
        # 0/None means the served context is unknown (no chat launch adopted yet);
        # a real per-slot context is always positive, so skip windowing.
        if not self._chat_ctx:
            return messages
        reserve = (options or {}).get("num_predict") or _DEFAULT_GENERATION_RESERVE
        budget = self._chat_ctx - reserve - _CONTEXT_WINDOW_MARGIN
        result = window_messages(messages, tools, budget)
        if not result.fits:
            raise ProviderError(
                f"Prompt of about {result.prompt_tokens} tokens exceeds the "
                f"{self._chat_ctx}-token context window for {model!r}. Shorten the "
                "conversation or the system prompt.",
                provider=_PROVIDER_NAME,
                kind=ProviderErrorKind.CONTEXT_OVERFLOW,
            )
        return result.messages

    def embed(self, texts: list[str]) -> list[list[float]]:
        clients = self._require_clients(WorkerRole.EMBED)
        return _call_with_failover(clients, lambda client: client.embed(texts))

    def vision_ocr(
        self, png_bytes: bytes, model: str, prompt: str = "", *, timeout: float | None = None
    ) -> str:
        from lilbee.core.config import cfg
        from lilbee.vision import build_vision_messages, resolve_ocr_prompt

        self._require_configured_model(model, str(cfg.vision_model), WorkerRole.VISION)
        clients = self._require_clients(WorkerRole.VISION)
        effective = model or str(cfg.vision_model)
        messages = build_vision_messages(prompt or resolve_ocr_prompt(effective), png_bytes)
        # Retry a busy fleet inside the held slot: backoff time must not free the
        # slot for more work, and each attempt re-picks the least-busy replica.
        with _VISION_GATE.slot(self._vision_gate_capacity()):
            return retry_on_busy(
                lambda: _call_with_failover(
                    clients, lambda client: _vision_call(client, messages, timeout)
                ),
                retries=_VISION_BUSY_RETRIES,
            )

    def _vision_gate_capacity(self) -> int:
        """The vision servers' total continuous-batching slots, the 429-free ceiling.

        Summed from the adopted launches' fitted ``--parallel`` counts, which can
        be lower than ``vision_ocr_concurrency`` when memory forced a smaller fit;
        capping at the configured ceiling instead over-subscribes the servers.
        Falls back to the configured formula when no launch snapshot exists (a
        reload can momentarily drop it between two reads).
        """
        launches = self._launches.get(WorkerRole.VISION)
        if launches:
            return max(1, sum(launch.slots for launch in launches))
        replicas = resolve_replica_count(WorkerRole.VISION, gpu_device_count())
        return max(1, replicas * cfg.vision_ocr_concurrency)

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
            PageText,
            pdf_page_count,
            rasterize_pdf,
            resolve_ocr_prompt,
        )

        del quiet  # protocol parity; no server-side Rich progress to suppress.
        self._require_configured_model(model, str(cfg.vision_model), WorkerRole.VISION)
        clients = self._require_clients(WorkerRole.VISION)
        # The model is fixed for the whole document, so resolve its prompt once.
        ocr_prompt = resolve_ocr_prompt(model or str(cfg.vision_model))
        log.debug("OCR prompt for %s -> %r", model or cfg.vision_model, ocr_prompt)
        total = pdf_page_count(path)
        # One document-wide deadline (pages*per_page + load grace), not a per-page
        # cap: each page gets whatever budget remains, so a slow page borrows from
        # fast ones and the cold first-inference is covered, matching in-process OCR.
        budget = _pdf_drain_budget(total, per_page_timeout_s)
        deadline = (time.monotonic() + budget) if budget is not None else None

        _ocr = functools.partial(
            _ocr_pdf_page,
            clients=clients,
            ocr_prompt=ocr_prompt,
            deadline=deadline,
            page_path=path,
            gate_capacity=self._vision_gate_capacity(),
        )

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
        clients = self._require_clients(WorkerRole.RERANK)
        return _call_with_failover(clients, lambda client: client.rerank(query, candidates))

    # --- model management: registry / GGUF reads, no running server needed ---

    def supports_rerank(self) -> bool:
        """Serve a cross-encoder (rank pooling) or an LLM reranker (yes/no logprob)."""
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
        (or once the fleet is up) is a no-op.
        """
        with self._lock:
            if self._swaps or self._warming:
                return
            self._warming = True
        threading.Thread(
            target=self._warm_up_blocking,
            name="fleet-warm-up",
            daemon=True,
        ).start()

    def _warm_up_blocking(self) -> None:
        """Start the fleet and pre-load every role on a background thread.

        Runs on a daemon thread with no caller to catch failures, so a startup
        error is logged and swallowed: a role that can't load surfaces a
        user-facing ProviderError on the next call, not a thread traceback.
        """
        try:
            self._ensure_fleet()
            self._preload_roles()
        except Exception as exc:
            if isinstance(exc, RuntimeError) and sys.is_finalizing():
                # A fast CLI exit can tear down the interpreter while this daemon
                # thread is still warming; pool submission then raises "cannot
                # schedule new futures after interpreter shutdown". The process is
                # leaving anyway, so drop it quietly instead of stack-tracing.
                log.debug("Engine warm-up abandoned during interpreter shutdown: %s", exc)
            else:
                # A warm-up failure is handled (roles lazy-load on first use), so
                # keep the full traceback at debug: a WARNING carrying exc_info
                # reads like a crash for a condition the next real call recovers
                # from.
                log.warning("Engine warm-up failed; roles will load on first use: %s", exc)
                log.debug("Engine warm-up failure detail.", exc_info=True)
        finally:
            with self._lock:
                self._warming = False

    def _preload_roles(self, roles: frozenset[WorkerRole] | None = None) -> None:
        """Issue a cheap request per replica so llama-swap loads each upstream now.

        llama-swap starts an upstream on its first request, so warming sends a
        minimal call to every replica of every role (firing the spawn listeners
        around each role). A per-replica failure is logged and skipped; that replica
        still loads on its first real use. The chat role routes through
        :meth:`_warm_chat_role` so a launcher gets granular progress. *roles*
        narrows the warm to just those roles (a reload warms only what restarted).

        Roles warm concurrently: chat is the long pole (a large model's load
        dominates), so the light roles load alongside it instead of before it.
        """
        with self._lock:
            pools = {
                role: list(clients)
                for role, clients in self._clients.items()
                if roles is None or role in roles
            }
            on_spawning, on_spawned = self._on_spawning, self._on_spawned

        def _warm_one(role: WorkerRole, clients: list[LlamaServerClient]) -> None:
            if on_spawning is not None:
                on_spawning(role)
            if role is WorkerRole.CHAT:
                self._warm_chat_role(clients)
            else:
                self._warm_role_clients(role, clients)
            if on_spawned is not None:
                on_spawned(role)

        if not pools:
            return
        with ThreadPoolExecutor(max_workers=len(pools), thread_name_prefix="fleet-preload") as pool:
            futures = [pool.submit(_warm_one, role, clients) for role, clients in pools.items()]
            for future in futures:
                future.result()

    def _warm_role_clients(self, role: WorkerRole, clients: list[LlamaServerClient]) -> bool:
        """Warm every replica of *role*; return whether at least one loaded."""
        warmed = False
        for client in clients:
            try:
                _warm_role(role, client)
                warmed = True
            except Exception:
                log.debug("Warm-up request for %s failed.", role.value, exc_info=True)
        return warmed

    def _warm_chat_role(self, clients: list[LlamaServerClient]) -> None:
        """Warm the chat role, driving the tracker through read -> load -> ready/fail.

        Readiness is decided by whether a warm request actually returned, not by
        re-probing llama-swap (which can transiently report empty right after a
        successful load). The terminal phase is stamped in ``finally`` so an
        unexpected error mid-warm still ends the launcher's progress stream.
        """
        self._warm_tracker.begin(str(cfg.chat_model))
        warmed = False
        try:
            self._prewarm_chat_weights()
            self._warm_tracker.loading_engine()
            warmed = self._warm_role_clients(WorkerRole.CHAT, clients)
        finally:
            if warmed:
                self._warm_tracker.ready()
            else:
                self._warm_tracker.fail("The chat model did not finish loading.")

    def _prewarm_chat_weights(self) -> None:
        """Page the chat model's GGUF shards into the OS cache, reporting byte progress.

        Reading the shards before llama-swap loads them does two things: it gives a
        true read-phase percentage for the warm tracker, and it warms the page cache
        so the engine's mmap faults hit memory (a large win on a network filesystem,
        where random mmap faults stalled cold loads). Best-effort: any failure to
        resolve or size the shards (unregistered ref, cache miss, I/O error) is
        skipped, and the model still loads on the warm request.
        """
        try:
            shards = ModelRegistry(cfg.models_dir).shard_paths(str(cfg.chat_model))
            total = sum(shard.stat().st_size for shard in shards)
        except Exception:
            log.debug("Prewarm skipped; could not resolve chat shards.", exc_info=True)
            return
        if total <= 0:
            return
        keys = [_prewarm_key(shard) for shard in shards]
        if all(key in _PREWARMED_SHARDS for key in keys):
            # Already paged in this boot (e.g. a placement rebuild); the cache is hot.
            self._warm_tracker.reading(total, total)
            return
        done = 0
        self._warm_tracker.reading(0, total)
        chunk = bytearray(_PREWARM_CHUNK_BYTES)
        for index, (shard, key) in enumerate(zip(shards, keys, strict=True)):
            detail = f"shard {index + 1}/{len(shards)}" if len(shards) > 1 else None
            try:
                with shard.open("rb", buffering=0) as handle:
                    while True:
                        read = handle.readinto(chunk)
                        if not read:
                            break
                        done += read
                        self._warm_tracker.reading(done, total, detail=detail)
                _PREWARMED_SHARDS.add(key)
            except OSError:
                # A partial/locked shard just shortens the read bar; the engine load
                # surfaces any real fault as a user-facing error on the warm request.
                log.debug("Prewarm read of %s stopped early.", shard, exc_info=True)

    def cancel_inference(self) -> None:
        """No-op: a llama-server stops generating when its client disconnects.

        The caller (the TUI chat worker) triggers that disconnect by closing the
        active stream, so there is no in-process abort flag to flip here.
        """
        return

    def reload_role(self, role: WorkerRole, *, wait: bool = False) -> None:
        """Apply a model/settings change for *role* with current cfg.

        The whole fleet is re-planned, but only the roles whose launches changed
        restart, so the other roles' loaded models stay resident (*role* names
        the change for the thread label; the diff decides what restarts).
        """
        self._dispatch_reload(f"fleet-reload-{role.value}", wait=wait)

    def reload_placement(self, *, wait: bool = False) -> None:
        """Apply a placement change with current cfg, restarting only moved roles.

        The fresh plan is diffed per role against the running fleet: a role whose
        devices (and so its launch argv) did not change keeps serving through the
        change -- moving the embedder never unloads a 100GB chat model. When no
        fleet is up, the next use plans fresh, so this returns at once.
        """
        self._dispatch_reload("fleet-reload-placement", wait=wait)

    def _dispatch_reload(self, thread_name: str, *, wait: bool) -> None:
        """Run the diff-driven reload once, off-thread unless *wait*.

        Dispatched to a background thread because the slow restart (rewrite config +
        respawn + wait-ready) must not block the settings/model-picker callback.
        If no group is up yet, the next use starts the fleet with current cfg.
        Single-flight: a reload while one is in flight sets the pending flag (the
        in-flight pass may have already snapshotted its plan), and the in-flight
        thread runs one more pass per pending flag so the change is applied, not
        dropped.

        ``wait=True`` runs the reload in the caller's thread and returns only once
        the restart (and any reload already in flight that will run the pending
        pass) has finished and the proxies are healthy again, so a caller already
        off the event loop gets a real completion signal. A restarted role's model
        still loads lazily (the reload kicks an off-thread warm). It propagates a
        reload failure as an exception.
        """
        with self._lock:
            if not self._swaps:
                return
            if self._reloading:
                self._reload_pending = True
                if wait:
                    while self._reloading:
                        self._reload_done.wait()
                return
            self._reloading = True
            self._reload_pending = False
        if wait:
            self._reload_blocking()
            return
        threading.Thread(
            target=self._reload_blocking,
            name=thread_name,
            daemon=True,
        ).start()

    def _reload_blocking(self) -> None:
        """Run reload passes until no further reload arrived mid-pass.

        A failed pass with the pending flag set still runs the pending pass (the
        fresh plan may succeed under the new cfg); only the final pass's failure
        propagates, after dropping the refs to a dead swap so the next call can
        rebuild. The pending check and the guard release happen under one lock
        acquisition, so a reload_role landing between them cannot be acknowledged
        and dropped.
        """
        while True:
            try:
                self._reload_pass()
            except BaseException:
                with self._lock:
                    rerun = self._reload_pending
                    self._reload_pending = False
                    if not rerun:
                        self._reloading = False
                        self._reload_done.notify_all()
                if rerun:
                    log.warning(
                        "Engine reload failed; retrying with the pending change.", exc_info=True
                    )
                    continue
                self._drop_dead_swaps()
                raise
            with self._lock:
                if not self._reload_pending:
                    self._reloading = False
                    self._reload_done.notify_all()
                    return
                self._reload_pending = False

    def _reload_pass(self, force: frozenset[WorkerRole] = frozenset()) -> None:
        """One re-plan from current cfg, restarting only the roles that changed.

        The fresh plan is diffed per role against the launches each running group
        was started with; a role restarts only when its launches differ (covers
        added and removed roles too), so an untouched role's loaded model stays
        resident through a placement or per-role model change. *force* adds roles
        to the restart set even when their plan is unchanged (dead-swap recovery).
        Changed groups stop before the new ones start, so the planned VRAM is
        actually free when the new servers spawn. Runs under the build lock so a
        racing shutdown/build can't interleave with the restart and leak a live
        llama-swap holding GPU memory.
        """
        from lilbee.core.config import cfg

        with self._build_lock:
            with self._lock:
                if self._shut_down:
                    # Terminal shutdown landed while this reload was queued; a
                    # rebuild here would spawn a fleet no live provider owns.
                    return
                running = set(self._swaps)
                old = dict(self._launches)
            # Reap dead owners' swaps before re-planning, same as the first build.
            reap_stale(cfg.data_dir)
            if not running:
                # Nothing loaded (a resurrect after a failed pass): the box is
                # clean, so refresh the plan snapshot like a first build would.
                planning.capture_plan_probe()
            new = _launches_by_role(planning.plan_all_launches())
            # A role restarts when its launches changed OR its running/planned
            # presence disagrees (covers a group the new plan drops or adds).
            changed = {
                role
                for role in running | set(new)
                if (role in running) != (role in new) or old.get(role, ()) != new.get(role, ())
            }
            changed |= set(force)
            # Stop phase: free the changed roles' VRAM before their replacements
            # (or another role's grown plan) spawn against it.
            for role in sorted(changed, key=lambda r: r.value):
                with self._lock:
                    swap = self._drop_role(role)
                if swap is not None:
                    swap.shutdown()
            # Start phase: spawn the changed roles present in the new plan.
            restarted: list[WorkerRole] = []
            for role in sorted(changed & set(new), key=lambda r: r.value):
                role_launches = list(new[role])
                swap = SwapManager(cfg.data_dir, role.value)
                swap.start(role_launches)
                with self._lock:
                    self._adopt_role(role, swap, role_launches)
                restarted.append(role)
        if restarted:
            # Load the restarted roles' models off-thread (llama-swap spawns an
            # upstream on its first request): the reload returns once the proxies
            # answer, and the UI's spawn listeners track the model loads.
            threading.Thread(
                target=self._preload_roles,
                kwargs={"roles": frozenset(restarted)},
                name="fleet-reload-warm",
                daemon=True,
            ).start()

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
        self._shutdown_swap(latch=False)

    def drop_loaded_models_async(self) -> None:
        """Drop the swap off the caller's thread; next use restarts with current cfg.

        ``_shutdown_swap`` stops llama-swap and waits on its process group, so a
        role-agnostic load-key change (num_ctx, kv_cache_type) routes here rather
        than blocking the settings callback. A no-op when no swap is up.
        """
        with self._lock:
            if not self._swaps:
                return
        threading.Thread(
            target=lambda: self._shutdown_swap(latch=False),
            name="fleet-drop",
            daemon=True,
        ).start()

    def shutdown(self) -> None:
        self._shutdown_swap()
