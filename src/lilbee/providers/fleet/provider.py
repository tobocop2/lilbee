"""FleetProvider: the local llama-server engine for every role.

On first use it plans GPU placement and starts one llama-swap process per swap
group, each fronting that group's llama-server(s); each call routes to its role's
proxy by replica model id. Per-group processes let a reload restart only the
groups whose launches changed, so a placement or model change never unloads an
untouched group's model. There is no in-process fallback, so a missing role
surfaces a user-facing ``ProviderError``. Model management
(list/show/capabilities) reads the registry and GGUF headers directly and needs no
running server. See docs/architecture.md for swap tenancy.
"""

from __future__ import annotations

import functools
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, overload

import httpx

from lilbee.catalog import clean_display_name
from lilbee.core.config import cfg
from lilbee.core.vectors import Vector
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.base import (
    GENERATION_RESERVE_TOKENS,
    ProviderError,
    ProviderErrorKind,
    prompt_token_budget,
)
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.binary import engine_pin, resolve_llama_server
from lilbee.providers.fleet.client import (
    ChatDeadlineError,
    LlamaServerClient,
    is_connection_failure,
    is_load_capacity_failure,
    is_rebuildable_failure,
    retry_on_busy,
)
from lilbee.providers.fleet.contract import (
    chat_ctx_covers,
    contract_matches,
    decoded_launches,
    served_pairs,
)
from lilbee.providers.fleet.groups import SwapGroup, group_for
from lilbee.providers.fleet.ingest_warmth import ingest_keep_warm
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import cold_load_timeout_s
from lilbee.providers.fleet.swap_manager import (
    SwapManager,
    SwapState,
    engine_record_exists,
    find_live_state,
    reap_stale,
    state_is_healthy,
    stop_engine,
)
from lilbee.providers.fleet.windowing import window_messages
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import MODEL_FIELD_TO_ROLE, WorkerRole, configured_model_message
from lilbee.providers.warm_progress import WarmPhase, WarmProgress, WarmProgressTracker
from lilbee.runtime.engine_lock import (
    ENGINE_DIR_ENV,
    UserLockHold,
    build_lock,
    hold_user_lock,
    keep_warm_requested,
    kernel_arbitrates_locks,
    live_users_exist,
    machine_engine_dir,
    private_engine_dir,
    request_keep_warm,
    withdraw_keep_warm,
)

log = logging.getLogger(__name__)

# How long a shutdown waits for an in-flight build to finish before tearing down
# regardless. Generous against a legitimate llama-swap spawn (a 30 s boot budget)
# and far short of any supervisor's patience for a process that will not exit.
_SHUTDOWN_BUILD_LOCK_WAIT_S = 45.0

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from lilbee.providers.base import (
        ChatMessage,
        ChatResult,
        ChatStreamItem,
        ChatToolResult,
        ClosableIterator,
    )

# User-facing name for this engine in error messages.
_PROVIDER_NAME = "llama-server"
# Tokens held back from the served context for the model's own generation when the
# request does not cap it, plus a margin for chat-template overhead and estimate drift.
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
# Attempt cap for the busy-retry only when a page has no deadline (ocr_timeout=0,
# "no limit"): it backstops the retry so a persistently busy fleet can't spin
# forever. A page with a deadline retries until that deadline instead (see
# _ocr_dispatch), so the count doesn't bound the common case.
_VISION_BUSY_RETRIES = 18
# How often a waiter blocked on full replicas re-polls their health: an
# unhealthy replica re-admits itself by cool-down expiry, which notifies nobody.
_DISPATCH_HEALTH_RECHECK_S = 0.5
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


def _launches_by_group(
    plan: planning.FleetPlan,
) -> dict[SwapGroup, tuple[InstanceLaunch, ...]]:
    """Group a plan's launches by swap group, replica order preserved within each group.

    Co-tenant roles land in one group, so llama-swap evicts between them rather
    than holding both resident.
    """
    grouped: dict[SwapGroup, list[InstanceLaunch]] = {}
    for launch in plan.launches:
        grouped.setdefault(group_for(launch.role, plan.co_tenants), []).append(launch)
    return {group: tuple(group_launches) for group, group_launches in grouped.items()}


def _by_role(launches: list[InstanceLaunch]) -> dict[WorkerRole, list[InstanceLaunch]]:
    """Split one group's launches per role, replica order preserved."""
    grouped: dict[WorkerRole, list[InstanceLaunch]] = {}
    for launch in launches:
        grouped.setdefault(launch.role, []).append(launch)
    return grouped


def _least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Pick the healthy client with the fewest in-flight requests.

    Falls back to the full pool when every client is marked unhealthy, so a
    fully-dead pool still gets a call (which surfaces the error and lets a
    recovered replica mark itself healthy again).
    """
    healthy = [client for client in clients if client.healthy]
    return min(healthy or clients, key=lambda c: c.in_flight)


# Serializes pick-and-reserve so concurrent routers see each other's assignment.
# Held only for the O(replicas) selection, never across the request itself.
_ROUTE_LOCK = threading.Lock()


def _reserve_least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Atomically pick the least-loaded healthy client and reserve a slot on it.

    Selection and reservation are one critical section: without it, concurrent
    callers all read the same idlest replica before any of them increments its
    counter and route there together (a thundering herd that starves the rest of
    the fleet). The caller must :meth:`~LlamaServerClient.release` the slot.
    """
    with _ROUTE_LOCK:
        client = _least_in_flight(clients)
        client.reserve()
        return client


def _healthy_groups_ours(
    states: dict[SwapGroup, SwapState], pin: str, wanted: set[tuple[WorkerRole, str]]
) -> bool:
    """Whether every healthy group in *states* is pin-equal and serves only wanted pairs.

    True marks the incumbent as this contract's own engine that a full bind
    could not cover (a dead group, or config grew a role): the ladder rebuilds
    it in place even with live users, since those users need the rebuild too.
    False (a foreign pin or a model outside the contract) keeps the incumbent
    protected while in use. Vacuously False with no healthy group.
    """
    if not states:
        return False
    for state in states.values():
        if not contract_matches(state, (), pin):
            return False
        pairs = served_pairs(state)
        if pairs is None or not pairs <= wanted:
            return False
    return True


def _healthy_states(engine_dir: Path) -> dict[SwapGroup, SwapState]:
    """One probe pass over *engine_dir*: the recorded, answering group states.

    The ladder's single view of a dir. Bind eligibility and the replaceability
    check both read this snapshot, so they cannot disagree about an engine that
    died between them, and one wedged proxy port is paid for once per ladder
    pass rather than once per decision -- all of it under the build lock, which
    every other lilbee start is waiting on.
    """
    found: dict[SwapGroup, SwapState] = {}
    for group in SwapGroup:
        state = find_live_state(engine_dir, group)
        if state is not None and state_is_healthy(state):
            found[group] = state
    return found


def _bindable_group(
    state: SwapState, pin: str, wanted: set[tuple[WorkerRole, str]]
) -> tuple[SwapState, list[InstanceLaunch], set[tuple[WorkerRole, str]]] | None:
    """*state*'s launches and the wanted pairs it covers, or ``None``.

    ``None`` for every reason an already-healthy group is not bindable by us:
    a foreign pin, an undecodable contract, or serving nothing we want.
    """
    if not contract_matches(state, (), pin):
        # Pin mismatch or undecodable contract: not bindable by us.
        return None
    launches = decoded_launches(state)
    if launches is None:
        return None
    pairs = {(launch.role, launch.model) for launch in launches} & wanted
    return (state, launches, pairs) if pairs else None


class _PrimedStream:
    """A stream re-fronted with its eagerly-pulled first frame.

    close() always reaches the source stream, even before any iteration, so a
    caller that truncates immediately still releases the fleet's in-flight
    request slot (an unstarted chaining generator would silently drop it).
    """

    def __init__(self, first: ChatStreamItem, source: ClosableIterator[ChatStreamItem]) -> None:
        self._first: list[ChatStreamItem] = [first]
        self._source = source

    def __iter__(self) -> _PrimedStream:
        return self

    def __next__(self) -> ChatStreamItem:
        if self._first:
            return self._first.pop()
        return next(self._source)

    def close(self) -> None:
        self._source.close()


def _primed_stream(items: ClosableIterator[ChatStreamItem]) -> ClosableIterator[ChatStreamItem]:
    """Pull the first frame of *items* now, so a dead engine raises to the caller.

    The stream connects lazily on first iteration; without priming, a proxy
    that died raises only inside the consumer's loop, past any rediscovery.
    """
    try:
        first = next(items)
    except StopIteration:
        return items  # already exhausted; still closable
    return _PrimedStream(first, items)


def _call_with_failover(
    clients: list[LlamaServerClient],
    call: Callable[[LlamaServerClient], _T],
) -> _T:
    """Run *call* on the least-busy healthy client, retrying once on another replica.

    The client is reserved at selection so concurrent ingest threads spread
    across replicas. A connection-level failure marks the client unhealthy and
    retries once on a different replica; with no other replica the failure
    surfaces. The reservation is released once the call resolves.
    """
    client = _reserve_least_in_flight(clients)
    try:
        result = call(client)
    except Exception as exc:
        if not is_connection_failure(exc):
            raise
        client.mark_unhealthy()
        return _retry_on_other_replica(clients, client, call, exc)
    else:
        client.mark_healthy()
        return result
    finally:
        client.release()


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
    retry = _reserve_least_in_flight(others)
    try:
        retry_result = call(retry)
    except Exception as retry_exc:
        if is_connection_failure(retry_exc):
            retry.mark_unhealthy()
        raise
    else:
        retry.mark_healthy()
        return retry_result
    finally:
        retry.release()


def _no_healthy_replica_error() -> ProviderError:
    """User-facing error for a call with no healthy replica left to retry on."""
    return ProviderError(
        "The model server is not responding and no healthy replica is available. "
        "It may be restarting; try again in a moment.",
        provider=_PROVIDER_NAME,
        kind=ProviderErrorKind.CONNECTION,
    )


# Env vars a launch pins its devices with, one per backend (Metal has none).
_VISIBLE_DEVICE_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
)


def _role_device_sets(
    launches: Iterable[InstanceLaunch],
) -> dict[WorkerRole, frozenset[str]]:
    """Backend-qualified device tokens each role's launches pin, by role.

    A role's set is the union across its replicas. Roles whose launches carry
    no visibility env (Metal, or an unpinned backend) are absent: without
    pinning there is no proof of sharing, so they keep the concurrent warm.
    """
    sets: dict[WorkerRole, set[str]] = {}
    for launch in launches:
        for var in _VISIBLE_DEVICE_ENV_VARS:
            value = launch.env_overrides.get(var)
            if value:
                sets.setdefault(launch.role, set()).update(
                    f"{var}={part.strip()}" for part in value.split(",")
                )
    return {role: frozenset(tokens) for role, tokens in sets.items()}


def _warm_chains(
    warm_roles: list[WorkerRole], device_sets: dict[WorkerRole, frozenset[str]]
) -> list[list[WorkerRole]]:
    """Group *warm_roles* into chains warmed sequentially; chains run in parallel.

    Roles with overlapping device sets land in one chain, merged transitively.
    Within a chain chat goes last: it sizes its KV against the headroom the
    settled residents leave, so it must not race their loads. A role with no
    device set shares nothing provable and gets its own chain.
    """
    chains: list[tuple[set[str], list[WorkerRole]]] = []
    ordered = sorted(warm_roles, key=lambda r: (r is WorkerRole.CHAT, list(WorkerRole).index(r)))
    for role in ordered:
        tokens = device_sets.get(role)
        if not tokens:
            chains.append((set(), [role]))
            continue
        merged_tokens, merged_roles = set(tokens), [role]
        kept: list[tuple[set[str], list[WorkerRole]]] = []
        for chain_tokens, chain_roles in chains:
            if chain_tokens & merged_tokens:
                merged_tokens |= chain_tokens
                merged_roles = chain_roles + merged_roles
            else:
                kept.append((chain_tokens, chain_roles))
        kept.append((merged_tokens, merged_roles))
        chains = kept
    return [roles for _tokens, roles in chains]


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


class _VisionReplica(NamedTuple):
    """One vision server paired with its fitted ``--parallel`` slot count."""

    client: LlamaServerClient
    slots: int


class _PageBudgetExhausted(Exception):  # noqa: N818 - internal control flow, not an error API
    """A page's document-wide OCR budget ran out before its slot came up."""


class _VisionDispatcher:
    """Process-wide per-replica slot assignment for vision requests.

    The ingest file fan-out runs many OCR requests at once; each request is
    assigned one specific replica and only while that replica has a free
    continuous-batching slot, so lilbee's own traffic can never oversubscribe a
    vision server into a 429 (an aggregate cap plus racy least-busy routing
    can). Requests past capacity wait in-process until any usable replica
    frees a slot; unhealthy replicas take no new work until their half-open
    cool-down re-admits them.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._assigned: dict[LlamaServerClient, int] = {}

    @contextmanager
    def slot(self, pool: Sequence[_VisionReplica]) -> Iterator[LlamaServerClient]:
        """Hold one batching slot on the pool's best replica; yields that client."""
        client = self._acquire(pool)
        try:
            yield client
        finally:
            self._release(client)

    def _acquire(self, pool: Sequence[_VisionReplica]) -> LlamaServerClient:
        with self._cond:
            while True:
                client = self._pick(pool)
                if client is not None:
                    self._assigned[client] = self._assigned.get(client, 0) + 1
                    return client
                # The timed wait re-polls health: a replica can become routable
                # again by cool-down expiry alone, which notifies no waiter.
                self._cond.wait(timeout=_DISPATCH_HEALTH_RECHECK_S)

    def _pick(self, pool: Sequence[_VisionReplica]) -> LlamaServerClient | None:
        """The usable replica with the most free slots, or None while all are full.

        Falls back to the full pool when every replica is unhealthy (mirrors
        ``_least_in_flight``), so a dead pool surfaces the error instead of
        queueing forever.
        """
        usable = [replica for replica in pool if replica.client.healthy] or list(pool)
        best = max(usable, key=self._free_slots)
        return best.client if self._free_slots(best) > 0 else None

    def _free_slots(self, replica: _VisionReplica) -> int:
        return replica.slots - self._assigned.get(replica.client, 0)

    def _release(self, client: LlamaServerClient) -> None:
        with self._cond:
            remaining = self._assigned.get(client, 0) - 1
            if remaining <= 0:
                self._assigned.pop(client, None)
            else:
                self._assigned[client] = remaining
            self._cond.notify_all()


_VISION_DISPATCHER = _VisionDispatcher()


def _dispatch_vision(pool: Sequence[_VisionReplica], call: Callable[[LlamaServerClient], _T]) -> _T:
    """Run *call* on a replica with a free batching slot, failing over once.

    Blocks until a slot frees rather than racing requests at a full server. A
    connection-level failure marks the replica unhealthy and retries once on
    another replica's slot; with no other replica the failure surfaces.
    """
    with _VISION_DISPATCHER.slot(pool) as client:
        try:
            result = call(client)
        except Exception as exc:
            if not is_connection_failure(exc):
                raise
            client.mark_unhealthy()
            failed, cause = client, exc
        else:
            client.mark_healthy()
            return result
    others = [replica for replica in pool if replica.client is not failed]
    if not others:
        raise _no_healthy_replica_error() from cause
    with _VISION_DISPATCHER.slot(others) as retry_client:
        try:
            retry_result = call(retry_client)
        except Exception as retry_exc:
            if is_connection_failure(retry_exc):
                retry_client.mark_unhealthy()
            raise
        retry_client.mark_healthy()
        return retry_result


def _vision_call(
    client: LlamaServerClient, messages: Sequence[Mapping[str, Any]], timeout: float | None
) -> str:
    """Run a vision chat on *client*, enforcing *timeout* like the in-process OCR.

    Caps generation at ``cfg.vision_ocr_max_tokens`` so a runaway repetition loop
    on one page (seen looping to tens of thousands of chars) can't dominate a
    scan's OCR time; a real page stays well under the cap. A timeout surfaces as
    a ``ProviderError`` so the page-level OCR caller can fail just that page.
    Callers hold a dispatcher slot, so queue time isn't billed against the timeout.
    """

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


def _ocr_dispatch(
    pool: Sequence[_VisionReplica],
    messages: Sequence[Mapping[str, Any]],
    deadline: float | None,
) -> str:
    """OCR *messages* on a free replica slot, retrying transient failures until *deadline*.

    Backpressure (the dispatcher blocking until a slot frees) makes a
    self-inflicted 429 unreachable; a residual busy response is a still-warming
    server or foreign traffic, and a gateway error is a replica restarting
    mid-run. The retry is deadline-bound rather than
    attempt-bound so a page on a deep queue waits for a genuinely free slot until
    its own budget passes instead of dropping after a fixed count. Each attempt
    is bounded by the budget remaining before *deadline*; an exhausted budget
    raises :class:`_PageBudgetExhausted`. A ``None`` deadline (no limit) falls
    back to a bounded attempt count so the retry can't spin forever.
    """

    def _attempt(client: LlamaServerClient) -> str:
        remaining = max(0.0, deadline - time.monotonic()) if deadline is not None else None
        if remaining == 0.0:
            raise _PageBudgetExhausted
        return _vision_call(client, messages, remaining)

    return retry_on_busy(
        lambda: _dispatch_vision(pool, _attempt),
        retries=_VISION_BUSY_RETRIES,
        deadline=deadline,
    )


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


def _ocr_deadline(per_page_timeout_s: float | None) -> float | None:
    """Absolute monotonic deadline for one image OCR, or None when uncapped.

    An image is a one-page document, so it gets the same budget as a PDF page:
    the per-page timeout plus the cold-load grace, spanning queue wait and
    generation together.
    """
    budget = _pdf_drain_budget(1, per_page_timeout_s)
    return None if budget is None else time.monotonic() + budget


_ROLE_TO_MODEL_FIELD = {role: field for field, role in MODEL_FIELD_TO_ROLE.items()}


def _configured_model_for(role: WorkerRole) -> str:
    """The cfg model ref for *role*, empty when the role is unset."""
    field = _ROLE_TO_MODEL_FIELD.get(role)
    return getattr(cfg, field) or "" if field else ""


def _unusable_engine_reason() -> str | None:
    """Why no server can start on this host, or None once an engine resolves.

    Planning drops an engine-less host to serving nothing and says so only at
    debug, so by the time a surface has an empty pool the engine is the one cause
    it cannot see. Re-resolving here is also what keeps an engine installed
    mid-session from being reported as still missing.
    """
    try:
        resolve_llama_server()
    except ProviderError as exc:
        return str(exc)
    return None


def _chat_needs_local_engine() -> bool:
    """Whether the configured chat model is one this host has to serve itself.

    A chat ref routed to an SDK backend runs without any local engine, so a
    missing one is not its failure and must not be stamped on its warm.
    """
    ref = _configured_model_for(WorkerRole.CHAT)
    return bool(ref) and not parse_model_ref(ref).is_remote


def _no_server_message(role: WorkerRole) -> str:
    """User-facing reason *role* has no server, engine state first.

    A missing engine and a model that never placed both arrive as an empty pool,
    and reading the second onto the first sends the reader to a model
    configuration that is already correct.
    """
    reason = _unusable_engine_reason()
    if reason is not None:
        return f"No {role.value} model server is running: {reason}"
    return (
        f"No {role.value} model server is running. Make sure the {role.value} "
        "model is installed and configured, then try again."
    )


class _EngineDemand(NamedTuple):
    """What this process needs an engine to serve: pairs plus its chat window."""

    pairs: set[tuple[WorkerRole, str]]
    # Per-slot chat tokens this process needs; 0 demands nothing.
    chat_ctx: int
    # Configured roles the plan skipped because their model is not installed.
    # Carried out of the demand plan so the warm tracker can name the missing
    # model even when the ladder never reaches _plan_and_spawn (zero installed
    # models fail _can_build_engine first) or binds an existing engine.
    skipped_not_installed: dict[WorkerRole, str]
    # Launches the demand plan refused for an unusable window (role -> reason);
    # recorded even when the ladder binds an engine or never builds one.
    skipped_unusable_ctx: dict[WorkerRole, str]


def _placeable_demand() -> _EngineDemand:
    """Configured (role, model) pairs a fresh plan would serve, and the chat window.

    A configured role is wanted only when the planner would place it. The plan
    is the co-placement authority: a role that fits alone but cannot co-tenant (a
    unified-memory box past its budget) gets no launch, so it is dropped here too,
    and bind matches a running engine instead of judging it a partial cover and
    restarting the shared engine on every process start. The per-role check stays
    as the cheap gate for the reasons in its own docstring. Empty when no engine
    binary resolves: nothing is placeable, so the ladder serves nothing.
    """
    from lilbee.providers.fleet.planning import (
        placeable_total_vram,
        plan_all_launches,
        role_model_placeable,
    )

    try:
        plan = plan_all_launches()
    except ProviderError as exc:
        if exc.kind is ProviderErrorKind.NOT_FOUND:
            return _EngineDemand(set(), 0, {}, {})
        raise
    placed = {launch.role for launch in plan.launches}
    total_vram = placeable_total_vram()
    pairs = {
        (role, model)
        for role in WorkerRole
        if role in placed
        and (model := _configured_model_for(role))
        and role_model_placeable(role, model, total_vram)
    }
    return _EngineDemand(
        pairs,
        _demanded_chat_ctx(plan.launches, pairs),
        dict(plan.skipped_not_installed),
        dict(plan.skipped_unusable_ctx),
    )


def _demanded_chat_ctx(
    launches: Iterable[InstanceLaunch], pairs: set[tuple[WorkerRole, str]]
) -> int:
    """Per-slot chat window this process needs an engine to serve; 0 for none.

    The cfg target (a ``num_ctx`` pin, else ``chat_n_ctx_target``) capped by
    this process's own planned chat window: a window the plan itself cannot
    reach (model ceiling, hardware) is not a demand a rebuild could satisfy,
    so capping keeps the fit check from rebuilding the engine in a loop.

    The cap applies only to a single-device chat plan, whose window is sized
    against device totals and holds regardless of what is resident. A
    tensor-split plan is sized against live free VRAM, which a resident
    incumbent deflates; capping by it would shrink the demand to whatever the
    incumbent left free and let the fit check pass vacuously.
    """
    if not any(role is WorkerRole.CHAT for role, _model in pairs):
        return 0
    chat_launches = [launch for launch in launches if launch.role is WorkerRole.CHAT]
    planned = max((launch.ctx for launch in chat_launches), default=0)
    if planned <= 0:
        return 0
    # Always positive: num_ctx validates ge=1 and chat_n_ctx_target ge=512.
    target = cfg.num_ctx if cfg.num_ctx is not None else cfg.chat_n_ctx_target
    split = any(len(launch.est_vram_by_device) > 1 for launch in chat_launches)
    return target if split else min(target, planned)


def _can_build_engine(wanted: set[tuple[WorkerRole, str]]) -> bool:
    """Preconditions for a viable build, checked before stopping a warm engine.

    A process that can serve nothing (no placeable model, an unresolvable engine
    binary) must not stop an engine another setup left warm and then spawn nothing.
    Probing the engine here resolves the binary AND enumerates devices, so a wedged
    GPU probe or an unusable CUDA runtime raises loud at this point -- before the
    caller stops a replaceable incumbent. Were the probe left to run only inside
    ``_plan_and_spawn`` (after the stop), that raise would kill an engine other
    members still hold and then skip the overflow build, leaving zero engines. This
    takes no memory snapshot (device enumeration reads no residency); the clean-box
    sizing snapshot is captured by ``_plan_and_spawn`` after the stop.
    """
    from lilbee.providers.fleet import planning

    if not wanted:
        return False
    try:
        planning.assert_engine_probeable()
    except ProviderError as exc:
        # A genuinely-missing engine binary keeps the quiet serve-nothing path;
        # every other probe failure must propagate (fail loud) rather than be read
        # as "cannot build" and silently stand down.
        if exc.kind is not ProviderErrorKind.NOT_FOUND:
            raise
        return False
    except OSError:
        return False
    return True


def _warm_ttl_seconds(*, hold_warm_for_session: bool = False) -> int:
    """llama-swap idle-unload timer in seconds for the spawned fleet.

    A ttl of 0 keeps weights resident until the engine is stopped; otherwise an
    idle engine releases its weights after ``engine_idle_ttl_minutes`` and reloads
    transparently on the next prompt. The timer is held off (ttl 0) whenever
    someone is actively depending on an instant response: *hold_warm_for_session*
    is set for a provider serving an interactive session, which owns the process
    for its whole lifetime (close lilbee to release the engine); a bulk ingest
    holds the fleet resident for its run so an unevenly loaded replica cannot
    idle-unload and reload cold mid-run; and ``keep_engine_warm`` pins the weights
    for a process meant to stay ready.
    """
    if hold_warm_for_session or ingest_keep_warm() or cfg.keep_engine_warm:
        return 0
    return cfg.engine_idle_ttl_minutes * 60


class FleetProvider:
    """Routes every role to the managed llama-server fleet (a fleet-of-one on one box)."""

    def __init__(self, *, hold_warm: bool = False) -> None:
        # An interactive session (the TUI) owns this process for its whole
        # lifetime, so its fleet stays resident instead of idle-unloading under a
        # user who is still in the app; closing lilbee releases it. Set by the
        # container that built this provider, never mutated afterwards.
        self._hold_warm_for_session = hold_warm
        # One llama-swap per placed group, so restarting one group's servers (a
        # placement or per-role model change) never unloads another group's. A
        # co-tenant group holds chat and vision, which evict each other on load.
        self._swaps: dict[SwapGroup, SwapManager] = {}
        # The group each placed role runs in, so a role's clients and its swap
        # process can be reached from the role alone.
        self._role_group: dict[WorkerRole, SwapGroup] = {}
        # The launches each running group was started with, kept so a reload can
        # diff the fresh plan against what is running and restart only the groups
        # whose launches actually changed. Launch argv is port-free (ports are
        # injected at config render), so the comparison is stable across starts.
        self._launches: dict[SwapGroup, tuple[InstanceLaunch, ...]] = {}
        # Engine dirs this provider holds membership in (machine slot and/or
        # the private overflow), and the dir each running group lives in.
        self._engine_holds: dict[Path, UserLockHold] = {}
        self._group_dirs: dict[SwapGroup, Path] = {}
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
        # Latest chat prefill progress reported by a streaming client, cleared
        # by the same stream when generation starts or the stream ends.
        self._chat_prefill: tuple[int, int] | None = None
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
        # The engine's own reason a role's model failed to warm, so the launcher and
        # the TUI report the real cause instead of a generic "did not load".
        self._warm_errors: dict[WorkerRole, str] = {}
        # Configured roles the last plan left unplaced because their model isn't
        # installed (role -> ref). The warm finalizer reads it to fail a not-installed
        # chat with a named reason instead of clearing to a silent "not ready" retry.
        self._skipped_not_installed: dict[WorkerRole, str] = {}
        # Launches the last plan refused for an unusable window (role -> reason);
        # read by the warm finalizer and _require_clients.
        self._skipped_unusable_ctx: dict[WorkerRole, str] = {}
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

            return self._acquire_engine(cfg.data_root)

    def _acquire_engine(self, config_root: Path) -> bool:
        """The acquisition ladder: bind to a compatible engine, else build one.

        Machine slot first. An incumbent is replaced in place when no live
        user holds it, or when it is this contract's own engine (pin-equal,
        serving only wanted models) left partially dead or partially covering:
        its members are waiting for exactly that rebuild, and overflowing
        around it would load duplicate weights. Only a live incompatible
        engine in active use sends the build to the config root's private
        overflow dir. Per-dir the step is all-or-nothing: every configured
        (role, model) pair bound, or built fresh. Runs under the cross-process
        build lock, so two starts never both build and stop-if-last never
        races an arrival.
        """
        pin = engine_pin()
        demand = _placeable_demand()
        # Record the demand plan's skips before walking the ladder: a bind or an
        # early serve-nothing exit never reaches _plan_and_spawn, and the warm
        # tracker must still be able to say "chat model X is not installed"
        # rather than a retryable not-ready.
        self._skipped_not_installed = dict(demand.skipped_not_installed)
        self._skipped_unusable_ctx = dict(demand.skipped_unusable_ctx)
        machine_dir = machine_engine_dir()
        if kernel_arbitrates_locks(machine_dir):
            machine = self._acquire_in_dir(machine_dir, pin, demand, is_overflow=False)
            if machine is not None:
                return machine
        else:
            # Without kernel-arbitrated locks the membership refcount cannot be
            # trusted, and sharing is exactly what needs it: a probe would
            # destroy a live member's lock, so the slot would look free while
            # another setup is serving from it. Keep to our own dir instead.
            log.warning(
                "Engine dir %s is on a filesystem without working file locks; "
                "using a private engine instead of the shared one. Set %s to a "
                "path on a local filesystem to share one engine across lilbees.",
                machine_dir,
                ENGINE_DIR_ENV,
            )
        # The machine slot holds a live incompatible engine in active use: overflow
        # to this config root's private dir rather than evict another model setup.
        private = private_engine_dir(config_root)
        return self._acquire_in_dir(private, pin, demand, is_overflow=True) or False

    def _acquire_in_dir(
        self, engine_dir: Path, pin: str, demand: _EngineDemand, *, is_overflow: bool
    ) -> bool | None:
        """Bind or build one engine dir; ``None`` on the slot means overflow next.

        Binds a compatible running engine. Whether an incumbent may be replaced
        is decided by kernel-refcounted membership, not the proxy HTTP probe: an
        engine with a live user is never reaped or stopped, so a transient probe
        failure (fd exhaustion, host thrash) cannot kill a busy engine. Replace in
        place only when no live user holds it or it is this contract's own engine
        (pin-equal, serving only wanted models). A live incompatible engine in
        active use is never evicted or stacked on: on the machine slot it returns
        ``None`` (overflow), and in the overflow dir it serves nothing rather than
        duplicate weights beside it. Before building, any recorded engine is cleared
        -- keyed on
        the state file, not the probe -- so an unprobeable incumbent is stopped
        rather than double-built beside. The stop is gated on ``_can_build_engine``
        so a process that can serve nothing never destroys a warm engine it can't
        replace. Held under the cross-process build lock.
        """
        wanted = demand.pairs
        with build_lock(engine_dir):
            states = _healthy_states(engine_dir)
            if wanted and self._bind_all_in_dir(engine_dir, states, pin, demand):
                self._hold_membership(engine_dir)
                return True
            replaceable = not live_users_exist(engine_dir) or _healthy_groups_ours(
                states, pin, wanted
            )
            if not replaceable:
                # A live engine another setup is actively using is never evicted or
                # stacked on. On the machine slot that means overflow (None); in the
                # overflow dir there is nowhere further to go, so serve nothing rather
                # than kill the incumbent or load a second fleet's weights beside it
                # (an OOM on a small-VRAM box).
                return None if not is_overflow else False
            if not _can_build_engine(wanted):
                return False
            # No live user holds this dir now (or it is ours to rebuild): reap dead
            # leftovers and stop any recorded engine so planning sees true free VRAM
            # and the build never lands beside an unprobeable incumbent.
            reap_stale(engine_dir)
            if engine_record_exists(engine_dir):
                stop_engine(engine_dir)
            if self._plan_and_spawn(engine_dir):
                self._hold_membership(engine_dir)
                return True
            return False

    def _bind_all_in_dir(
        self,
        engine_dir: Path,
        states: dict[SwapGroup, SwapState],
        pin: str,
        demand: _EngineDemand,
    ) -> bool:
        """Bind every group needed to cover the demanded pairs; False leaves nothing bound.

        Binding never touches groups serving models outside the demand; the dir
        matches only when healthy, pin-equal groups cover every wanted pair and
        the served chat window covers the demanded per-slot ctx. (Whether an
        unmatched dir's engine is then replaced or overflowed around is the
        ladder's call, based on live users.)
        """
        wanted = demand.pairs
        candidates: list[tuple[SwapGroup, SwapState, list[InstanceLaunch]]] = []
        covered: set[tuple[WorkerRole, str]] = set()
        for group, state in states.items():
            found = _bindable_group(state, pin, wanted)
            if found is None:
                continue
            bindable, launches, pairs = found
            if not chat_ctx_covers(launches, demand.chat_ctx):
                # The live chat window is smaller than this process needs.
                return False
            candidates.append((group, bindable, launches))
            covered |= pairs
        if covered != wanted:
            return False
        bound: dict[SwapGroup, tuple[SwapManager, list[InstanceLaunch]]] = {}
        for group, state, launches in candidates:
            swap = SwapManager(engine_dir, group)
            if not swap.bind(state):
                for prior, _launches in bound.values():
                    prior.shutdown()
                return False
            bound[group] = (swap, launches)
        with self._lock:
            for group, (swap, launches) in bound.items():
                self._adopt_group(group, swap, launches)
                self._group_dirs[group] = engine_dir
        log.info("Bound to the running engine at %s", engine_dir)
        return True

    def _reload_dir(self) -> Path:
        """The engine dir a reload rebuilds into: where our groups already live.

        All this provider's groups share one dir by construction (the ladder is
        all-or-nothing per dir); an empty provider rebuilds into the machine slot.
        """
        with self._lock:
            dirs = set(self._group_dirs.values())
        return next(iter(dirs)) if dirs else machine_engine_dir()

    def _hold_membership(self, engine_dir: Path) -> None:
        """Record this process as a user of *engine_dir*'s engine.

        The single point every acquisition passes through, bind and build alike,
        so it is also where this user's persistence opt-in is recorded against
        the engine. Marking on bind (not only on build) is what makes the
        setting mean what it says on a shared slot: a user who asked for a warm
        engine keeps it warm even when a default-config sibling is last out.
        """
        from lilbee.core.config import cfg

        if engine_dir not in self._engine_holds:
            self._engine_holds[engine_dir] = hold_user_lock(engine_dir)
        if cfg.keep_engine_warm:
            request_keep_warm(engine_dir, cfg.data_root)

    def _plan_and_spawn(self, data_dir: Path) -> bool:
        """Plan placement against the clean box and start one swap per group.

        Caller holds the build lock. False when the engine binary is missing or
        nothing is installed/configured, so the provider serves nothing.
        """
        try:
            # Snapshot the clean box; this plan and every later reload size
            # ctx, slots, and budgets against it (a live probe under a loaded
            # fleet would report our own residency as unavailable). Inside the
            # try: capturing resolves the engine binary, and a binary-less
            # host must serve nothing, not raise. Every other planning failure
            # (a wedged GPU probe, an unusable CUDA runtime) propagates so the
            # warm tracker and the caller report the real reason instead of a
            # silent never-ready fleet.
            planning.capture_plan_probe()
            plan = planning.plan_all_launches()
        except ProviderError as exc:
            # Only a genuinely-missing engine binary keeps the quiet no-fleet path;
            # any other planning failure (a wedged GPU probe, an unusable CUDA
            # runtime) must surface to the warm tracker and on-demand callers
            # rather than silently serving nothing (#540).
            if exc.kind is not ProviderErrorKind.NOT_FOUND:
                raise
            log.debug("Engine binary unavailable; no swap started")
            plan = None
        # plan None (no engine binary) keeps the demand-time record from
        # _acquire_engine instead of wiping it.
        if plan is not None:
            self._skipped_not_installed = dict(plan.skipped_not_installed)
            self._skipped_unusable_ctx = dict(plan.skipped_unusable_ctx)
        if plan is None or not plan.launches:
            # No engine binary, or no installed/configured model: serve nothing.
            return False
        by_group = _launches_by_group(plan)
        started: dict[SwapGroup, SwapManager] = {}
        try:
            for group, group_launches in by_group.items():
                swap = SwapManager(data_dir, group)
                swap.start(
                    list(group_launches),
                    ttl_seconds=_warm_ttl_seconds(
                        hold_warm_for_session=self._hold_warm_for_session
                    ),
                    bind_lifetime=not cfg.keep_engine_warm,
                )
                started[group] = swap
        except BaseException:
            for swap in started.values():
                swap.shutdown()
            raise
        with self._lock:
            for group, swap in started.items():
                self._adopt_group(group, swap, list(by_group[group]))
                self._group_dirs[group] = data_dir
        return True

    def _adopt_group(
        self, group: SwapGroup, swap: SwapManager, launches: list[InstanceLaunch]
    ) -> None:
        """Record *group*'s freshly started swap and build a client pool per role.

        Caller holds ``self._lock``. Each launch (one per replica) becomes a client
        keyed by its replica model id against this group's own proxy endpoint;
        the chat launch carries the slots/ctx so the capacity and served context
        come from the launch, not a probe.
        """
        self._swaps[group] = swap
        self._launches[group] = tuple(launches)
        endpoint = swap.endpoint()
        for role, role_launches in _by_role(launches).items():
            # Retire the role's previous clients (a reload re-adopts over an existing
            # pool): closing them now would error a reader still mid-call on an old
            # client snapshot, and never closing leaks an httpx pool per replica.
            old_clients = list(self._clients.get(role, []))
            self._role_group[role] = group
            # token_cap truncates oversize embed/rerank inputs to the per-slot context
            # (the in-process backstop); the longer timeout covers a cold upstream load.
            self._clients[role] = [
                LlamaServerClient(
                    endpoint,
                    launch.model_id,
                    token_cap=launch.token_cap,
                    timeout=_request_timeout_s(launch.weights_bytes),
                    rerank_mode=launch.rerank_mode,
                    inline_reasoning=role is WorkerRole.CHAT,
                    on_prefill=self._record_chat_prefill if role is WorkerRole.CHAT else None,
                    # A cold embed replica 429s bulk ingest until its slots load; wait
                    # out the same cold-load budget llama-swap keeps it alive for so a
                    # burst never drops files while the server is legitimately warming.
                    embed_busy_deadline_s=(
                        cold_load_timeout_s(launch.weights_bytes)
                        if role is WorkerRole.EMBED
                        else None
                    ),
                )
                for launch in role_launches
            ]
            if role is WorkerRole.CHAT:
                self._chat_slots = role_launches[0].slots
                self._chat_ctx = role_launches[0].ctx
                # Every serving process passes through adoption (fresh launch,
                # reload, guest bind), so the warning fires in each of them.
                planning.warn_when_chat_downsized(role_launches[0])
            self._retire_clients(old_clients)

    def _swap_for(self, role: WorkerRole) -> SwapManager | None:
        """The swap process serving *role*, or None when the role has no server.

        Caller holds ``self._lock``.
        """
        group = self._role_group.get(role)
        return None if group is None else self._swaps.get(group)

    def _role_launches(self, role: WorkerRole) -> tuple[InstanceLaunch, ...]:
        """*role*'s launch snapshot, empty when it has no server.

        A co-tenant group holds more than one role's launches, so the group's
        snapshot is filtered down to this role's replicas.
        """
        group = self._role_group.get(role)
        if group is None:
            return ()
        return tuple(launch for launch in self._launches.get(group, ()) if launch.role is role)

    def _drop_group(self, group: SwapGroup) -> SwapManager | None:
        """Forget *group*'s swap/launches and every member role's clients.

        Caller holds ``self._lock``. Member clients are retired (closed at a later
        reload or shutdown, never while a reader could still hold one) and the chat
        capacity falls back to its defaults when chat's group is dropped.
        """
        swap = self._swaps.pop(group, None)
        self._launches.pop(group, None)
        # Prune the dir map with the group: a stale entry outliving its group makes
        # _reload_dir see two dirs and pick one arbitrarily, splitting the provider.
        self._group_dirs.pop(group, None)
        for role in [r for r, g in self._role_group.items() if g is group]:
            del self._role_group[role]
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
            swap = self._swap_for(role)
        if not clients and swap is not None and not swap.is_live():
            self._rebuild_role(role)
            with self._lock:
                clients = self._clients.get(role)
        if not clients:
            # A refused launch's recorded reason wins over the generic line.
            # BAD_REQUEST so the HTTP surfaces return this deterministic,
            # actionable refusal in a 4xx body instead of a generic 500.
            reason = self._skipped_unusable_ctx.get(role)
            if reason is not None:
                raise ProviderError(
                    reason, provider=_PROVIDER_NAME, kind=ProviderErrorKind.BAD_REQUEST
                )
            raise ProviderError(_no_server_message(role), provider=_PROVIDER_NAME)
        return list(clients)

    def _with_rediscover(self, call: Callable[[], _T], *, role: WorkerRole | None = None) -> _T:
        """Run *call*; on a connection-kind or load-capacity failure, retry once.

        A vanished engine (its last user left on a config change, or it died)
        surfaces as ProviderErrorKind.CONNECTION, or as a raw httpx transport
        error when the proxy itself is gone (nothing listening to answer with
        a status). Membership is still held, so dropping the swap refs and
        retrying sends the call through _ensure_fleet, which rediscovers the
        new proxy ports or rebuilds. One retry only; a second failure surfaces
        to the caller.

        A ProviderErrorKind.CAPACITY failure is the engine dying on load because
        the estimate was too optimistic. Retrying it unchanged respawns the same
        launch into the same death, so *role*'s auto context steps down first and
        the role is rebuilt against the smaller plan. When there is no step left
        to take (a user-pinned context, or already at the floor) the failure
        surfaces instead: a retry that asks for the same thing is a crash loop.
        """
        try:
            return call()
        except (ProviderError, httpx.TransportError) as err:
            if is_rebuildable_failure(err) and role is not None:
                return self._retry_rebuilt(call, role, err)
            if not is_connection_failure(err):
                raise
            log.info("Engine unreachable; rediscovering before one retry")
            self._drop_swap_refs()
            self._release_holds()
            return call()

    def _retry_rebuilt(self, call: Callable[[], _T], role: WorkerRole, err: BaseException) -> _T:
        """Rebuild *role* so the retry is a different launch, and run *call* again.

        A held port just needs the rebuild, which picks a new one. A memory
        shortfall needs the plan to come back smaller too, so the context steps
        down first; when there is no step left to take, *err* is re-raised
        untouched rather than rebuilding into the same death.
        """
        from lilbee.providers.fleet.planning import record_ctx_downshift

        if is_load_capacity_failure(err):
            if not record_ctx_downshift(role):
                log.warning(
                    "%s ran out of device memory on load and its context cannot be "
                    "reduced further; lower num_ctx or use a smaller model",
                    role.value,
                )
                raise err
            log.warning(
                "%s ran out of device memory on load; re-planning it with a smaller "
                "context where its window has room to give",
                role.value,
            )
        else:
            log.warning("%s could not claim its port; rebuilding it on a new one", role.value)
        self._rebuild_role(role)
        return call()

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
            swap = self._swap_for(role)
        return swap is not None and swap.role_ready(role)

    def max_concurrent_chats(self) -> int:
        """The chat server's batching-slot capacity, so the gate admits that many.

        Falls back to ``1`` before the chat group is up, so chat is serialized
        until the slot count is known (the launcher warms the engine before a
        client connects, so the real capacity is in effect by the first chat).
        """
        with self._lock:
            if WorkerRole.CHAT not in self._role_group:
                return 1
            return self._chat_slots

    def served_chat_ctx(self) -> int | None:
        """Per-slot context the chat server runs with, or None if not up."""
        with self._lock:
            return self._chat_ctx if WorkerRole.CHAT in self._role_group else None

    def served_chat_slots(self) -> int | None:
        """Batching slots the chat server runs with, or None if not up."""
        with self._lock:
            return self._chat_slots if WorkerRole.CHAT in self._role_group else None

    def chat_prefill_progress(self) -> tuple[int, int] | None:
        """``(processed, total)`` of a chat prefill in flight, or None when idle."""
        with self._lock:
            return self._chat_prefill

    def _record_chat_prefill(self, progress: tuple[int, int] | None) -> None:
        """Store a stream's prefill reading. Latest writer wins across concurrent
        streams; a live prefill rewrites itself on its next engine batch."""
        with self._lock:
            self._chat_prefill = progress

    def warm_pending(self) -> bool:
        """Whether a requested warm is still running.

        The tracker only stamps a phase once the chat role starts loading, which is
        seconds after the swap is spawned, so ``warm_progress`` alone cannot tell a
        not-yet-started warm from no warm at all.
        """
        with self._lock:
            return self._warming

    def warm_progress(self) -> WarmProgress | None:
        """Live cold-load progress for the chat role, or None before warm begins."""
        return self._warm_tracker.snapshot()

    def _shutdown_swap(self, *, latch: bool = True) -> None:
        """Release this process's engine use; ``latch=False`` keeps the provider reusable.

        Terminal ``shutdown()`` latches ``_shut_down`` so a discarded provider's
        in-flight warm/reload thread can't spawn an orphan swap, then releases
        membership: the engine stops only when this was the last user and
        persistence was not opted into. The cache-drop paths
        (``invalidate_load_cache``, ``drop_loaded_models_async``) pass
        ``latch=False``: a config change restarts the shared engine for every
        user (they rediscover), and this provider rebuilds on next use.
        """
        # Latched before the lock, not inside it: every _shut_down check runs after
        # acquiring the build lock, so a warm or reload thread queued behind us can
        # only bail early if the flag is already set when its turn comes.
        if latch:
            with self._lock:
                self._shut_down = True
        # The build lock serializes shutdown against a concurrent reload/build:
        # both mutate self._swaps and the llama-swap processes, so an unserialized
        # loser would overwrite the winner's state and leak a live llama-swap.
        # Bounded, because a wedged engine start holds this lock and an unbounded
        # wait would hang process exit outright. On timeout the teardown proceeds
        # anyway: whatever the builder leaves behind is recorded in the engine
        # dir's state files, so the next start's reap finds it by record, while a
        # shutdown that never returns cannot be recovered from at all.
        acquired = self._build_lock.acquire(timeout=_SHUTDOWN_BUILD_LOCK_WAIT_S)
        if not acquired:
            log.warning(
                "Engine build still in progress after %.0fs; shutting down without "
                "waiting for it. Leftovers are reaped from their records on the next start.",
                _SHUTDOWN_BUILD_LOCK_WAIT_S,
            )
        try:
            # Terminal shutdown closes every client; a config-change teardown
            # retires them so an in-flight reader is never severed.
            self._drop_swap_refs(close_all=latch)
            self._release_engines(config_changed=not latch)
        finally:
            if acquired:
                self._build_lock.release()

    def _release_engines(self, *, config_changed: bool = False) -> None:
        """Drop membership in every used engine dir; stop each engine we leave last.

        Runs under each dir's cross-process build lock so a departing last user
        can never race an arriving binder: the arrival either sees the engine
        (and its bind holds it live) or sees the slot empty and builds.

        Whether the engine outlives us is the union of every user's opt-in, not
        just the exiting process's config: the machine slot is shared by
        installations that configure it differently, and which one leaves last
        is arbitrary.

        *config_changed* is the cache-drop path, where this provider's settings
        or model changed. That makes the running engine stale for us, so no
        persistence opt-in preserves it -- but it says nothing about the peers
        still serving requests against it, so a shared engine is left running
        and the next use re-runs the ladder, binding it if it happens to match
        and overflowing to a private dir if it does not.

        The hold map is cleared either way. Leaving a stale hold behind is not
        benign: after a lazy rebuild overflows to a private dir (a foreign
        process having claimed the machine slot in the gap), the next release
        would iterate the stale machine hold and stop that foreign engine
        mid-use, and the stale flock would keep live_users_exist true so the
        foreign engine's real last user could never reap it.
        """
        from lilbee.core.config import cfg

        for engine_dir, hold in list(self._engine_holds.items()):
            with build_lock(engine_dir, best_effort=True):
                last = hold.release_and_check_last()
                # A flip after binding never re-acquires, so reconcile our own mark
                # here. Skipped on a config change, which stops and clears regardless.
                if not config_changed:
                    if cfg.keep_engine_warm:
                        request_keep_warm(engine_dir, cfg.data_root)
                    else:
                        withdraw_keep_warm(engine_dir, cfg.data_root)
                # Any remaining opt-in keeps the engine, including a peer's.
                keep = not config_changed and keep_warm_requested(engine_dir)
                if last and not keep:
                    stop_engine(engine_dir)
                    log.info("Engine stopped at %s (last user out)", engine_dir)
                elif last:
                    log.info("Engine left warm at %s (last user out)", engine_dir)
                elif config_changed:
                    log.info("Engine left running at %s (still in use by peers)", engine_dir)
        self._engine_holds = {}

    def _release_holds(self) -> None:
        """Drop this process's engine memberships without stopping any engine.

        The rediscover retry re-runs the acquisition ladder. A retained membership
        would make this process count itself as a live user of the machine slot, so
        the ladder would judge the slot in use and overflow to a private engine
        instead of rebinding a recovered engine or rebuilding a dead one -- N
        private engines and N times the VRAM after the shared engine first dies.
        Nothing is stopped here: a live engine is rebound by the retry, and a dead
        one is cleared by the rebuild that retry triggers.
        """
        for hold in list(self._engine_holds.values()):
            hold.release_and_check_last()
        self._engine_holds = {}

    def _drop_swap_refs(self, *, close_all: bool = False) -> None:
        """Clear every group's swap/clients and the chat capacity so the next call rebuilds.

        Live pools are RETIRED through the ``in_flight`` check rather than closed
        outright: ``_with_rediscover`` reaches this on any connection blip, so a
        chat proxy hiccup must not sever the client another thread is mid-embed or
        mid-stream on (a streamed response is handed out after the retry returns,
        and failures past the first frame are not retried). Idle clients close now,
        busy ones stay retired for a later pass.

        *close_all* is the terminal-shutdown path, where nothing will read again and
        whatever remains must actually be closed.
        """
        doomed: list[LlamaServerClient] = []
        with self._lock:
            live = [client for pool in self._clients.values() for client in pool]
            self._swaps = {}
            self._launches = {}
            self._role_group = {}
            self._group_dirs = {}
            self._clients = {}
            if close_all:
                doomed = live + self._retiring_clients
                self._retiring_clients = []
            else:
                self._retire_clients(live)
            self._chat_slots = 1
            self._chat_ctx = None
            # A torn-down fleet's load failures describe servers that no longer
            # exist; the next warm records its own.
            self._warm_errors = {}
        # Full teardown: the next build starts from a clean box, so it must
        # re-snapshot memory rather than plan against this boot's probe.
        planning.clear_plan_probe()
        for client in doomed:
            client.close()

    def _drop_dead_swaps(self) -> None:
        """Drop the refs of groups whose process is gone so the next call rebuilds them.

        A no-op for groups still running (e.g. the failure was in planning), so
        a live engine is never abandoned unstopped.
        """
        with self._build_lock, self._lock:
            for group in [g for g, swap in self._swaps.items() if not swap.running]:
                self._drop_group(group)

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
        from lilbee.providers.engine_params import chat_options_to_kwargs

        self._require_configured_model(model, str(cfg.chat_model), WorkerRole.CHAT)
        self._require_clients(WorkerRole.CHAT)
        messages = self._fit_chat_context(messages, tools, options, model or str(cfg.chat_model))
        # Translate options exactly as the in-process path did (validate via
        # LLMOptions, num_predict -> max_tokens, drop num_ctx) so the server
        # honors the same generation settings; a raw passthrough would drop
        # num_predict and leak the load-only num_ctx.
        server_options = chat_options_to_kwargs(options) or None
        if stream:
            # The first frame is pulled eagerly so a dead proxy fails inside
            # _with_rediscover; a failure past the first frame surfaces to the
            # caller as a retry error, and rediscovery covers the next call.
            return self._with_rediscover(
                lambda: _primed_stream(
                    _least_in_flight(self._require_clients(WorkerRole.CHAT)).chat_stream_items(
                        messages, tools=tools, tool_choice=tool_choice, options=server_options
                    )
                ),
                role=WorkerRole.CHAT,
            )
        return self._with_rediscover(
            lambda: _least_in_flight(self._require_clients(WorkerRole.CHAT)).chat_result(
                messages, tools=tools, tool_choice=tool_choice, options=server_options
            ),
            role=WorkerRole.CHAT,
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
        from lilbee.providers.engine_params import chat_options_to_kwargs

        self._require_configured_model(model, str(cfg.chat_model), WorkerRole.CHAT)
        self._require_clients(WorkerRole.CHAT)
        messages = self._fit_chat_context(messages, tools, options, model or str(cfg.chat_model))
        server_options = chat_options_to_kwargs(options) or None
        return self._with_rediscover(
            lambda: _least_in_flight(self._require_clients(WorkerRole.CHAT)).chat_tools(
                messages, tools=tools, tool_choice=tool_choice, options=server_options
            ),
            role=WorkerRole.CHAT,
        )

    def _fit_chat_context(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any] | None,
        model: str,
    ) -> list[ChatMessage]:
        """Drop oldest turns so the prompt fits the served context.

        A ``num_predict`` reservation larger than the default generation room is
        capped to it, so an agent client that over-reserves keeps its history
        instead of having it evicted; llama-server stops at the context edge
        anyway. A smaller reservation is honored as-is and widens the prompt.
        Raises ``ProviderError(CONTEXT_OVERFLOW)`` only when system messages,
        tools, and the final turn exceed the window even with the capped
        reserve (mapped to a 400 by the chat-completions route).
        """
        # 0/None means the served context is unknown (no chat launch adopted yet);
        # a real per-slot context is always positive, so skip windowing.
        if not self._chat_ctx:
            return messages
        # An output reservation only ever buys the prompt MORE room, never less:
        # a num_predict past the default is a ceiling on what the model may
        # generate, not a claim on prompt space, and llama-server stops at the
        # context edge regardless. Capping it here rather than retrying after a
        # failed fit is the difference between a policy and a rescue -- an agent
        # reserving most of the window leaves a budget of a few dozen tokens, in
        # which the final turn still "fits" while the whole conversation is
        # silently evicted.
        requested = (options or {}).get("num_predict")
        reserve = min(requested, GENERATION_RESERVE_TOKENS) if requested else None
        result = window_messages(messages, tools, prompt_token_budget(self._chat_ctx, reserve))
        if not result.fits:
            raise ProviderError(
                f"Prompt of about {result.prompt_tokens} tokens exceeds the "
                f"{self._chat_ctx}-token context window for {model!r}. Shorten the "
                "conversation or the system prompt.",
                provider=_PROVIDER_NAME,
                kind=ProviderErrorKind.CONTEXT_OVERFLOW,
            )
        return result.messages

    def embed(self, texts: list[str]) -> list[Vector]:
        return self._with_rediscover(lambda: self._embed_once(texts), role=WorkerRole.EMBED)

    def _embed_once(self, texts: list[str]) -> list[Vector]:
        clients = self._require_clients(WorkerRole.EMBED)
        return _call_with_failover(clients, lambda client: client.embed(texts))

    def count_tokens(self, text: str) -> int:
        """Exact token count of *text* under the embedding model's tokenizer.

        Routes to the embed server's ``/tokenize`` so chunk sizing counts the same
        tokens the embedder will consume. Raises ``ProviderError`` when no embed
        server is configured; callers on the chunk-sizing path degrade to an
        estimate rather than propagate it.
        """
        clients = self._require_clients(WorkerRole.EMBED)
        return _call_with_failover(clients, lambda client: client.count_tokens(text))

    def vision_ocr(
        self, png_bytes: bytes, model: str, prompt: str = "", *, timeout: float | None = None
    ) -> str:
        from lilbee.vision import build_vision_messages, resolve_ocr_prompt

        self._require_configured_model(model, str(cfg.vision_model), WorkerRole.VISION)
        pool = self._vision_pool()
        effective = model or str(cfg.vision_model)
        messages = build_vision_messages(prompt or resolve_ocr_prompt(effective), png_bytes)
        try:
            return _ocr_dispatch(pool, messages, _ocr_deadline(timeout))
        except _PageBudgetExhausted:
            raise ProviderError(
                "Vision OCR timed out waiting for a free vision slot.",
                provider=_PROVIDER_NAME,
            ) from None

    def vision_slot_capacity(self) -> int | None:
        """Total fitted ``--parallel`` slots across the running vision replicas.

        ``None`` before the fleet is up (no launch snapshot yet), so the ingest
        fan-out keeps its own estimate until real capacity is known. A modest
        card that fit fewer slots than requested reports the smaller real number,
        so the fan-out never queues more pages than the servers can serve.
        """
        launches = self._role_launches(WorkerRole.VISION)
        if not launches:
            return None
        return max(1, sum(launch.slots for launch in launches))

    def _vision_pool(self) -> list[_VisionReplica]:
        """Each vision replica paired with its fitted ``--parallel`` slot count.

        The fitted count can be lower than ``vision_ocr_concurrency`` when memory
        forced a smaller fit; dispatching at the configured ceiling instead
        over-subscribes that server. The configured ceiling applies per replica
        only when no matching launch snapshot exists (a reload can momentarily
        drop it between two reads).
        """
        clients = self._require_clients(WorkerRole.VISION)
        launches = self._role_launches(WorkerRole.VISION)
        if launches and len(launches) == len(clients):
            return [
                _VisionReplica(client, max(1, launch.slots))
                for client, launch in zip(clients, launches, strict=True)
            ]
        fallback_slots = max(1, cfg.vision_ocr_concurrency)
        return [_VisionReplica(client, fallback_slots) for client in clients]

    # PDF/image OCR now runs inside xberg via the registered lilbee-vision
    # backend (see data.extract.backends.vision_ocr); this provider only exposes
    # single-image vision_ocr, which that backend calls.

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        return self._with_rediscover(
            lambda: self._rerank_once(query, candidates), role=WorkerRole.RERANK
        )

    def _rerank_once(self, query: str, candidates: list[str]) -> list[float]:
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
            if self._warming:
                return
            fleet_up = bool(self._swaps)
        # A live swap whose model llama-swap idle-unloaded (its ttl stops only the
        # llama-server child, leaving the swap handle in _swaps) reports its role
        # cold. Re-warm so a prompt sent into that gap drives llama-swap's
        # on-demand reload; bailing on "swaps exist" alone stranded every later
        # prompt on a stale not-ready. A fully-loaded fleet still short-circuits.
        # The probe runs off the lock (role_ready may hit the proxy).
        if fleet_up and self._roles_ready():
            return
        with self._lock:
            if self._warming:
                return
            self._warming = True
        threading.Thread(
            target=self._warm_up_blocking,
            name="fleet-warm-up",
            daemon=True,
        ).start()

    def _roles_ready(self) -> bool:
        """Whether every configured role's upstream is loaded (fleet fully warm)."""
        with self._lock:
            roles = list(self._role_group)
        return bool(roles) and all(self.role_ready(role) for role in roles)

    def _warm_up_blocking(self) -> None:
        """Start the fleet and pre-load every role on a background thread.

        Runs on a daemon thread with no caller to catch failures, so a startup
        error is logged and swallowed: a role that can't load surfaces a
        user-facing ProviderError on the next call, not a thread traceback.

        The tracker is stamped STARTING before the fleet spawn so surfaces
        show the engine coming up from the first moment (spawn plus health
        check takes seconds and previously reported nothing), and stamped
        ERROR with the real reason when the warm fails before the chat warm
        proper begins.
        """
        try:
            self._warm_tracker.begin(str(cfg.chat_model))
            self._ensure_fleet()
            self._preload_roles()
            self._finalize_warm_if_chat_never_ran()
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
                log.warning("Engine warm-up failed: %s", exc)
                log.debug("Engine warm-up failure detail.", exc_info=True)
            self._fail_warm_unless_ready(str(exc))
        finally:
            with self._lock:
                self._warming = False

    def _fail_warm_unless_ready(self, message: str) -> None:
        """Stamp the warm tracker ERROR unless the chat warm already finished.

        A failure in a later role's preload must not clobber a chat warm that
        reached READY; every earlier failure leaves the tracker mid-phase,
        where surfaces would spin forever and the prompt path could not name
        the reason.
        """
        snapshot = self._warm_tracker.snapshot()
        if snapshot is None or snapshot.phase is not WarmPhase.READY:
            self._warm_tracker.fail(message)

    def _finalize_warm_if_chat_never_ran(self) -> None:
        """Terminate the early STARTING stamp when no chat instance was placed.

        ``_warm_chat_role`` always ends in READY or ERROR when it runs, so a
        snapshot still on STARTING after a successful preload means the plan had
        no chat instance. A chat model that isn't installed, one whose launch the
        plan refused for an unusable window, and one with no engine to run it all
        fail the warm with a user-facing reason so the prompt path renders
        "failed to load" instead of spinning a "not ready" retry that can never
        succeed; any other reason (a remote-routed chat has no local server to
        warm) clears the stamp.
        """
        snapshot = self._warm_tracker.snapshot()
        if snapshot is None or snapshot.phase is not WarmPhase.STARTING:
            return
        missing = self._skipped_not_installed.get(WorkerRole.CHAT)
        if missing is not None:
            self._warm_tracker.fail(f"chat model {clean_display_name(missing)} is not installed")
            return
        unusable = self._skipped_unusable_ctx.get(WorkerRole.CHAT)
        if unusable is not None:
            self._warm_tracker.fail(unusable)
            return
        if _chat_needs_local_engine() and (reason := _unusable_engine_reason()) is not None:
            self._warm_tracker.fail(reason)
            return
        self._warm_tracker.clear()

    def _preload_roles(self, roles: frozenset[WorkerRole] | None = None) -> None:
        """Issue a cheap request per replica so llama-swap loads each upstream now.

        llama-swap starts an upstream on its first request, so warming sends a
        minimal call to every replica of every role (firing the spawn listeners
        around each role). A per-replica failure is logged and skipped; that replica
        still loads on its first real use. The chat role routes through
        :meth:`_warm_chat_role` so a launcher gets granular progress. *roles*
        narrows the warm to just those roles (a reload warms only what restarted).

        Roles on separate devices warm concurrently: chat is the long pole (a
        large model's load dominates), so the light roles load alongside it
        instead of before it. Roles whose launches pin overlapping devices warm
        one at a time instead, chat last: two engines loading into the same
        card at once race each other for VRAM, and the loser's first load can
        OOM even though both fit once settled.
        """
        with self._lock:
            pools = {
                role: list(clients)
                for role, clients in self._clients.items()
                if roles is None or role in roles
            }
            on_spawning, on_spawned = self._on_spawning, self._on_spawned
            device_sets = _role_device_sets(
                launch for launches in self._launches.values() for launch in launches
            )

        if not pools:
            return
        listeners = (on_spawning, on_spawned)
        chains = _warm_chains(list(pools), device_sets)
        with ThreadPoolExecutor(
            max_workers=len(chains), thread_name_prefix="fleet-preload"
        ) as pool:
            futures = [pool.submit(self._warm_chain, chain, pools, listeners) for chain in chains]
            for future in futures:
                future.result()

    def _warm_chain(
        self,
        chain: list[WorkerRole],
        pools: dict[WorkerRole, list[LlamaServerClient]],
        listeners: tuple[Callable[[WorkerRole], None] | None, Callable[[WorkerRole], None] | None],
    ) -> None:
        """Warm *chain*'s roles one at a time; every role gets its attempt.

        An unexpected error warming one role (a listener blowing up) must not rob
        the roles behind it of their warm, so the first error is re-raised only
        after the chain finishes.
        """
        on_spawning, on_spawned = listeners
        first_exc: Exception | None = None
        for role in chain:
            try:
                if on_spawning is not None:
                    on_spawning(role)
                if role is WorkerRole.CHAT:
                    self._warm_chat_role(pools[role])
                else:
                    self._warm_role_clients(role, pools[role])
                if on_spawned is not None:
                    on_spawned(role)
            except Exception as exc:
                first_exc = first_exc or exc
        if first_exc is not None:
            raise first_exc

    def _warm_role_clients(self, role: WorkerRole, clients: list[LlamaServerClient]) -> bool:
        """Warm every replica of *role*; return whether at least one loaded.

        A replica that fails to load is reported at warning level with the engine's
        own message (an unsupported architecture, a corrupt file). Warm-up stays
        best-effort, but the failure must not be silent: the role then serves
        nothing, and a caller that never reaches it would otherwise see only an
        unexplained empty answer.
        """
        warmed = False
        self._warm_errors.pop(role, None)
        for client in clients:
            try:
                _warm_role(role, client)
                client.mark_healthy()
                warmed = True
            except Exception as exc:
                # A replica that cannot load is not routable. Marking it takes it
                # out of the pool so calls go to a sibling on a device that works,
                # instead of every request picking the dead one again. It is a
                # device fault as often as a model one: an adapter that enumerates
                # but cannot allocate fails here and nowhere else. The health flag
                # carries its own cool-down, so a device that recovers rejoins
                # without anything having to remember it was bad.
                client.mark_unhealthy()
                self._warm_errors[role] = str(exc)
                log.warning(
                    "The %s model failed to load: %s",
                    role.value,
                    exc,
                    exc_info=log.isEnabledFor(logging.DEBUG),
                )
        if warmed:
            self._warm_errors.pop(role, None)
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
                self._warm_tracker.fail(self._chat_load_failure())

    def _chat_load_failure(self) -> str:
        """The engine's own reason the chat model did not load, when it gave one."""
        reason = self._warm_errors.get(WorkerRole.CHAT)
        if not reason:
            return "The chat model did not finish loading."
        return f"The chat model did not load: {reason}"

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
        """Sever every in-flight chat stream so its blocked reader unwinds.

        A cooperative worker cancel cannot reach a thread blocked in a socket
        read, and the reader's own close runs only when its worker unwinds, so
        the disconnect must happen here. Retired clients are swept too: a
        model-swap reload retires a busy client before the cancel lands.
        """
        with self._lock:
            clients = [*self._clients.get(WorkerRole.CHAT, ()), *self._retiring_clients]
        for client in clients:
            client.abort_streams()

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

    def _rebind_or_overflow(self) -> list[WorkerRole]:
        """Re-acquire a bound engine after a config change; caller holds the build lock.

        A provider that rode another process's engine owns none of its groups and
        cannot restart them. Restarting "in place" would spawn a duplicate fleet
        into the shared slot (a bound manager's ``shutdown`` only detaches, leaving
        the incumbent resident) and size it blind against VRAM the incumbent still
        holds. Instead drop every binding and this process's membership, then re-run
        the acquisition ladder: it rebinds to the reconfigured shared engine, builds
        fresh in the machine slot if we were its last user, or overflows to a private
        engine sized against a fresh probe. Returns the roles now served (to preload).
        """
        from lilbee.core.config import cfg

        with self._lock:
            groups = list(self._swaps)
        for group in groups:
            with self._lock:
                swap = self._drop_group(group)  # also prunes _group_dirs
            if swap is not None:
                swap.shutdown()  # bound: detaches; the shared engine keeps running
        self._release_engines()  # a shared engine's builder keeps it live; no stop here
        if not self._acquire_engine(cfg.data_root):
            return []
        with self._lock:
            return list(self._role_group)

    def _reload_pass(self, force: frozenset[WorkerRole] = frozenset()) -> None:
        """One re-plan from current cfg, restarting only the groups that changed.

        The fresh plan is diffed per swap group against the launches each running
        group was started with; a group restarts only when its launches differ
        (covers added and removed groups too), so an untouched group's loaded model
        stays resident through a placement or per-role model change. *force* adds a
        role's group to the restart set even when its plan is unchanged (dead-swap
        recovery). Changed groups stop before the new ones start, so the planned
        VRAM is actually free when the new servers spawn. Runs under the build lock
        so a racing shutdown/build can't interleave with the restart and leak a live
        llama-swap holding GPU memory.
        """

        restarted: list[WorkerRole] = []
        with self._build_lock:
            # The device list is structural and was captured once at boot, so a
            # card that has since left keeps being planned onto. The memory
            # figures beside it are deliberately not re-taken: this fleet is
            # resident, and charging it against itself is what the snapshot exists
            # to prevent.
            planning.refresh_plan_devices()
            with self._lock:
                if self._shut_down:
                    # Terminal shutdown landed while this reload was queued; a
                    # rebuild here would spawn a fleet no live provider owns.
                    return
                running = set(self._swaps)
                old = dict(self._launches)
                # All groups share one dir and one ownership by construction, so any
                # bound manager means this provider rides another process's engine.
                bound = any(swap.bound for swap in self._swaps.values())
            if bound:
                # Cannot restart a shared engine's groups in place (that duplicates
                # the fleet into the slot); drop the bindings and re-acquire.
                restarted = self._rebind_or_overflow()
                self._preload_restarted(restarted)
                return
            # Reap dead engines in our dirs before re-planning, as a build would.
            reload_dir = self._reload_dir()
            # Serialize the reload against peer acquisitions with the same
            # cross-process lock a build takes. Without it, this reap_stale can kill
            # a peer's swap that is spawned but not yet answering its proxy, and the
            # stop-then-spawn gap lets a peer's ladder see a half-stopped slot and
            # build a second fleet into the same dir, double-allocating VRAM.
            with build_lock(reload_dir):
                reap_stale(reload_dir)
                try:
                    if not running:
                        # Nothing loaded (a resurrect after a failed pass): the box is
                        # clean, so refresh the plan snapshot like a first build would.
                        planning.capture_plan_probe()
                    plan = planning.plan_all_launches()
                except ProviderError as exc:
                    # Same policy as the initial build: a genuinely-missing engine
                    # binary aborts the reload quietly (nothing to serve), while any
                    # other planning failure (a wedged GPU probe, an unusable CUDA
                    # runtime) propagates to fail loud. The raise lands before the
                    # stop phase, so a running fleet is left intact rather than half
                    # torn down.
                    if exc.kind is not ProviderErrorKind.NOT_FOUND:
                        raise
                    log.debug("Engine binary unavailable; reload left the fleet as-is")
                    return
                # Keep the skip reasons in step with the fresh plan.
                self._skipped_not_installed = dict(plan.skipped_not_installed)
                self._skipped_unusable_ctx = dict(plan.skipped_unusable_ctx)
                new = _launches_by_group(plan)
                # A group restarts when its launches changed OR its running/planned
                # presence disagrees (covers a group the new plan drops or adds).
                changed = {
                    group
                    for group in running | set(new)
                    if (group in running) != (group in new)
                    or old.get(group, ()) != new.get(group, ())
                }
                changed |= {group_for(role, plan.co_tenants) for role in force}
                # Stop phase: free the changed groups' VRAM before their replacements
                # (or another group's grown plan) spawn against it.
                for group in sorted(changed, key=lambda g: g.value):
                    with self._lock:
                        swap = self._drop_group(group)
                    if swap is not None:
                        swap.shutdown()
                # Start phase: spawn the changed groups present in the new plan.
                for group in sorted(changed & set(new), key=lambda g: g.value):
                    group_launches = list(new[group])
                    swap = SwapManager(reload_dir, group)
                    swap.start(
                        group_launches,
                        ttl_seconds=_warm_ttl_seconds(
                            hold_warm_for_session=self._hold_warm_for_session
                        ),
                        bind_lifetime=not cfg.keep_engine_warm,
                    )
                    with self._lock:
                        self._adopt_group(group, swap, group_launches)
                        self._group_dirs[group] = reload_dir
                    restarted.extend(_by_role(group_launches))
        self._preload_restarted(restarted)

    def _preload_restarted(self, restarted: list[WorkerRole]) -> None:
        """Load the restarted roles' models off-thread (a no-op for none).

        llama-swap spawns an upstream on its first request, so the reload returns
        once the proxies answer and the UI's spawn listeners track the model loads.
        """
        if not restarted:
            return
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
