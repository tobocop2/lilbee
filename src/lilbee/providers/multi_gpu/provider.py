"""FleetProvider: the local llama-server engine for every role.

On first use it plans GPU placement and spawns one llama-server per configured
role (chat/embed/rerank/vision), then routes each call to the least-busy healthy
server for that role. A single machine is a fleet-of-one; there is no in-process
fallback, so a missing or unhealthy server surfaces a user-facing
``ProviderError``. Model management (list/show/capabilities) reads the registry
and GGUF headers directly and needs no running server.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType
from lilbee.providers.multi_gpu.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.multi_gpu.binary import llama_server_runtime_env, resolve_llama_server_binary
from lilbee.providers.multi_gpu.client import LlamaServerClient
from lilbee.providers.multi_gpu.devices import FleetDevice, probe_devices, visible_env
from lilbee.providers.multi_gpu.fleet import Fleet, InstanceLaunch
from lilbee.providers.multi_gpu.placement import (
    InstancePlan,
    ModelPlacementInput,
    estimate_model_vram,
    plan_placement,
)
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lilbee.providers.base import (
        ChatMessage,
        ClosableIterator,
        OcrBackend,
        PageText,
    )

# Fleet-only concurrency: continuous-batching slots (--parallel) per server. The
# in-process pool has no equivalent (one worker per role), so these are the only
# genuine constants. Context size and GPU-layer offload are NOT hardcoded -- they
# derive from cfg + the model's training context exactly as in-process (_role_ctx,
# _role_gpu_layers), so a user's num_ctx / n_gpu_layers and the model are honored.
_CHAT_SLOTS = 4
_AUX_SLOTS = 1
_EMBED_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK)
# Roles that offload every layer in-process regardless of cfg.n_gpu_layers: the
# embedding loader and the mtmd vision loader both hardcode all-layer offload;
# only chat honors cfg.n_gpu_layers.
_ALL_LAYER_ROLES = (WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION)
# llama-server --flash-attn values; vision matches the in-process loader's
# full-core thread default (os.cpu_count() or this floor).
_FLASH_ON = "on"
_FLASH_OFF = "off"
_DEFAULT_THREADS = 4
# User-facing name for this engine in error messages.
_PROVIDER_NAME = "llama-server"

# Server roles -> (slots, model-ref accessor). chat/embed are always configured;
# reranker_model/vision_model may be "" (unconfigured) -> skipped, so that role
# has no server (its calls error). Vision additionally needs an mmproj projector.
_SERVER_ROLE_PARAMS: dict[WorkerRole, tuple[int, Callable[[Any], str]]] = {
    WorkerRole.CHAT: (_CHAT_SLOTS, lambda c: str(c.chat_model)),
    WorkerRole.EMBED: (_AUX_SLOTS, lambda c: str(c.embedding_model)),
    WorkerRole.RERANK: (_AUX_SLOTS, lambda c: str(c.reranker_model)),
    WorkerRole.VISION: (_AUX_SLOTS, lambda c: str(c.vision_model)),
}


def _role_ctx(role: WorkerRole, model_path: Path, meta: dict[str, str] | None) -> int:
    """Per-slot context for a role, derived exactly as the in-process loader does.

    Embed/rerank use the embedding model's training context; vision uses the
    vision loader's training-context picker (not the chat ctx); chat honors
    ``cfg.num_ctx`` then falls back to the dynamic chat-ctx picker. Never hardcoded.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.engine_params import (
        EMBED_FALLBACK_CTX,
        resolve_chat_ctx,
        resolve_vision_ctx,
    )

    if role in _EMBED_ROLES:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        return train_ctx_from_meta(meta, fallback=EMBED_FALLBACK_CTX, model_path=model_path)
    if role is WorkerRole.VISION:
        return resolve_vision_ctx(model_path)
    if cfg.num_ctx is not None:
        return cfg.num_ctx
    return resolve_chat_ctx(model_path, meta)


def _role_gpu_layers(role: WorkerRole) -> int:
    """GPU-layer offload for a role. Chat honors ``cfg.n_gpu_layers``; embed/rerank
    and vision always offload all layers, mirroring their in-process loaders."""
    from lilbee.providers.engine_params import resolve_n_gpu_layers

    return resolve_n_gpu_layers(embedding=role in _ALL_LAYER_ROLES)


def _flash_attn_flag() -> str:
    """``--flash-attn`` value for chat, mirroring the in-process loader.

    In-process forces flash attention on for chat unless ``cfg.flash_attention``
    is explicitly ``False``; ``None`` (auto) and ``True`` both enable it.
    """
    from lilbee.core.config import cfg

    return _FLASH_OFF if cfg.flash_attention is False else _FLASH_ON


def _cache_type_flag() -> str | None:
    """KV cache type string for chat, or ``None`` to leave llama-server's f16 default.

    Mirrors ``_apply_kv_cache_type``: f16 is the engine default (no flag), any
    other configured type maps to its llama.cpp type name (== the enum value).
    """
    from lilbee.core.config import cfg
    from lilbee.core.config.enums import KvCacheType

    if cfg.kv_cache_type is KvCacheType.F16:
        return None
    return cfg.kv_cache_type.value


def _least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Pick the healthy client with the fewest in-flight requests."""
    return min(clients, key=lambda c: c.in_flight)


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


class FleetProvider:
    """Routes every role to the managed llama-server fleet (a fleet-of-one on one box)."""

    def __init__(self) -> None:
        self._fleet: Fleet | None = None
        # Single-flight guard: the HTTP/MCP servers route concurrently, so two
        # first-requests must not each build a fleet (double GPU allocation) or
        # tear one down mid-route. Reentrant: invalidate_load_cache nests calls.
        self._lock = threading.RLock()
        # Spawn-lifecycle listeners (set by the TUI via add_spawn_listener). Stored
        # so they survive a fleet rebuild and attach to every fleet we construct.
        self._on_spawning: Callable[[WorkerRole], None] | None = None
        self._on_spawned: Callable[[WorkerRole], None] | None = None

    def _server_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        with self._lock:
            if self._fleet is None:
                self._fleet = _build_fleet(self._on_spawning, self._on_spawned)
            return self._fleet.healthy_clients(role)

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

    # --- inference: routed to the managed fleet, no in-process fallback ---

    @overload
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: Literal[False] = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: Literal[True],
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ClosableIterator[str]: ...

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str | ClosableIterator[str]:
        from lilbee.core.config import cfg
        from lilbee.providers.engine_params import chat_options_to_kwargs

        self._require_configured_model(model, str(cfg.chat_model), "chat")
        clients = self._require_clients(WorkerRole.CHAT)
        # Translate options exactly as the in-process path did (validate via
        # LLMOptions, num_predict -> max_tokens, drop num_ctx) so the server
        # honors the same generation settings; a raw passthrough would drop
        # num_predict and leak the load-only num_ctx.
        server_options = chat_options_to_kwargs(options) or None
        # generator satisfies ClosableIterator; client.chat isn't overloaded.
        return _least_in_flight(clients).chat(  # type: ignore[return-value]
            messages, options=server_options, stream=stream
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
        pages: list[PageText] = []
        for idx, png_bytes in rasterize_pdf(path):
            messages = build_vision_messages(OCR_PROMPT, bytes(png_bytes))
            text = _vision_call(_least_in_flight(clients), messages, per_page_timeout_s)
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

    def warm_up_pool(self) -> None:
        """Spawn the configured role servers eagerly (idempotent)."""
        with self._lock:
            if self._fleet is None:
                self._fleet = _build_fleet(self._on_spawning, self._on_spawned)

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
        devices = _resolve_devices(binary)
        by_index = {d.index: d for d in devices}
        launches = _plan_launches((role,), binary, by_index, devices)
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

    def shutdown(self) -> None:
        self._shutdown_fleet()


def _build_fleet(
    on_spawning: Callable[[WorkerRole], None] | None = None,
    on_spawned: Callable[[WorkerRole], None] | None = None,
) -> Fleet:
    """Resolve devices via the binary, plan placement, spawn and monitor the fleet."""
    from lilbee.core.config import cfg
    from lilbee.providers.multi_gpu.gpu_env import apply_fleet_gpu_env

    # Disable crash-prone Vulkan overlay layers / dual-vendor ICDs and apply any
    # cfg.gpu_devices pin before the device probe and the servers spawn (both
    # inherit this environment). This was the in-process engine's bootstrap; the
    # fleet must carry it now that the in-process path is gone.
    apply_fleet_gpu_env()
    binary = resolve_llama_server_binary()
    devices = _resolve_devices(binary)
    by_index = {d.index: d for d in devices}
    launches = _plan_launches(None, binary, by_index, devices)
    fleet = Fleet(data_dir=cfg.data_dir, on_spawning=on_spawning, on_spawned=on_spawned)
    fleet.start(launches)
    return fleet


def _resolve_devices(binary: Path) -> list[FleetDevice]:
    """Enumerate devices in the binary's index space, or the Vulkan VRAM probe.

    The binary's ``--list-devices`` is authoritative (its index space is what the
    per-server device pin uses). When it enumerates nothing, fall back to the
    Vulkan VRAM probe, which reports the same index space.
    """
    from lilbee.providers.multi_gpu.gpu_select import enumerate_gpu_vram

    devices = probe_devices(binary)
    if not devices:
        devices = [
            FleetDevice("Vulkan", idx, "", vram, vram) for idx, vram in (enumerate_gpu_vram() or [])
        ]
    return devices


def _plan_launches(
    roles: tuple[WorkerRole, ...] | None,
    binary: Path,
    by_index: dict[int, FleetDevice],
    devices: list[FleetDevice],
) -> list[InstanceLaunch]:
    """Plan placement for *roles* (``None`` = all configured) and build their launches."""
    from lilbee.core.config import cfg

    inputs, model_refs = _server_model_inputs(roles)
    placement = plan_placement(inputs, [(d.index, d.free_bytes) for d in devices])
    return [
        _launch_for(plan, model_refs[plan.role], binary, cfg.data_dir, by_index)
        for plan in placement.instances
    ]


def _server_model_inputs(
    roles: tuple[WorkerRole, ...] | None = None,
) -> tuple[list[ModelPlacementInput], dict[WorkerRole, str]]:
    """Build placement inputs for the configured server roles.

    When *roles* is given, only those roles are considered (used by per-role
    reload). Skips an unconfigured optional role (empty reranker_model/
    vision_model) and a vision model with no resolvable mmproj projector.
    """
    from lilbee.core.config import cfg

    inputs: list[ModelPlacementInput] = []
    model_refs: dict[WorkerRole, str] = {}
    for role, (slots, accessor) in _SERVER_ROLE_PARAMS.items():
        if roles is not None and role not in roles:
            continue
        ref = accessor(cfg)
        if not ref:
            continue  # unconfigured optional role -> no server
        if role is WorkerRole.VISION and _vision_mmproj(ref) is None:
            continue  # no projector -> vision can't run on a server
        inputs.append(_estimate_role(role, ref, slots=slots))
        model_refs[role] = ref
    return inputs, model_refs


def _vision_mmproj(model_ref: str) -> Path | None:
    """Resolve a vision model's mmproj sidecar, or ``None`` if absent."""
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import find_mmproj_for_model

    try:
        return find_mmproj_for_model(resolve_model_path(model_ref))
    except (ProviderError, OSError, ValueError, KeyError):
        return None


def _estimate_role(role: WorkerRole, model_ref: str, *, slots: int) -> ModelPlacementInput:
    """Estimate one role-model's VRAM from its GGUF on disk (+ mmproj for vision)."""
    from lilbee.core.config import cfg
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    path = resolve_model_path(model_ref)
    weights = path.stat().st_size
    if role is WorkerRole.VISION:
        mmproj = _vision_mmproj(model_ref)
        if mmproj is not None:
            weights += mmproj.stat().st_size
    meta = read_gguf_metadata(path)
    ctx = _role_ctx(role, path, meta)
    # Only chat passes --cache-type to its server; embed/rerank/vision run f16 KV
    # (their in-process loaders apply no KV quant), so estimate their KV at f16 to
    # match the runtime rather than the chat-tuned cfg.kv_cache_type.
    kv_type = cfg.kv_cache_type if role is WorkerRole.CHAT else KvCacheType.F16
    est = estimate_model_vram(
        weights, meta, ctx=ctx, slots=slots, kv_elem_bytes=KV_CACHE_TYPE_BYTES[kv_type]
    )
    return ModelPlacementInput(role=role, est_vram_bytes=est)


def _launch_for(
    plan: InstancePlan,
    model_ref: str,
    binary: Path,
    data_dir: Path,
    by_index: dict[int, FleetDevice],
) -> InstanceLaunch:
    """Build the launch spec (argv + device-pinning env) for one planned instance."""
    from lilbee.providers.engine_params import resolve_model_path
    from lilbee.providers.gguf_meta import read_gguf_metadata

    slots, _accessor = _SERVER_ROLE_PARAMS[plan.role]
    model_path = resolve_model_path(model_ref)
    ctx = _role_ctx(plan.role, model_path, read_gguf_metadata(model_path))
    chosen = tuple(by_index[i] for i in plan.devices)
    is_chat = plan.role is WorkerRole.CHAT
    is_vision = plan.role is WorkerRole.VISION
    mmproj = _vision_mmproj(model_ref) if is_vision else None
    argv = build_server_argv(
        binary=binary,
        spec=ROLE_SPECS[plan.role],
        model_path=model_path,
        devices=plan.devices,
        n_gpu_layers=_role_gpu_layers(plan.role),
        slots=slots,
        ctx_per_slot=ctx,
        tensor_split=plan.tensor_split,
        mmproj=mmproj,
        # Chat mirrors the in-process loader's flash-attn + KV-cache-type; embed/
        # rerank raise the batch/ubatch to the full context (server caps embeddings
        # at n_ubatch); vision matches the in-process loader's full-core threads.
        flash_attn=_flash_attn_flag() if is_chat else None,
        cache_type=_cache_type_flag() if is_chat else None,
        batch_size=ctx if plan.role in _EMBED_ROLES else None,
        threads=(os.cpu_count() or _DEFAULT_THREADS) if is_vision else None,
    )
    return InstanceLaunch(
        role=plan.role,
        argv=argv,
        # Device pinning plus, for the bundled server, the lib path that lets it
        # share llama-cpp-python's ggml/llama backend instead of carrying its own.
        env_overrides={**visible_env(chosen), **llama_server_runtime_env()},
        model=model_ref,
        # Stamp the owning lilbee pid so a concurrent instance's reaper won't
        # touch this server (only a dead parent's orphans get reaped).
        port_file=data_dir / f"llama-server-{plan.role.value}-{os.getpid()}.port",
        # Embed/rerank truncate oversize inputs to the per-slot context; chat and
        # vision do not (they handle long prompts in the engine).
        token_cap=ctx if plan.role in _EMBED_ROLES else None,
    )
