"""FleetProvider: route chat/embed to the llama-server fleet, delegate the rest.

A local-inference sibling of ``LlamaCppProvider``. On first use it plans GPU
placement, spawns the sidecar fleet, and routes chat/embed to the least-busy
healthy server for that role; every other ``LLMProvider`` method (rerank, vision,
PDF OCR, model management) delegates to an in-process ``LlamaCppProvider``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES
from lilbee.providers.multi_gpu.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.multi_gpu.binary import resolve_llama_server_binary
from lilbee.providers.multi_gpu.client import LlamaServerClient
from lilbee.providers.multi_gpu.devices import FleetDevice, probe_devices, visible_env
from lilbee.providers.multi_gpu.fleet import Fleet, InstanceLaunch
from lilbee.providers.multi_gpu.placement import (
    InstancePlan,
    ModelPlacementInput,
    estimate_model_vram,
    plan_placement,
)
from lilbee.providers.worker.transport import WorkerRole

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.providers.base import (
        ChatMessage,
        ClosableIterator,
        LLMProvider,
        OcrBackend,
        PageText,
    )
    from lilbee.providers.llama_cpp import LlamaCppProvider

# Planner-internal slot counts and context budgets (no config knobs yet).
_CHAT_SLOTS = 4
_AUX_SLOTS = 1
_DEFAULT_CHAT_CTX = 8192
_EMBED_CTX = 2048
_ALL_GPU_LAYERS = -1

# Server-capable roles -> (slots, ctx-per-slot, model-ref accessor). Both models
# are min_length>=1 config fields, so the ref is always present.
_SERVER_ROLE_PARAMS: dict[WorkerRole, tuple[int, int, Callable[[Any], str]]] = {
    WorkerRole.CHAT: (_CHAT_SLOTS, _DEFAULT_CHAT_CTX, lambda c: str(c.chat_model)),
    WorkerRole.EMBED: (_AUX_SLOTS, _EMBED_CTX, lambda c: str(c.embedding_model)),
}


def _least_in_flight(clients: list[LlamaServerClient]) -> LlamaServerClient:
    """Pick the healthy client with the fewest in-flight requests."""
    return min(clients, key=lambda c: c.in_flight)


class FleetProvider:
    """Routes chat/embed to the managed fleet; delegates everything else local."""

    def __init__(self) -> None:
        self._fleet: Fleet | None = None
        self._local: LlamaCppProvider | None = None
        # Single-flight guard: the HTTP/MCP servers route concurrently, so two
        # first-requests must not each build a fleet (double GPU allocation) or
        # tear one down mid-route. Reentrant: invalidate_load_cache nests calls.
        self._lock = threading.RLock()

    def _local_provider(self) -> LLMProvider:
        with self._lock:
            if self._local is None:
                from lilbee.providers.llama_cpp import LlamaCppProvider

                self._local = LlamaCppProvider()
            return self._local

    def _server_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        with self._lock:
            if self._fleet is None:
                self._fleet = _build_fleet()
            return self._fleet.healthy_clients(role)

    def _shutdown_fleet(self) -> None:
        with self._lock:
            if self._fleet is not None:
                self._fleet.shutdown()
                self._fleet = None

    # --- routed to the fleet (fall back to in-process if the role has no server) ---

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
        clients = self._server_clients(WorkerRole.CHAT)
        if clients:
            # generator satisfies ClosableIterator; client.chat isn't overloaded.
            return _least_in_flight(clients).chat(  # type: ignore[return-value]
                messages, options=options, stream=stream
            )
        local = self._local_provider()
        if stream:
            return local.chat(messages, stream=True, options=options, model=model)
        return local.chat(messages, stream=False, options=options, model=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        clients = self._server_clients(WorkerRole.EMBED)
        if clients:
            return _least_in_flight(clients).embed(texts)
        return self._local_provider().embed(texts)

    # --- delegated to the in-process provider ---

    def vision_ocr(
        self, png_bytes: bytes, model: str, prompt: str = "", *, timeout: float | None = None
    ) -> str:
        return self._local_provider().vision_ocr(png_bytes, model, prompt, timeout=timeout)

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
        return self._local_provider().pdf_ocr(
            path,
            backend=backend,
            model=model,
            per_page_timeout_s=per_page_timeout_s,
            quiet=quiet,
            on_progress=on_progress,
        )

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        return self._local_provider().rerank(query, candidates)

    def supports_rerank(self) -> bool:
        return self._local_provider().supports_rerank()

    def list_models(self) -> list[str]:
        return self._local_provider().list_models()

    def list_chat_models(self, provider: str) -> list[str]:
        return self._local_provider().list_chat_models(provider)

    def pull_model(self, model: str, *, on_progress: Callable[..., Any] | None = None) -> None:
        self._local_provider().pull_model(model, on_progress=on_progress)

    def show_model(self, model: str) -> dict[str, Any] | None:
        return self._local_provider().show_model(model)

    def get_capabilities(self, model: str) -> list[str]:
        return self._local_provider().get_capabilities(model)

    def warm_up_pool(self) -> None:
        self._local_provider().warm_up_pool()

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        # A model change must respawn the affected servers, so drop the fleet.
        self._shutdown_fleet()
        self._local_provider().invalidate_load_cache(model_path)

    def shutdown(self) -> None:
        self._shutdown_fleet()
        if self._local is not None:
            self._local.shutdown()


def _build_fleet() -> Fleet:
    """Resolve devices via the binary, plan placement, spawn and monitor the fleet."""
    from lilbee.core.config import cfg
    from lilbee.providers.llama_cpp.gpu_select import enumerate_gpu_vram

    binary = resolve_llama_server_binary()
    devices = probe_devices(binary)
    if not devices:
        # Fallback: the binary couldn't enumerate; use the Vulkan VRAM probe and
        # pin via the Vulkan index space (matching how the probe enumerates).
        devices = [
            FleetDevice("Vulkan", idx, "", vram, vram) for idx, vram in (enumerate_gpu_vram() or [])
        ]
    by_index = {d.index: d for d in devices}
    inputs, model_refs = _server_model_inputs()
    placement = plan_placement(inputs, [(d.index, d.free_bytes) for d in devices])
    launches = [
        _launch_for(plan, model_refs[plan.role], binary, cfg.data_dir, by_index)
        for plan in placement.instances
    ]
    fleet = Fleet(data_dir=cfg.data_dir)
    fleet.start(launches)
    return fleet


def _server_model_inputs() -> tuple[list[ModelPlacementInput], dict[WorkerRole, str]]:
    """Build placement inputs for the configured, server-capable roles."""
    from lilbee.core.config import cfg

    inputs: list[ModelPlacementInput] = []
    model_refs: dict[WorkerRole, str] = {}
    for role, (slots, ctx, accessor) in _SERVER_ROLE_PARAMS.items():
        ref = accessor(cfg)
        inputs.append(_estimate_role(role, ref, slots=slots, ctx=ctx))
        model_refs[role] = ref
    return inputs, model_refs


def _estimate_role(
    role: WorkerRole, model_ref: str, *, slots: int, ctx: int
) -> ModelPlacementInput:
    """Estimate one role-model's VRAM from its GGUF on disk."""
    from lilbee.core.config import cfg
    from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata
    from lilbee.providers.llama_cpp.provider import resolve_model_path

    path = resolve_model_path(model_ref)
    weights = path.stat().st_size
    meta = read_gguf_metadata(path)
    kv_bytes = KV_CACHE_TYPE_BYTES.get(cfg.kv_cache_type, 2)
    est = estimate_model_vram(weights, meta, ctx=ctx, slots=slots, kv_elem_bytes=kv_bytes)
    return ModelPlacementInput(role=role, est_vram_bytes=est)


def _launch_for(
    plan: InstancePlan,
    model_ref: str,
    binary: Path,
    data_dir: Path,
    by_index: dict[int, FleetDevice],
) -> InstanceLaunch:
    """Build the launch spec (argv + device-pinning env) for one planned instance."""
    from lilbee.providers.llama_cpp.provider import resolve_model_path

    slots, ctx, _accessor = _SERVER_ROLE_PARAMS[plan.role]
    chosen = tuple(by_index[i] for i in plan.devices)
    argv = build_server_argv(
        binary=binary,
        spec=ROLE_SPECS[plan.role],
        model_path=resolve_model_path(model_ref),
        devices=plan.devices,
        n_gpu_layers=_ALL_GPU_LAYERS,
        slots=slots,
        ctx_per_slot=ctx,
        tensor_split=plan.tensor_split,
    )
    return InstanceLaunch(
        role=plan.role,
        argv=argv,
        env_overrides=visible_env(chosen),
        model=model_ref,
        port_file=data_dir / f"llama-server-{plan.role.value}.port",
    )
