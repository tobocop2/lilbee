"""Surface-agnostic placement use-cases: inspect, preview, and set GPU placement."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from lilbee.app.services import peek_services
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.fleet.planning import (
    ResolvedPlacement,
    clear_read_device_cache,
    resolve_placement_plan,
)
from lilbee.providers.roles import WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress

_PLACEMENT_KEY = "placement"

# Ceiling and cadence for waiting out the post-reload chat warm: a cold
# tensor-split giant off a slow filesystem takes minutes, and the wait stops
# early when nothing is warming or the warm failed. The grace covers the gap
# between reload_placement returning and its off-thread warm stamping a phase.
_CHAT_READY_TIMEOUT_S = 1800.0
_CHAT_READY_POLL_S = 0.5
_CHAT_READY_GRACE_S = 3.0
_ACTIVE_WARM_PHASES = frozenset(
    {WarmPhase.STARTING, WarmPhase.READING_WEIGHTS, WarmPhase.LOADING_ENGINE}
)


@dataclass(frozen=True)
class GpuInfo:
    """One detected GPU as a surface can render it."""

    index: int
    backend: str
    label: str
    name: str
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class RolePlacementView:
    """Where one role's model is placed in the resolved plan."""

    role: WorkerRole
    model: str
    devices: tuple[int, ...]
    tensor_split: tuple[int, ...] | None
    replicas: int


@dataclass(frozen=True)
class SkippedRole:
    """A configured role left unplaced because its model isn't downloaded."""

    role: WorkerRole
    model: str


@dataclass(frozen=True)
class PlacementView:
    """The full placement picture: GPUs, per-role placement, and whether manual."""

    gpus: tuple[GpuInfo, ...]
    roles: tuple[RolePlacementView, ...]
    unplaceable: tuple[WorkerRole, ...]
    manual: bool
    spec_json: str | None
    # Configured roles absent from the plan because their model isn't installed,
    # so a surface can show "not downloaded" instead of an unexplained empty table.
    skipped_not_installed: tuple[SkippedRole, ...] = ()
    # Roles sharing one swap group: each is placed, but only one is resident at a
    # time, so their footprints do not sum against the card they name.
    co_tenants: tuple[WorkerRole, ...] = ()


def _active_spec() -> PlacementSpec | None:
    raw = cfg.placement
    return PlacementSpec.from_json(raw) if raw else None


def _view(resolved: ResolvedPlacement, *, manual: bool, spec_json: str | None) -> PlacementView:
    gpus = tuple(
        GpuInfo(
            index=d.index,
            backend=d.backend,
            label=f"{d.backend}{d.index}",
            name=d.name,
            total_bytes=d.total_bytes,
            free_bytes=d.free_bytes,
        )
        for d in resolved.devices
    )
    by_role: dict[WorkerRole, RolePlacementView] = {}
    for plan in resolved.instances:
        existing = by_role.get(plan.role)
        if existing is not None:
            devices = tuple(sorted(set(existing.devices) | set(plan.devices)))
            by_role[plan.role] = replace(existing, devices=devices, replicas=existing.replicas + 1)
        else:
            by_role[plan.role] = RolePlacementView(
                role=plan.role,
                model=resolved.model_refs.get(plan.role, ""),
                devices=plan.devices,
                tensor_split=plan.tensor_split or None,
                replicas=1,
            )
    return PlacementView(
        gpus=gpus,
        roles=tuple(by_role.values()),
        unplaceable=resolved.unplaceable_roles,
        manual=manual,
        spec_json=spec_json,
        skipped_not_installed=tuple(
            SkippedRole(role=role, model=ref)
            for role, ref in resolved.skipped_not_installed.items()
        ),
        co_tenants=tuple(sorted(resolved.co_tenants, key=lambda role: role.value)),
    )


def get_placement() -> PlacementView:
    """The current effective placement (manual if a spec is set, else auto)."""
    spec = _active_spec()
    resolved = resolve_placement_plan(spec)
    return _view(resolved, manual=spec is not None, spec_json=spec.to_json() if spec else None)


def preview_placement(spec: PlacementSpec | None = None) -> PlacementView:
    """Dry-run: what spec (or auto, when None) would place. No persistence or reload."""
    resolved = resolve_placement_plan(spec)
    return _view(resolved, manual=spec is not None, spec_json=spec.to_json() if spec else None)


def placement_refused_message() -> str:
    """Shared refusal for placement changes on the shared HTTP server.

    Kept in one place so the REST routes and the HTTP-mounted MCP tools
    cannot drift apart.
    """
    return (
        "Changing placement on the HTTP server is unavailable: it rebuilds the shared "
        "fleet for every connected client. Enable allow_http_placement "
        "(LILBEE_ALLOW_HTTP_PLACEMENT) on a single-client deployment, or change it "
        "from the CLI or TUI."
    )


def set_placement(spec: PlacementSpec | None) -> PlacementView:
    """Validate, persist to config.toml, apply to the live fleet, and return the new view.

    Raises PlacementError before any write when the spec does not fit the hardware.
    The live fleet applies the change surgically (``reload_placement`` restarts
    only the roles whose placement moved), so an untouched role's loaded model
    stays resident; with no services built there is nothing running and the next
    use plans fresh. On the live path the planner re-plans against its clean-box
    plan snapshot (see ``planning.capture_plan_probe``): probing under a loaded
    fleet would report our own residency as unavailable and poison the chat
    context sizing, while charging stays against total capacity (bb-a8f).
    """
    resolved = resolve_placement_plan(spec)
    if spec is None:
        settings.delete_values(cfg.data_root, [_PLACEMENT_KEY])
        cfg.placement = None
    else:
        spec_json = spec.to_json()
        settings.update_values(cfg.data_root, {_PLACEMENT_KEY: spec_json})
        cfg.placement = spec_json
    services = peek_services()
    if services is None:
        clear_read_device_cache()  # nothing running; let the next boot probe fresh
    else:
        services.provider.reload_placement(wait=True)
    return _view(resolved, manual=spec is not None, spec_json=spec.to_json() if spec else None)


def wait_chat_ready(timeout_s: float = _CHAT_READY_TIMEOUT_S) -> bool:
    """Block while a chat warm is in flight after a placement change; True when ready.

    ``reload_placement(wait=True)`` returns once the proxies are healthy while the
    restarted model still warms off-thread, so a chat request sent right after an
    apply hits the busy 429 path. Callers that gate user input on the reload call
    this to hold until the model actually serves. Waits only while a warm is
    actively in flight: with no fleet, no warm, or a failed/finished warm it
    returns at once, so a change that never restarts chat cannot stall the caller.
    The brief grace covers the reload kicking its warm on a separate thread.
    """
    services = peek_services()
    if services is None:
        return False
    provider = services.provider
    started = time.monotonic()
    deadline = started + timeout_s
    grace_deadline = started + _CHAT_READY_GRACE_S
    while time.monotonic() < deadline:
        if provider.role_ready(WorkerRole.CHAT):
            return True
        snapshot = provider.warm_progress()
        warm_in_flight = snapshot is not None and snapshot.phase in _ACTIVE_WARM_PHASES
        if not warm_in_flight and time.monotonic() > grace_deadline:
            return False
        time.sleep(_CHAT_READY_POLL_S)
    return False


def active_chat_warm_progress() -> WarmProgress | None:
    """The chat warm snapshot while a cold load is genuinely in flight, else None.

    A surface gates interactive input on this: ``None`` covers ready, no fleet, a
    missing model, and a finished or failed warm, so nothing traps the input in a
    locked state. Non-``None`` carries the phase and byte progress to render.
    """
    services = peek_services()
    if services is None:
        return None
    provider = services.provider
    if provider.role_ready(WorkerRole.CHAT):
        return None
    snapshot = provider.warm_progress()
    if snapshot is not None and snapshot.phase in _ACTIVE_WARM_PHASES:
        return snapshot
    return None
