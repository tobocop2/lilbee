"""Surface-agnostic placement use-cases: inspect, preview, and set GPU placement."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.app.services import reset_services
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.providers.fleet.planning import ResolvedPlacement, resolve_placement_plan
from lilbee.providers.roles import WorkerRole

_PLACEMENT_KEY = "placement"


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
class PlacementView:
    """The full placement picture: GPUs, per-role placement, and whether manual."""

    gpus: tuple[GpuInfo, ...]
    roles: tuple[RolePlacementView, ...]
    unplaceable: tuple[WorkerRole, ...]
    manual: bool
    spec_json: str | None


def _active_spec() -> PlacementSpec | None:
    return cfg.placement


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
        replicas = (existing.replicas + 1) if existing else 1
        by_role[plan.role] = RolePlacementView(
            role=plan.role,
            model=resolved.model_refs.get(plan.role, ""),
            devices=plan.devices,
            tensor_split=plan.tensor_split or None,
            replicas=replicas,
        )
    return PlacementView(
        gpus=gpus,
        roles=tuple(by_role.values()),
        unplaceable=resolved.unplaceable_roles,
        manual=manual,
        spec_json=spec_json,
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


def set_placement(spec: PlacementSpec | None) -> PlacementView:
    """Validate, persist to config.toml, reset the fleet, and return the new view.

    Raises PlacementError before any write when the spec does not fit the hardware.
    """
    resolved = resolve_placement_plan(spec)
    if spec is None:
        settings.delete_values(cfg.data_root, [_PLACEMENT_KEY])
    else:
        settings.update_values(cfg.data_root, {_PLACEMENT_KEY: spec.to_json()})
    cfg.placement = spec
    reset_services()
    return _view(resolved, manual=spec is not None, spec_json=spec.to_json() if spec else None)
