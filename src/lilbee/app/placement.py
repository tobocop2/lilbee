"""Surface-agnostic placement use-cases: inspect, preview, and set GPU placement."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from lilbee.app.engine_choice import BookReading, EngineBook
from lilbee.app.services import Services, peek_services
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.security import FileLockUnavailableError
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.groups import class_roles
from lilbee.providers.fleet.model_memory import tight_memory_text, unknown_size_text
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.fleet.planning import (
    ResolvedPlacement,
    clear_read_device_cache,
    resolve_placement_plan,
)
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import ROLE_REGISTRY, EngineBackend, RoleTarget, WorkerRole
from lilbee.providers.warm_progress import WarmPhase, WarmProgress, is_active_warm

log = logging.getLogger(__name__)

PLACEMENT_READ_ERRORS = (PlacementError, ProviderError, OSError)
"""What reading the placement raises: a bad spec, an engine probe that fails, an unreadable file."""

_PLACEMENT_KEY = "placement"


class PlacementConfigWriteError(OSError):
    """Saving the placement to config.toml failed; the device probe is not at fault."""


# Ceiling and cadence for waiting out the post-reload chat warm: a cold
# tensor-split giant off a slow filesystem takes minutes, and the wait stops
# early when nothing is warming or the warm failed. The grace covers the gap
# between replan_fleet returning and its off-thread warm stamping a phase.
_CHAT_READY_TIMEOUT_S = 1800.0
_CHAT_READY_POLL_S = 0.5
_CHAT_READY_GRACE_S = 3.0


class PlacementRefusal(StrEnum):
    """Why a placement reads as refused: the spec waits, cannot be read, or an engine refuses."""

    WAITING = "waiting"
    UNREADABLE = "unreadable"
    REFUSED = "refused"


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
    engine: str
    # Where the role runs. ``devices`` is empty for a CPU role, which holds no card.
    target: RoleTarget = RoleTarget.GPU
    # What the planner charged one instance of the role, in bytes; None when it charged none.
    charge_bytes: int | None = None


@dataclass(frozen=True)
class SkippedRole:
    """A configured role left unplaced because its model isn't downloaded."""

    role: WorkerRole
    model: str


@dataclass(frozen=True)
class SkippedNoEngine:
    """A configured role left unplaced because no installed engine loads its model."""

    role: WorkerRole
    model: str
    notice: str


@dataclass(frozen=True)
class SizeUnknownRole:
    """A placed role whose model has no known size, with the sentence that says so."""

    role: WorkerRole
    model: str
    notice: str


@dataclass(frozen=True)
class TightRole:
    """A placed role whose estimate exceeds the memory on the card it landed on."""

    role: WorkerRole
    model: str
    shortfall_bytes: int
    notice: str


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
    # Configured roles left out because no installed engine loads their model, each
    # with the notice that says why, so a surface can say it and name the fix.
    skipped_no_engine: tuple[SkippedNoEngine, ...] = ()
    # Roles sharing one swap process: each is placed, and the roles of one class stay
    # resident together while the other classes are evicted, so only the largest class
    # sums against the card they name.
    swap_classes: tuple[frozenset[WorkerRole], ...] = ()
    # A saved spec that is not in use: this hardware does not satisfy it, no engine
    # or model is installed for it, or it places no role. The auto plan is what runs,
    # but the spec stays in config.toml and reapplies once it fits again, so a
    # surface has to say it is there rather than report placement as plain auto.
    rejected_spec_json: str | None = None
    # Roles placed on a card that cannot hold them, with the shortfall in bytes.
    # They load on demand and may fail; a view that omits this shows them as
    # comfortably placed right up until they do.
    tight: tuple[TightRole, ...] = ()
    # The backend the engine selected. A surface reports this rather than reading
    # it off ``gpus``: an empty device list is a CPU host and a failed probe
    # alike, and only UNKNOWN says which one this is.
    engine_backend: EngineBackend = EngineBackend.UNKNOWN
    # Why the saved spec is not in use here, in words a user can act on: refused by
    # the planner, or waiting for an engine or a model. The spec itself is in
    # ``rejected_spec_json`` so it can be corrected.
    refusal: str | None = None
    # Which case ``refusal`` is, so a client never reads the sentence to tell them apart.
    refusal_kind: PlacementRefusal | None = None
    # Where each pinned role runs once the auto plan replaced the saved spec; empty
    # when the spec applies or none of its roles is pinned to a card.
    pins_dropped: str = ""
    # Vendors whose hardware is in this machine but hidden from the engine by the
    # launch, so no row in ``gpus`` is a pin target for them, and the line that
    # says how to get them back.
    hidden_gpus: tuple[str, ...] = ()
    hidden_note: str | None = None
    # Roles placed on the unknown-size reserve, so a surface can say the figure is a guess.
    size_unknown: tuple[SizeUnknownRole, ...] = ()

    @property
    def co_tenants(self) -> tuple[WorkerRole, ...]:
        """Every role in the shared swap process, in role name order."""
        return tuple(sorted(class_roles(self.swap_classes), key=lambda role: role.value))


def _active_spec() -> PlacementSpec | None:
    raw = cfg.placement
    return PlacementSpec.from_json(raw) if raw else None


def _tight_role(role: WorkerRole, model: str, shortfall_bytes: int) -> TightRole:
    """A tight role with the sentence that says how much memory it lacks and what to do."""
    subject = f"the {role.value} model {model}" if model else f"the {role.value} model"
    return TightRole(
        role=role,
        model=model,
        shortfall_bytes=shortfall_bytes,
        notice=tight_memory_text(subject, shortfall_bytes),
    )


def _view(
    resolved: ResolvedPlacement,
    engines: Mapping[WorkerRole, EngineAdapter],
    *,
    manual: bool,
    spec_json: str | None,
    rejected_spec_json: str | None = None,
    refusal: str | None = None,
    refusal_kind: PlacementRefusal | None = None,
    pins_dropped: str = "",
) -> PlacementView:
    if refusal_kind is None and resolved.unlaunched_roles:
        refusal_kind = PlacementRefusal.REFUSED
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
                engine=str(engines[plan.role].name),
                target=plan.target,
                charge_bytes=resolved.charged_bytes.get(plan.role),
            )
    return PlacementView(
        gpus=gpus,
        roles=tuple(by_role.values()),
        unplaceable=resolved.unplaceable_roles,
        manual=manual,
        spec_json=spec_json,
        tight=tuple(
            _tight_role(role, resolved.model_refs.get(role, ""), shortfall)
            for role, shortfall in sorted(resolved.tight_roles.items(), key=lambda kv: kv[0].value)
        ),
        skipped_not_installed=tuple(
            SkippedRole(role=role, model=ref)
            for role, ref in resolved.skipped_not_installed.items()
        ),
        skipped_no_engine=tuple(
            SkippedNoEngine(role=role, model=skip.model, notice=skip.notice)
            for role, skip in resolved.skipped_no_engine.items()
        ),
        swap_classes=resolved.swap_classes,
        rejected_spec_json=rejected_spec_json,
        engine_backend=resolved.engine_backend,
        refusal=" ".join(
            filter(
                None,
                (
                    refusal,
                    *resolved.unlaunched_roles.values(),
                    *resolved.unplaceable_reasons.values(),
                ),
            )
        )
        or None,
        refusal_kind=refusal_kind,
        pins_dropped=pins_dropped,
        hidden_gpus=tuple(v.label for v in resolved.hidden.hidden) if resolved.hidden else (),
        hidden_note=resolved.hidden.note() if resolved.hidden else None,
        size_unknown=tuple(
            SizeUnknownRole(
                role=role,
                model=resolved.model_refs[role],
                notice=unknown_size_text(f"the {role.value} model {resolved.model_refs[role]}"),
            )
            for role in sorted(resolved.size_unknown_roles, key=lambda r: r.value)
        ),
    )


def _planning_engines() -> BookReading:
    """The engine of each role now, read from the engine book at this call."""
    return EngineBook(cfg).read()


def _resolve(
    spec: PlacementSpec | None, reading: BookReading, *, fall_back_to_auto: bool = False
) -> ResolvedPlacement:
    """Plan *spec* with the engines of one book reading, carrying its no-engine notices."""
    return resolve_placement_plan(
        spec,
        engines=reading.adapters,
        no_engine_notices=reading.notices,
        fall_back_to_auto=fall_back_to_auto,
    )


def _log_size_unknown(notices: Iterable[str]) -> None:
    """Record each sentence that says a model has no known size."""
    for notice in notices:
        log.warning("%s", notice)


SAVED_PLACEMENT_WAITS_FOR_ENGINE = (
    "The saved placement is kept but not used: no installed engine can run the models. "
    "It applies once an engine is installed."
)
SAVED_PLACEMENT_WAITS_FOR_A_MODEL = (
    "The saved placement is kept but not used: no model is installed for any role it names. "
    "It applies once a model is installed."
)
SAVED_PLACEMENT_PLACES_NO_ROLE = (
    "The saved placement is kept but not used: it places no role on this machine."
)


def get_placement() -> PlacementView:
    """The current effective placement (manual if a spec is set, else auto).

    A saved spec that no longer fits the hardware is not the effective placement:
    the fleet runs the auto plan, and this reports that rather than a manual layout
    nothing is using. A spec that drops a role (a target its engine cannot run, a
    card the machine lacks) reports the roles that run and the reason for the rest.
    Each model of unknown size is logged once per read. With no engine installed
    nothing can be planned: this reports each missing engine, and a saved spec as
    kept but not in use.
    """
    reading = _planning_engines()
    if not reading.adapters:
        return _no_engine_view(reading)
    return _logged_placement(reading)


def get_planned_placement() -> PlacementView:
    """The placement of :func:`get_placement` for a reader that needs the devices.

    Raises the planner's no-engine error when no engine is installed.
    """
    return _logged_placement(_planning_engines())


def _logged_placement(reading: BookReading) -> PlacementView:
    """The placement planned with *reading*, with each unknown size logged."""
    view = _read_placement(reading)
    _log_size_unknown(unknown.notice for unknown in view.size_unknown)
    return view


def _read_placement(reading: BookReading) -> PlacementView:
    """The placement planned with *reading*, without the size log."""
    try:
        spec = _active_spec()
        resolved = _resolve(spec, reading, fall_back_to_auto=True)
    except PlacementError as exc:
        return _view(
            _resolve(None, reading),
            reading.adapters,
            manual=False,
            spec_json=None,
            rejected_spec_json=cfg.placement,
            refusal=str(exc),
            refusal_kind=PlacementRefusal.UNREADABLE,
        )
    engines = reading.adapters
    if spec is None:
        return _view(resolved, engines, manual=False, spec_json=None)
    if not resolved.spec_applied:
        return _view(
            resolved,
            engines,
            manual=False,
            spec_json=None,
            rejected_spec_json=spec.to_json(),
            pins_dropped=resolved.pins_dropped,
        )
    if resolved.refused_roles:
        return _view(
            resolved,
            engines,
            manual=False,
            spec_json=None,
            rejected_spec_json=spec.to_json(),
            refusal=" ".join(resolved.refused_roles.values()),
            refusal_kind=PlacementRefusal.REFUSED,
        )
    return _spec_view(resolved, engines, spec)


def _spec_view(
    resolved: ResolvedPlacement, engines: Mapping[WorkerRole, EngineAdapter], spec: PlacementSpec
) -> PlacementView:
    """The view of an applied *spec*: manual when it places a role, else kept with its reason."""
    if resolved.instances:
        return _view(resolved, engines, manual=True, spec_json=spec.to_json())
    if spec.roles.keys() & resolved.skipped_not_installed.keys():
        refusal = SAVED_PLACEMENT_WAITS_FOR_A_MODEL
    else:
        refusal = SAVED_PLACEMENT_PLACES_NO_ROLE
    return _view(
        resolved,
        engines,
        manual=False,
        spec_json=None,
        rejected_spec_json=spec.to_json(),
        refusal=refusal,
        refusal_kind=PlacementRefusal.WAITING,
    )


def size_unknown_notices(role: WorkerRole) -> list[str]:
    """The sentences that say the model of *role* has no known size; none if unreadable.

    Each sentence is logged once per call, so a surface that asks for the role it
    just started logs the fact once.
    """
    try:
        view = _read_placement(_planning_engines())
    except PLACEMENT_READ_ERRORS as exc:
        log.debug("No size notice for the %s model: the placement could not be read: %s", role, exc)
        return []
    notices = [unknown.notice for unknown in view.size_unknown if unknown.role is role]
    _log_size_unknown(notices)
    return notices


def preview_placement(spec: PlacementSpec | None = None) -> PlacementView:
    """Dry-run: what spec (or auto, when None) would place. No persistence or reload."""
    reading = _planning_engines()
    resolved = _resolve(spec, reading)
    if spec is None:
        return _view(resolved, reading.adapters, manual=False, spec_json=None)
    return _spec_view(resolved, reading.adapters, spec)


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


def _apply_to_running(services: Services) -> None:
    """Apply a saved placement to the fleet; a failure says that the placement is saved."""
    try:
        services.provider.replan_fleet(wait=True)
    except (ProviderError, OSError) as exc:
        log.warning("The running server did not apply the placement: %s", exc)
        raise ProviderError(
            "The placement is saved, but the running server could not apply it: "
            f"{str(exc).rstrip('.')}. Restart lilbee to use the placement.",
            kind=exc.kind if isinstance(exc, ProviderError) else ProviderErrorKind.UNKNOWN,
        ) from exc


def set_placement(spec: PlacementSpec | None) -> PlacementView:
    """Validate, persist to config.toml, apply to the live fleet, and return the new view.

    Raises PlacementError before any write when the spec does not fit the hardware.
    The live fleet applies the change surgically (``replan_fleet`` restarts
    only the roles whose placement moved), so an untouched role's loaded model
    stays resident; with no services built there is nothing running and the next
    use plans fresh. On the live path the planner re-plans against its clean-box
    plan snapshot (see ``planning.capture_plan_probe``): probing under a loaded
    fleet would report our own residency as unavailable and poison the chat
    context sizing, while charging stays against total capacity (bb-a8f).
    """
    reading = _planning_engines()
    spec_json = None if spec is None else spec.to_json()
    resolved = _plan_for_write(spec, reading)
    _persist(spec_json)
    settings.sync_from_resolver(cfg, [_PLACEMENT_KEY])
    services = peek_services()
    if resolved is None or services is None:
        clear_read_device_cache()  # nothing can run; let the next boot probe fresh
    else:
        _apply_to_running(services)
    if resolved is None:
        running = {} if services is None else _running_notices(services)
        return _no_engine_view(reading, running)
    if cfg.placement != spec_json:
        return get_placement()  # LILBEE_PLACEMENT outranks the saved spec; report what runs
    if spec is None:
        return _view(resolved, reading.adapters, manual=False, spec_json=None)
    return _spec_view(resolved, reading.adapters, spec)


def _persist(spec_json: str | None) -> None:
    """Write the spec to config.toml, or remove it when *spec_json* is None."""
    try:
        if spec_json is None:
            settings.delete_values(cfg.data_root, [_PLACEMENT_KEY])
        else:
            settings.update_values(cfg.data_root, {_PLACEMENT_KEY: spec_json})
    except (FileLockUnavailableError, settings.ReadOnlyConfigError):
        raise  # each has its own answer on every surface
    except OSError as exc:
        raise PlacementConfigWriteError(exc.errno, exc.strerror or str(exc), exc.filename) from exc


def _plan_for_write(spec: PlacementSpec | None, reading: BookReading) -> ResolvedPlacement | None:
    """The plan for *spec*; None for a clear on a host that cannot plan (a clear needs none)."""
    try:
        return _resolve(spec, reading)
    except (ProviderError, OSError) as exc:
        if spec is not None:
            raise
        log.debug("A placement clear goes ahead without a plan: %s", exc)
        return None


def _running_notices(services: Services) -> dict[WorkerRole, str]:
    """The sentence that says a local model still runs, for each role whose server is up."""
    notices: dict[WorkerRole, str] = {}
    for role, entry in ROLE_REGISTRY.items():
        ref = str(getattr(cfg, entry.config_field))
        if ref and not parse_model_ref(ref).is_remote and services.provider.role_ready(role):
            notices[role] = (
                f"The {role.value} model {ref} is still running: the placement could not be "
                "read, so the clear is saved but not applied. Restart lilbee to apply it."
            )
    return notices


def _no_engine_view(
    reading: BookReading, running: dict[WorkerRole, str] | None = None
) -> PlacementView:
    """The view of a host whose engine cannot plan: nothing is placed, the saved spec is kept.

    *running* maps each role whose server is up to its notice; the engine book's notice wins.
    """
    notices = {**(running or {}), **reading.notices}
    rejected_spec_json: str | None = None
    refusal: str | None = None
    refusal_kind: PlacementRefusal | None = None
    try:
        spec = _active_spec()
    except PlacementError as exc:
        rejected_spec_json, refusal = cfg.placement, str(exc)
        refusal_kind = PlacementRefusal.UNREADABLE
    else:
        if spec is not None:
            rejected_spec_json, refusal = spec.to_json(), SAVED_PLACEMENT_WAITS_FOR_ENGINE
            refusal_kind = PlacementRefusal.WAITING
    return PlacementView(
        gpus=(),
        roles=(),
        unplaceable=(),
        manual=False,
        spec_json=None,
        rejected_spec_json=rejected_spec_json,
        refusal=refusal,
        refusal_kind=refusal_kind,
        skipped_no_engine=tuple(
            SkippedNoEngine(
                role=role,
                model=str(getattr(cfg, ROLE_REGISTRY[role].config_field)),
                notice=notice,
            )
            for role, notice in notices.items()
        ),
    )


def wait_chat_ready(
    timeout_s: float = _CHAT_READY_TIMEOUT_S,
    *,
    on_progress: Callable[[WarmProgress], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> bool:
    """Block while a chat warm is in flight; True once a prompt can be served.

    ``replan_fleet(wait=True)`` returns once the proxies are healthy while the
    restarted model still warms off-thread, so a chat request sent right after an
    apply reaches a model that is still loading. Callers that gate user input on the reload call
    this to hold until the model actually serves. Waits only while a warm is
    actively in flight: with no fleet, no warm, or a failed/finished warm it
    returns at once, so a change that never restarts chat cannot stall the caller.
    The brief grace covers the reload kicking its warm on a separate thread.

    ``on_progress`` receives each actively-reporting warm snapshot so the caller
    can render the load. ``should_abort`` is polled every cycle; True ends the
    wait at once, so a cancelled prompt never pins its worker thread.
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
        if should_abort is not None and should_abort():
            return False
        snapshot = provider.warm_progress()
        # A requested warm counts as in flight before it stamps a phase: the fleet
        # spawns and health-checks llama-swap first, which takes seconds.
        if is_active_warm(snapshot):
            if on_progress is not None and snapshot is not None:
                on_progress(snapshot)
            grace_deadline = time.monotonic() + _CHAT_READY_GRACE_S
        elif provider.warm_pending():
            grace_deadline = time.monotonic() + _CHAT_READY_GRACE_S
        elif time.monotonic() > grace_deadline:
            return False
        time.sleep(_CHAT_READY_POLL_S)
    return False


def request_engine_warm() -> None:
    """Kick the provider's warm-up when nothing is loaded or loading.

    ``warm_up_pool`` is idempotent (a no-op while a warm is in flight or the
    fleet is up), so a prompt sent after a failed boot warm drives a fresh
    engine start instead of bouncing for the rest of the session.
    """
    services = peek_services()
    if services is None:
        return
    services.provider.warm_up_pool()


def chat_engine_ready() -> bool:
    """Whether a chat prompt can be served right now.

    Positive readiness, not absence-of-warm: before the services container is
    built nothing is loading yet and nothing can answer, which a warm snapshot
    cannot distinguish from a finished load.
    """
    services = peek_services()
    if services is None:
        return False
    return services.provider.role_ready(WorkerRole.CHAT)


def active_chat_warm_progress() -> WarmProgress | None:
    """The chat warm snapshot while a cold load is genuinely in flight, else None.

    A surface gates interactive input on this: ``None`` covers ready, no fleet, a
    missing model, and a finished or failed warm, so nothing traps the input in a
    locked state. Non-``None`` carries the phase and byte progress to render.

    The in-process warm snapshot is checked before ``role_ready`` because the task
    bar polls this on every tick: the snapshot is a free attribute read, while
    ``role_ready`` is an HTTP probe of the engine. Ordering it this way keeps the
    probe to the seconds a load is actually in flight instead of firing forever on
    an idle TUI. The two orders return the same answer.
    """
    services = peek_services()
    if services is None:
        return None
    provider = services.provider
    snapshot = provider.warm_progress()
    if not is_active_warm(snapshot):
        return None
    return None if provider.role_ready(WorkerRole.CHAT) else snapshot


def chat_warm_error() -> str | None:
    """The failed chat warm's error text, or None when no failure is on record."""
    services = peek_services()
    if services is None:
        return None
    snapshot = services.provider.warm_progress()
    if snapshot is not None and snapshot.phase is WarmPhase.ERROR:
        return snapshot.error or ""
    return None
