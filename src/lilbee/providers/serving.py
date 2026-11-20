"""The engine and version that serve a model ref, as a log line or a status field reports them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from lilbee.app.engine_choice import EngineBook
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.providers.local_servers import local_server_for_key
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import API_PROVIDERS, route_for

if TYPE_CHECKING:
    from lilbee.providers.fleet.engines.base import EngineAdapter

PRESET_VERSION = "preset"
"""The version reported for a server preset, which has no pin."""

UNKNOWN_ENGINE = "unknown"
"""The engine name reported when a ref names none."""

UNKNOWN_VERSION = "unknown"
"""The version reported when the pin cannot be read."""

REMOTE_VERSION = "remote"
"""The version reported for a built-in hosted or local-server provider."""


PIN_LOOKUP_ERRORS = (OSError, KeyError, RuntimeError)
"""The errors an adapter can raise while it reads its pin record."""

NO_BUILD = "none"
"""The build reported when the engine has no upstream ref apart from its version."""


@dataclass(frozen=True)
class ServingEngine:
    """One engine name with its version and the upstream build ref."""

    engine: str
    version: str
    build: str = NO_BUILD
    unread: str = ""
    """Why the pin could not be read; empty when it was read."""


def log_name(model: str) -> str:
    """The model as a log line shows it: the file name of a local path, else the ref."""
    absolute = PurePosixPath(model).is_absolute() or PureWindowsPath(model).is_absolute()
    return PureWindowsPath(model).name if absolute else model


def _pinned(engine: EngineAdapter) -> ServingEngine:
    name = str(engine.name)
    try:
        pin = engine.pin_record()
    except PIN_LOOKUP_ERRORS as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return ServingEngine(name, UNKNOWN_VERSION, UNKNOWN_VERSION, reason)
    return ServingEngine(
        name, str(pin.get("version", UNKNOWN_VERSION)), str(pin.get("upstream_ref") or NO_BUILD)
    )


def served_by(model: str, engine: EngineAdapter | None, *, native: bool) -> ServingEngine:
    """The engine that serves *model* on the route taken; total, it raises for no ref."""
    if native:
        if engine is None:
            return ServingEngine(UNKNOWN_ENGINE, UNKNOWN_VERSION)
        return _pinned(engine)
    try:
        parse_model_ref(model)
    except ValueError:
        return ServingEngine(UNKNOWN_ENGINE, UNKNOWN_VERSION)
    prefix = model.split("/", 1)[0] or UNKNOWN_ENGINE
    spec = local_server_for_key(prefix)
    if prefix in API_PROVIDERS or spec is not None:
        return ServingEngine(spec.key if spec else prefix, REMOTE_VERSION)
    if route_for(prefix) is not None:
        return ServingEngine(prefix, PRESET_VERSION)
    return ServingEngine(prefix, REMOTE_VERSION)


def routes_native(model: str) -> bool:
    """True when *model* goes to the managed engine, by the rule every role shares."""
    try:
        return not parse_model_ref(model).is_remote
    except ValueError:
        return False


def role_engine(role: WorkerRole) -> EngineAdapter | None:
    """The adapter that runs *role* now; None when no engine loads its model."""
    try:
        return EngineBook(cfg).for_role(role).adapter
    except NoEngineError:
        return None


def serving_engine(model: str, engine: EngineAdapter | None) -> ServingEngine:
    """The engine that serves *model*, from the route the dispatch takes."""
    return served_by(model, engine, native=routes_native(model))


def role_serving(role: WorkerRole, model: str) -> ServingEngine:
    """The engine that serves *model* for *role* now, read from the config at this call."""
    return serving_engine(model, role_engine(role))
