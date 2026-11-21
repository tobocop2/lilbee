"""The engine and version that serve a model ref, as a log line or a status field reports them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from lilbee.app.engine_choice import EngineBook
from lilbee.app.services import get_services
from lilbee.catalog.refs import is_bare_hf_repo
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.providers.local_servers import local_server_for_key
from lilbee.providers.model_ref import parse_model_ref, routes_to_native_gguf
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


@dataclass(frozen=True)
class ServingEngine:
    """One engine name with its version."""

    engine: str
    version: str


def log_name(model: str) -> str:
    """The model as a log line shows it: the file name of a local path, else the ref."""
    absolute = PurePosixPath(model).is_absolute() or PureWindowsPath(model).is_absolute()
    return PureWindowsPath(model).name if absolute else model


def _pin_version(engine: EngineAdapter) -> str:
    try:
        return str(engine.pin_record().get("version", UNKNOWN_VERSION))
    except OSError:
        return UNKNOWN_VERSION


def served_by(model: str, engine: EngineAdapter | None, *, native: bool) -> ServingEngine:
    """The engine that serves *model* on the route taken; total, it raises for no ref."""
    if native:
        if engine is None:
            return ServingEngine(UNKNOWN_ENGINE, UNKNOWN_VERSION)
        return ServingEngine(str(engine.name), _pin_version(engine))
    prefix = model.split("/", 1)[0] or UNKNOWN_ENGINE
    spec = local_server_for_key(prefix)
    if prefix in API_PROVIDERS or spec is not None:
        return ServingEngine(spec.key if spec else prefix, REMOTE_VERSION)
    if route_for(prefix) is not None:
        return ServingEngine(prefix, PRESET_VERSION)
    return ServingEngine(prefix, REMOTE_VERSION)


def is_native_rerank_ref(model: str, engine: EngineAdapter | None) -> bool:
    """Return True iff *model* should route to the native rerank path of *engine*.

    Two acceptance paths:

    1. :func:`routes_to_native_gguf` accepts the ref: a native GGUF shape that
       no local-server prefix (``ollama/``, ``lm_studio/``) claims, matching
       :func:`parse_model_ref`'s exemption.
    2. The bare ``<org>/<repo>`` names a repo with an installed quant and no
       server preset has the name ``<org>``.

    The model's name is deliberately not consulted. Hosted rerankers are
    usually called rerankers too (``cohere/rerank-english-v3.0``), so matching
    on the name captures them and starves the SDK backend. The registry answers
    "is this one of ours" without guessing. Non-GGUF refs without a known SDK
    prefix still raise downstream through :func:`parse_model_ref`.

    An empty registry reports nothing installed; a registry that cannot be read
    raises, and the fault surfaces to the caller.
    """
    if not model or engine is None or WorkerRole.RERANK not in engine.served_roles:
        return False
    if routes_to_native_gguf(model):
        return True
    if not is_bare_hf_repo(model) or parse_model_ref(model).is_preset:
        return False
    return get_services().registry.installed_ref_for_repo(model) is not None


def routes_native(role: WorkerRole, model: str, engine: EngineAdapter | None) -> bool:
    """True when the request for *role* with *model* goes to the managed engine."""
    if role is WorkerRole.RERANK:
        return is_native_rerank_ref(model, engine)
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


def serving_engine(role: WorkerRole, model: str, engine: EngineAdapter | None) -> ServingEngine:
    """The engine that serves *model* for *role*, from the route the dispatch takes."""
    return served_by(model, engine, native=routes_native(role, model, engine))


def role_serving(role: WorkerRole, model: str) -> ServingEngine:
    """The engine that serves *model* for *role* now, read from the config at this call."""
    return serving_engine(role, model, role_engine(role))
