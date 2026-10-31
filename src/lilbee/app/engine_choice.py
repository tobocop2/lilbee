"""Collect the engine choice for every role from the config."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

from lilbee.catalog.refs import model_format
from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    AUTO,
    EngineAssignment,
    HostFacts,
    RoleEngine,
    choose_engine,
    installed_engines,
)
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

if TYPE_CHECKING:
    from lilbee.core.config import Config


def host_facts() -> HostFacts:
    """The facts about this host that the default rule reads."""
    return HostFacts(system=platform.system().lower(), machine=platform.machine().lower())


def resolve_engines(cfg: Config) -> EngineAssignment:
    """Ask the rule once for the model of each role and keep every answer."""
    host = host_facts()
    installed = installed_engines()
    choices = {}
    for role, info in ROLE_REGISTRY.items():
        fmt = model_format(str(getattr(cfg, info.config_field)))
        choices[role] = choose_engine(
            fmt=fmt, host=host, installed=installed, preferred=AUTO, override=None, role=role
        )
    return EngineAssignment(choices)


def engine_for_format(fmt: ModelFormat) -> EngineAdapter:
    """The adapter that loads *fmt*; raises ``NoEngineError`` when none does."""
    choice = choose_engine(
        fmt=fmt, host=host_facts(), installed=installed_engines(), preferred=AUTO, override=None
    )
    return ADAPTERS[choice.require()]


def engine_for_ref(ref: str) -> EngineAdapter:
    """The adapter that loads the format of the model *ref* names."""
    return engine_for_format(model_format(ref))


class EngineBook:
    """The engine of each role, read from the config on every call so it is never stale."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def for_role(self, role: WorkerRole) -> RoleEngine:
        """The engine for *role* now; raises ``NoEngineError`` when none loads its model."""
        return resolve_engines(self._cfg).for_role(role)
