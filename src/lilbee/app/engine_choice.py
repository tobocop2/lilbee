"""Collect the engine choice for every role from the config."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from lilbee.catalog.refs import model_format
from lilbee.catalog.runnable import session_host
from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    EngineAssignment,
    EngineChoice,
    NoEngineError,
    RoleEngine,
    choose_engine,
    installed_engines,
)
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

CHOICES_KEY = "engine_choices"
_UNREADABLE_CHOICES = (
    "The saved engine choices cannot be read. Using the default engine for each role. "
    "Run lilbee engine use to choose again."
)

if TYPE_CHECKING:
    from lilbee.core.config import Config


def _parse_choices(raw: str | None) -> dict[str, str] | None:
    """The per-model engine names in the stored text, or ``None`` when the text is unreadable."""
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        return None
    return data


def stored_choices(raw: str | None) -> dict[str, str]:
    """The per-model engine names in the stored text; unreadable text reads as none."""
    return _parse_choices(raw) or {}


def _choice_for(config: Config, ref: str, role: WorkerRole | None = None) -> EngineChoice:
    """The one rule for the engine of the model *ref* under *config*, for a run or a pull."""
    stored = _parse_choices(config.engine_choices)
    chosen = choose_engine(
        fmt=model_format(ref),
        host=session_host,
        installed=installed_engines(),
        preferred=str(config.preferred_engine),
        override=(stored or {}).get(ref),
        role=role,
        model=ref,
    )
    if stored is None and role is not None:
        return replace(chosen, notice=f"{_UNREADABLE_CHOICES} {chosen.notice or ''}".strip())
    return chosen


def resolve_engines(config: Config) -> EngineAssignment:
    """Ask the rule once for the model of each role and keep every answer."""
    return EngineAssignment(
        {
            role: _choice_for(config, str(getattr(config, info.config_field)), role)
            for role, info in ROLE_REGISTRY.items()
        }
    )


def engine_for_format(fmt: ModelFormat, config: Config) -> EngineAdapter:
    """The adapter that loads *fmt* under *config*; raises ``NoEngineError`` when none does."""
    choice = choose_engine(
        fmt=fmt,
        host=session_host,
        installed=installed_engines(),
        preferred=str(config.preferred_engine),
        override=None,
    )
    return ADAPTERS[choice.require()]


def engine_for_ref(ref: str, config: Config) -> EngineAdapter:
    """The adapter for the model *ref* under *config*: the same choice a run makes."""
    return ADAPTERS[_choice_for(config, ref).require()]


@dataclass(frozen=True)
class BookReading:
    """One read of the book: the adapter of each role with an engine, the notice of each without."""

    adapters: dict[WorkerRole, EngineAdapter]
    notices: dict[WorkerRole, str]


class EngineBook:
    """The engine of each role, read from the config on every call so it is never stale."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def for_role(self, role: WorkerRole) -> RoleEngine:
        """The engine for *role* now; raises ``NoEngineError`` when none loads its model."""
        return resolve_engines(self._cfg).for_role(role)

    def read(self) -> BookReading:
        """Every role's adapter now, and the notice of each role that has no engine."""
        assignment = resolve_engines(self._cfg)
        adapters: dict[WorkerRole, EngineAdapter] = {}
        notices: dict[WorkerRole, str] = {}
        for role, choice in assignment.choices.items():
            try:
                adapters[role] = assignment.for_role(role).adapter
            except NoEngineError:
                notices[role] = choice.notice or ""
        return BookReading(adapters, notices)

    def adapters(self) -> dict[WorkerRole, EngineAdapter]:
        """The adapter of every role that has an engine now; a role with none is left out."""
        return self.read().adapters

    def for_ref(self, ref: str) -> EngineAdapter:
        """The adapter for a pull of *ref*, by the rule a run uses."""
        return engine_for_ref(ref, self._cfg)
