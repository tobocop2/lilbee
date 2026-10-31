"""Surface-agnostic use-cases for choosing the engine of a model."""

from __future__ import annotations

import json
from dataclasses import dataclass

from lilbee.app.engine_choice import CHOICES_KEY, resolve_engines, stored_choices
from lilbee.catalog.refs import model_format
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    AUTO,
    LIST_HINT,
    InstalledEngines,
    installed_engines,
)
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

PREFERRED_KEY = "preferred_engine"


class EngineRefusedError(ValueError):
    """The engine service refuses a write; the text names the problem and the fix."""


@dataclass(frozen=True)
class RoleView:
    """What a surface shows of one role: its model, the engine it runs on and why."""

    role: WorkerRole
    model: str
    format: str
    engine: str | None
    notice: str | None


def _unknown(name: str) -> EngineRefusedError:
    known = ", ".join([AUTO, *EngineName])
    return EngineRefusedError(f"No engine is named {name!r}. Choose one of: {known}.")


def _engine_name(name: str) -> EngineName:
    try:
        return EngineName(name)
    except ValueError:
        raise _unknown(name) from None


def _require_installed(name: EngineName, installed: InstalledEngines) -> None:
    if name not in installed:
        raise EngineRefusedError(f"Engine {name} is not installed. {LIST_HINT}")


def roles(config: Config = cfg) -> list[RoleView]:
    """The engine of every role, with the notice for a role that fell back or has none."""
    assignment = resolve_engines(config)
    views = []
    for role, info in ROLE_REGISTRY.items():
        ref = str(getattr(config, info.config_field))
        choice = assignment.choices[role]
        views.append(
            RoleView(
                role, ref, str(model_format(ref)), choice.name and str(choice.name), choice.notice
            )
        )
    return views


def prefer(name: str) -> str:
    """Store the global engine preference; refuses an unknown or uninstalled engine."""
    try:
        setting = EngineSetting(name)
    except ValueError:
        raise _unknown(name) from None
    if setting is not EngineSetting.AUTO:
        _require_installed(EngineName(setting.value), installed_engines())

    def apply(_raw: object) -> tuple[str, str]:
        cfg.preferred_engine = setting
        return setting.value, setting.value

    return settings.mutate_value(cfg.data_root, PREFERRED_KEY, apply)


def use(model: str, name: str) -> str:
    """Store the engine for *model*; ``auto`` removes it. Refuses an engine that cannot run it."""
    if not model.strip():
        raise EngineRefusedError("Give the model to choose an engine for.")
    if name != AUTO:
        engine = _engine_name(name)
        installed = installed_engines()
        _require_installed(engine, installed)
        fmt = model_format(model)
        if fmt not in installed[engine]:
            raise EngineRefusedError(
                f"Engine {name} cannot load the {fmt} format of {model}. "
                f"Choose an engine that loads {fmt}."
            )

    def apply(raw: object) -> tuple[str | None, str]:
        chosen = stored_choices(raw if isinstance(raw, str) else None)
        if name == AUTO:
            chosen.pop(model, None)
        else:
            chosen[model] = name
        encoded = json.dumps(chosen, sort_keys=True) if chosen else None
        cfg.engine_choices = encoded
        return encoded, name

    return settings.mutate_value(cfg.data_root, CHOICES_KEY, apply)
