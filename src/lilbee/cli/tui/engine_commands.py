"""The /engine slash command: text in, text out, over the engine service."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lilbee.app import engine_service
from lilbee.app.engine_choice import CHOICES_KEY
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

_ROLES = "roles"
ENGINE_SETTING_KEYS = frozenset({CHOICES_KEY, engine_service.PREFERRED_KEY})
_USE = "use"
_PREFER = "prefer"
_USE_ARGS = 3
_PREFER_ARGS = 1
USAGE = "Use /engine roles, /engine use ROLE MODEL ENGINE, or /engine prefer ENGINE."


@dataclass(frozen=True)
class EngineCommandResult:
    """What an /engine command did: whether it worked, the text to show, the setting it wrote."""

    ok: bool
    text: str
    changed: str | None = None


def _roles_text() -> str:
    lines = []
    for view in engine_service.roles():
        lines.append(f"{view.role}: {view.model} ({view.format}) on {view.engine or 'no engine'}")
        if view.notice:
            lines.append(view.notice)
    return "\n".join(lines)


def _use(role: WorkerRole, model: str, engine: str) -> EngineCommandResult:
    chosen = engine_service.use_for_role(role, model, engine)
    text = msg.ENGINE_USE_DONE.format(role=role, model=model, engine=chosen)
    return EngineCommandResult(True, text, CHOICES_KEY)


def _run(verb: str, words: list[str]) -> EngineCommandResult:
    if verb == _ROLES and not words:
        return EngineCommandResult(True, _roles_text())
    if verb == _USE and len(words) == _USE_ARGS:
        role, model, engine = words
        return _use(engine_service.parse_role(role), model, engine)
    if verb == _PREFER and len(words) == _PREFER_ARGS:
        chosen = engine_service.prefer(words[0])
        text = msg.ENGINE_PREFER_DONE.format(engine=chosen)
        return EngineCommandResult(True, text, engine_service.PREFERRED_KEY)
    return EngineCommandResult(False, USAGE)


def chips_text(views: list[engine_service.RoleView]) -> str:
    """One chip per role that has a model, naming the engine it runs on."""
    return "  ".join(
        msg.ENGINE_CHIP.format(role=v.role, engine=v.engine or msg.ENGINE_CHIP_NONE)
        for v in views
        if v.model.strip()
    )


def choose_engine_for_role(role: WorkerRole, engine: str) -> EngineCommandResult:
    """Store *engine* for the current model of *role*; a refusal comes back as its text."""
    model = str(getattr(cfg, ROLE_REGISTRY[role].config_field))
    try:
        return _use(role, model, engine)
    except (engine_service.EngineRefusedError, engine_service.EngineNotSavedError) as exc:
        return EngineCommandResult(False, str(exc))


def run_engine_command(args: str) -> EngineCommandResult:
    """Run the text after /engine; a refusal comes back as the service's own text."""
    try:
        words = shlex.split(args)
    except ValueError:
        return EngineCommandResult(False, USAGE)
    if not words:
        return EngineCommandResult(True, _roles_text())
    try:
        return _run(words[0], words[1:])
    except (engine_service.EngineRefusedError, engine_service.EngineNotSavedError) as exc:
        return EngineCommandResult(False, str(exc))


def announce(app: LilbeeApp, result: EngineCommandResult) -> None:
    """Show the result, and tell every subscriber that the setting it wrote changed."""
    app.notify(result.text, severity="information" if result.ok else "error")
    if result.changed is not None:
        app.settings_changed_signal.publish((result.changed, getattr(cfg, result.changed)))
