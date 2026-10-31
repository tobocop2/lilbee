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
_INSTALL = "install"
_INSTALL_ARGS = 1
YES_WORD = "--yes"
USAGE = (
    "Use /engine roles, /engine use ROLE MODEL ENGINE [--yes], /engine prefer ENGINE [--yes], "
    "or /engine install ENGINE."
)


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


def _use(role: WorkerRole, model: str, engine: str, confirmed: bool = False) -> EngineCommandResult:
    chosen = engine_service.use_for_role(role, model, engine, confirmed=confirmed)
    text = msg.ENGINE_USE_DONE.format(role=role, model=model, engine=chosen)
    return EngineCommandResult(True, text, CHOICES_KEY)


def _run(verb: str, args: list[str]) -> EngineCommandResult:
    confirmed = YES_WORD in args
    words = [word for word in args if word != YES_WORD]
    if verb == _ROLES and not words and not confirmed:
        return EngineCommandResult(True, _roles_text())
    if verb == _USE and len(words) == _USE_ARGS:
        role, model, engine = words
        return _use(engine_service.parse_role(role), model, engine, confirmed)
    if verb == _PREFER and len(words) == _PREFER_ARGS:
        chosen = engine_service.prefer(words[0], confirmed)
        text = msg.ENGINE_PREFER_DONE.format(engine=chosen)
        return EngineCommandResult(True, text, engine_service.PREFERRED_KEY)
    if verb == _INSTALL and len(words) == _INSTALL_ARGS:
        built = engine_service.install(words[0])
        label = f"{built.engine} {built.version} ({built.backend})"
        done = msg.ENGINE_INSTALL_DONE.format(build=label)
        return EngineCommandResult(True, f"{done} {built.note}" if built.note else done)
    return EngineCommandResult(False, USAGE)


def chips_text(views: list[engine_service.RoleView]) -> str:
    """One chip per role that has a model, naming the engine it runs on."""
    return "  ".join(
        msg.ENGINE_CHIP.format(role=v.role, engine=v.engine or msg.ENGINE_CHIP_NONE)
        for v in views
        if v.model.strip()
    )


def choose_engine_for_role(
    role: WorkerRole, engine: str, confirmed: bool = False
) -> EngineCommandResult:
    """Store *engine* for the current model of *role*; a refusal comes back as its text.

    A change that needs the user's confirmation raises ``EmbedConfirmRequiredError``.
    """
    model = str(getattr(cfg, ROLE_REGISTRY[role].config_field))
    try:
        return _use(role, model, engine, confirmed)
    except engine_service.EmbedConfirmRequiredError:
        raise
    except engine_service.WRITE_FAILURES as exc:
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
    except engine_service.EmbedConfirmRequiredError as exc:
        text = msg.ENGINE_CONFIRM_REFUSED.format(reason=exc, yes=YES_WORD)
        return EngineCommandResult(False, text)
    except engine_service.WRITE_FAILURES as exc:
        return EngineCommandResult(False, str(exc))


def announce(app: LilbeeApp, result: EngineCommandResult) -> None:
    """Show the result, and tell every subscriber that the setting it wrote changed."""
    app.notify(result.text, severity="information" if result.ok else "error")
    if result.changed is not None:
        app.settings_changed_signal.publish((result.changed, getattr(cfg, result.changed)))
