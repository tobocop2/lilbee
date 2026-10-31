"""The /engine slash command: text in, text out, over the engine service."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from lilbee.app import engine_service

_ROLES = "roles"
_USE = "use"
_PREFER = "prefer"
_USE_ARGS = 3
_PREFER_ARGS = 1
USAGE = "Use /engine roles, /engine use ROLE MODEL ENGINE, or /engine prefer ENGINE."


@dataclass(frozen=True)
class EngineCommandResult:
    """What an /engine command did: whether it worked, and the text to show."""

    ok: bool
    text: str


def _roles_text() -> str:
    lines = []
    for view in engine_service.roles():
        lines.append(f"{view.role}: {view.model} ({view.format}) on {view.engine or 'no engine'}")
        if view.notice:
            lines.append(view.notice)
    return "\n".join(lines)


def _run(verb: str, words: list[str]) -> EngineCommandResult:
    if verb == _ROLES and not words:
        return EngineCommandResult(True, _roles_text())
    if verb == _USE and len(words) == _USE_ARGS:
        role, model, engine = words
        chosen = engine_service.use_for_role(engine_service.parse_role(role), model, engine)
        return EngineCommandResult(
            True, f"{role} model {model} uses engine {chosen}. {engine_service.RESTART_NOTE}"
        )
    if verb == _PREFER and len(words) == _PREFER_ARGS:
        return EngineCommandResult(
            True,
            f"The preferred engine is {engine_service.prefer(words[0])}. "
            f"{engine_service.RESTART_NOTE}",
        )
    return EngineCommandResult(False, USAGE)


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
