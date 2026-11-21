"""The /engine slash command: text in, text out, over the engine service."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.content import Content
from textual.style import Style

from lilbee.app import engine_service
from lilbee.app.engine_choice import CHOICES_KEY
from lilbee.catalog.refs import canonical_ref
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_mark import engine_mark
from lilbee.core.config import cfg
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

if TYPE_CHECKING:
    from lilbee.catalog.models import DownloadProgress
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.runtime.cancellation import CancelSignal

_ROLES = "roles"
ENGINE_SETTING_KEYS = frozenset({CHOICES_KEY, engine_service.PREFERRED_KEY})
_LIST = "list"
_REMOVE = "remove"
_REMOVE_ARGS = 3
_USE = "use"
_PREFER = "prefer"
_USE_ARGS = 3
_PREFER_ARGS = 1
_INSTALL = "install"
_INSTALL_ARGS = 1
YES_WORD = "--yes"
USAGE = (
    "Use /engine roles, /engine list, /engine remove NAME VERSION BACKEND, "
    "/engine use ROLE MODEL ENGINE [--yes], /engine prefer ENGINE [--yes], "
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


def _builds_text() -> str:
    lines = [view.label for view in engine_service.builds()] or [engine_service.NO_BUILDS]
    folder = engine_service.FOLDER_LINE.format(folder=engine_service.engines_folder())
    return "\n".join([*lines, folder])


def _use(role: WorkerRole, model: str, engine: str, confirmed: bool = False) -> EngineCommandResult:
    chosen = engine_service.use_for_role(role, model, engine, confirmed=confirmed)
    text = msg.ENGINE_USE_DONE.format(role=role, model=model, engine=chosen)
    return EngineCommandResult(True, text, CHOICES_KEY)


def _store_verb(
    verb: str,
    args: list[str],
    words: list[str],
    on_update: Callable[[DownloadProgress], None] | None = None,
    on_retry: Callable[[str], None] | None = None,
    cancel: CancelSignal | None = None,
    confirmed: bool = False,
) -> EngineCommandResult | None:
    """Run list, remove or install, the verbs that act on the engine store; None for any other."""
    if verb == _LIST and not args:
        return EngineCommandResult(True, _builds_text())
    if verb == _REMOVE and len(words) == _REMOVE_ARGS:
        name, version, backend = words
        label = engine_service.remove_build(name, version, backend, confirmed=confirmed)
        return EngineCommandResult(True, engine_service.REMOVE_DONE.format(build=label))
    if verb == _INSTALL and len(words) == _INSTALL_ARGS:
        built = engine_service.install(
            words[0], on_update=on_update, on_retry=on_retry, cancel=cancel
        )
        done = msg.ENGINE_INSTALL_DONE.format(build=built.label)
        return EngineCommandResult(True, f"{done} {built.note}" if built.note else done)
    return None


def _run(
    verb: str,
    args: list[str],
    on_update: Callable[[DownloadProgress], None] | None = None,
    on_retry: Callable[[str], None] | None = None,
    cancel: CancelSignal | None = None,
) -> EngineCommandResult:
    confirmed = YES_WORD in args
    words = [word for word in args if word != YES_WORD]
    if verb == _ROLES and not words and not confirmed:
        return EngineCommandResult(True, _roles_text())
    stored = _store_verb(verb, args, words, on_update, on_retry, cancel, confirmed)
    if stored is not None:
        return stored
    if verb == _USE and len(words) == _USE_ARGS:
        role, model, engine = words
        return _use(engine_service.parse_role(role), model, engine, confirmed)
    if verb == _PREFER and len(words) == _PREFER_ARGS:
        chosen = engine_service.prefer(words[0], confirmed)
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


def engine_line(views: list[engine_service.RoleView]) -> Content:
    """The engine line of the model bar: a mark and a name per role, each a click target.

    A click runs ``pick('<role>')`` on the widget that shows the line. A role with
    no usable engine reads as a warning instead of a name.
    """
    chips = []
    for view in views:
        if not view.model.strip():
            continue
        if view.engine:
            what = engine_mark(view.engine)
        else:
            what = Content.styled(msg.ENGINE_NEEDS, "$warning")
        chip = Content.assemble(f"{view.role.value} ", what)
        chips.append(chip.stylize(Style.from_meta({"@click": f"pick('{view.role.value}')"})))
    if not chips:
        return Content("")
    label = Content.styled(msg.ENGINE_LINE_LABEL + " ", "$text-muted")
    return Content.assemble(label, Content("  ").join(chips))


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


def choose_engine_for_model(
    model: str, engine: str, confirmed: bool = False
) -> EngineCommandResult:
    """Store *engine* for *model*, whatever role uses it; a refusal comes back as its text.

    A change that needs the user's confirmation raises ``EmbedConfirmRequiredError``.
    """
    try:
        chosen = engine_service.use(canonical_ref(model), engine, confirmed)
    except engine_service.EmbedConfirmRequiredError:
        raise
    except engine_service.WRITE_FAILURES as exc:
        return EngineCommandResult(False, str(exc))
    text = msg.ENGINE_MODEL_USE_DONE.format(model=model, engine=chosen)
    return EngineCommandResult(True, text, CHOICES_KEY)


def install_name(args: str) -> str | None:
    """The engine an /engine install command names, or None when *args* is another command."""
    try:
        words = [word for word in shlex.split(args) if word != YES_WORD]
    except ValueError:
        return None
    if len(words) == _INSTALL_ARGS + 1 and words[0] == _INSTALL:
        return words[1]
    return None


def run_engine_command(
    args: str,
    on_update: Callable[[DownloadProgress], None] | None = None,
    on_retry: Callable[[str], None] | None = None,
    cancel: CancelSignal | None = None,
) -> EngineCommandResult:
    """Run the text after /engine; a refusal comes back as the service's own text.

    *on_update* hears the download bytes of an install. *on_retry* hears each failed attempt.
    A set *cancel* ends an install.
    """
    try:
        words = shlex.split(args)
    except ValueError:
        return EngineCommandResult(False, USAGE)
    if not words:
        return EngineCommandResult(True, _roles_text())
    try:
        return _run(words[0], words[1:], on_update, on_retry, cancel)
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
