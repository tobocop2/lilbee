"""Ask before an engine download, run the install as a task, then run what follows.

The engine picker, the format picker and the Settings tab share this flow. The task
shows in the task center with progress. A failed install leaves the choice as it was.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption
from lilbee.catalog.models import DownloadProgress
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import (
    EngineCommandResult,
    announce,
    choose_engine_for_model,
    choose_engine_for_role,
    publish_engines_changed,
)
from lilbee.cli.tui.engine_mark import format_size
from lilbee.cli.tui.task_queue import TaskType
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.providers.roles import WorkerRole


def confirm_text(option: EngineOption, use_for_model: bool = True) -> str:
    """The question that asks before the download of the engine of *option*.

    With *use_for_model* the question says the engine is then used for a model.
    """
    if option.download_bytes is None:
        sizeless = (
            msg.ENGINE_DOWNLOAD_MESSAGE_NO_SIZE
            if use_for_model
            else msg.ENGINE_DOWNLOAD_ONLY_MESSAGE_NO_SIZE
        )
        return sizeless.format(engine=option.name)
    sized = msg.ENGINE_DOWNLOAD_MESSAGE if use_for_model else msg.ENGINE_DOWNLOAD_ONLY_MESSAGE
    return sized.format(engine=option.name, size=format_size(option.download_bytes))


def install_then(app: LilbeeApp, name: str, then: Callable[[], object]) -> None:
    """Install the engine *name* as a task; run *then* on the UI thread when the task ends well."""

    def target(reporter: ProgressReporter) -> None:
        def on_update(progress: DownloadProgress) -> None:
            reporter.update(progress.percent, f"{name}: {progress.detail}")

        def on_retry(text: str) -> None:
            call_from_thread(app, app.notify, text, severity="warning")

        engine_service.install(name, on_update=on_update, on_retry=on_retry, cancel=reporter)

    app.task_bar.start_task(
        msg.ENGINE_TASK_NAME.format(engine=name),
        TaskType.ENGINE,
        target,
        on_success=lambda: call_from_thread(app, _installed, app, then),
    )


def _installed(app: LilbeeApp, then: Callable[[], object]) -> None:
    publish_engines_changed(app)
    then()


def choose_for_model(
    app: LilbeeApp,
    model: str,
    option: EngineOption,
    after: Callable[[], object] | None = None,
    role: WorkerRole | None = None,
) -> None:
    """Store the engine of *option* for *model*; run *after* once the choice is saved.

    With a *role*, the model is the one of that role and the role must be able to use the engine.

    An engine that is not installed is saved only after the download confirm and the
    install task end well. A refusal or a failed install leaves the choice as it was.
    """
    if option.installed:
        _save_for_model(app, model, option.name, after, role)
        return
    confirm_and_install(app, option, lambda: _save_for_model(app, model, option.name, after, role))


def _save_for_model(
    app: LilbeeApp,
    model: str,
    name: str,
    after: Callable[[], object] | None,
    role: WorkerRole | None,
    confirmed: bool = False,
) -> None:
    """Write the choice off the UI thread, ask first when it can orphan the index vectors."""

    def write() -> None:
        try:
            if role is None:
                result = choose_engine_for_model(model, name, confirmed)
            else:
                result = choose_engine_for_role(role, name, confirmed)
        except engine_service.EmbedConfirmRequiredError as exc:
            call_from_thread(app, _confirm_embed, app, model, name, exc, after, role)
            return
        call_from_thread(app, _announce_and_after, app, result, after)

    app.run_worker(write, thread=True)


def _announce_and_after(
    app: LilbeeApp, result: EngineCommandResult, after: Callable[[], object] | None
) -> None:
    announce(app, result)
    if result.ok and after is not None:
        after()


def _confirm_embed(
    app: LilbeeApp,
    model: str,
    name: str,
    asked: engine_service.EmbedConfirmRequiredError,
    after: Callable[[], object] | None,
    role: WorkerRole | None,
) -> None:
    dialog = ConfirmDialog(
        msg.ENGINE_EMBED_CONFIRM_TITLE,
        msg.ENGINE_EMBED_CONFIRM_MESSAGE.format(library=asked.library, documents=asked.documents),
    )

    def answered(yes: bool | None) -> None:
        if yes:
            _save_for_model(app, model, name, after, role, True)
        else:
            app.notify(msg.ENGINE_EMBED_CANCELLED)

    app.push_screen(dialog, answered)


def confirm_and_install(
    app: LilbeeApp,
    option: EngineOption,
    then: Callable[[], object],
    use_for_model: bool = True,
) -> None:
    """Ask to download the engine of *option*; on yes, install it as a task, then run *then*."""

    def answered(yes: bool | None) -> None:
        if yes:
            install_then(app, option.name, then)

    app.push_screen(
        ConfirmDialog(
            msg.ENGINE_DOWNLOAD_TITLE.format(engine=option.name),
            confirm_text(option, use_for_model),
        ),
        answered,
    )
