"""The Engines tab of the Settings screen: the preferred engine and one row per engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.content import Content
from textual.widgets import DataTable, Select, Static

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption, EngineRow, EngineState
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import EngineCommandResult, announce
from lilbee.cli.tui.engine_install_flow import confirm_and_install
from lilbee.cli.tui.engine_mark import engine_mark, format_size
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.confirm_dialog import CancelFirstConfirmDialog, ConfirmDialog
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

ENGINES_PANE_ID = "settings-tab-engines"
TABLE_ID = "engines-table"
PREFERRED_ID = "engines-preferred"

_COLUMNS = (
    msg.ENGINES_COL_ENGINE,
    msg.ENGINES_COL_VERSION,
    msg.ENGINES_COL_BACKEND,
    msg.ENGINES_COL_SIZE,
    msg.ENGINES_COL_STATE,
    msg.ENGINES_COL_USED_BY,
    msg.ENGINES_COL_ACTION,
)
_STATE_TEXT = {
    EngineState.BUNDLED: msg.ENGINE_STATE_BUNDLED,
    EngineState.INSTALLED: msg.ENGINE_STATE_INSTALLED,
    EngineState.USER_INSTALLED: msg.ENGINE_STATE_INSTALLED,
    EngineState.NOT_INSTALLED: msg.ENGINE_STATE_NOT_INSTALLED,
}
_ACTION_TEXT = {
    EngineState.BUNDLED: "",
    EngineState.INSTALLED: msg.ENGINES_ACTION_REMOVE,
    EngineState.USER_INSTALLED: "",
    EngineState.NOT_INSTALLED: msg.ENGINES_ACTION_INSTALL,
}


def _row_key(row: EngineRow) -> str:
    return f"{row.engine}|{row.version}|{row.backend}|{row.state}"


def _size_text(row: EngineRow) -> str:
    """The size of a row: the word bundled, the size on disk or of the download, or a dash."""
    if row.state is EngineState.BUNDLED:
        return msg.ENGINE_STATE_BUNDLED
    size = row.size_bytes if row.state is EngineState.INSTALLED else row.download_bytes
    if size is not None:
        return format_size(size)
    return msg.ENGINES_NO_VALUE


def _build_label(row: EngineRow) -> str:
    return engine_service.BuildView(row.engine, row.version, row.backend).label


class EnginesPane(VerticalScroll):
    """The preferred engine, and a table of the engines with an Install or Remove action."""

    app: LilbeeApp  # type: ignore[assignment]

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._rows: dict[str, EngineRow] = {}

    def compose(self) -> ComposeResult:
        yield Static(msg.ENGINES_PREFERRED_LABEL, id="engines-preferred-label", markup=False)
        yield Select(
            [(EngineSetting.AUTO.value, EngineSetting.AUTO.value)],
            allow_blank=False,
            id=PREFERRED_ID,
        )
        yield Static(msg.ENGINES_TABLE_TITLE, id="engines-title", markup=False)
        table: DataTable[Text] = DataTable(id=TABLE_ID, cursor_type="row")
        table.add_columns(*_COLUMNS)
        yield table
        yield Static(msg.ENGINES_HINT, id="engines-hint", markup=False)

    def on_mount(self) -> None:
        self.reload()

    @work(thread=True, exclusive=True, group="engines-load")
    def reload(self) -> None:
        """Read the engine rows off the event loop, then show them."""
        try:
            rows = engine_service.engine_rows()
            offered = [str(name) for name in engine_service.offered_engines()]
        except OSError as exc:
            call_from_thread(
                self.app,
                self.app.notify,
                msg.ENGINES_LOAD_FAILED.format(error=exc),
                severity="error",
            )
            return
        call_from_thread(self, self._show, rows, offered)

    def _show(self, rows: list[EngineRow], offered: list[str]) -> None:
        self._rows = {_row_key(row): row for row in rows}
        table = self.query_one(f"#{TABLE_ID}", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*self._cells(row), key=_row_key(row))
        self._show_preferred(offered)

    def _cells(self, row: EngineRow) -> list[Text]:
        used = msg.ENGINES_USED_BY.format(roles=", ".join(row.roles)) if row.roles else ""
        return [
            self._styled(engine_mark(row.engine)),
            Text(row.version or msg.ENGINES_NO_VALUE),
            Text(row.backend or msg.ENGINES_NO_VALUE),
            Text(_size_text(row)),
            Text(_STATE_TEXT[row.state]),
            Text(used),
            Text(_ACTION_TEXT[row.state]),
        ]

    def _styled(self, content: Content) -> Text:
        """A table cell cannot resolve a theme variable, so resolve the mark color here."""
        text = Text()
        for part, style in content.render(end=""):
            text.append(part, style.rich_style)
        return text

    def _show_preferred(self, offered: list[str]) -> None:
        """Offer ``auto`` and each engine the service offers, as the preference check does."""
        current = cfg.preferred_engine.value
        names = [EngineSetting.AUTO.value, *offered]
        if current not in names:
            names.append(current)
        select = self.query_one(f"#{PREFERRED_ID}", Select)
        with self.prevent(Select.Changed):
            select.set_options((name, name) for name in names)
            select.value = current

    @on(Select.Changed, f"#{PREFERRED_ID}")
    def _on_preferred_changed(self, event: Select.Changed) -> None:
        event.stop()
        name = str(event.value)
        if name != cfg.preferred_engine.value:
            self._prefer(name)

    @work(thread=True, group="engines-prefer")
    def _prefer(self, name: str, confirmed: bool = False) -> None:
        """Store the preference off the UI thread; ask first when it can orphan the vectors."""
        try:
            engine_service.prefer(name, confirmed)
        except engine_service.EmbedConfirmRequiredError as exc:
            call_from_thread(self, self._confirm_prefer, name, exc)
            return
        except engine_service.WRITE_FAILURES as exc:
            call_from_thread(self, self._finish, EngineCommandResult(False, str(exc)))
            return
        text = msg.ENGINE_PREFER_DONE.format(engine=name)
        done = EngineCommandResult(True, text, engine_service.PREFERRED_KEY)
        call_from_thread(self, self._finish, done)

    def _finish(self, result: EngineCommandResult) -> None:
        announce(self.app, result)
        self.reload()

    def _confirm_prefer(self, name: str, asked: engine_service.EmbedConfirmRequiredError) -> None:
        dialog = CancelFirstConfirmDialog(
            msg.ENGINE_EMBED_CONFIRM_TITLE,
            msg.ENGINE_EMBED_CONFIRM_MESSAGE.format(
                library=asked.library, documents=asked.documents
            ),
        )

        def answered(yes: bool | None) -> None:
            if yes:
                self._prefer(name, True)
            else:
                self._finish(EngineCommandResult(False, msg.ENGINE_EMBED_CANCELLED))

        self.app.push_screen(dialog, answered)

    @on(DataTable.RowSelected, f"#{TABLE_ID}")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        row = self._rows[str(event.row_key.value)]
        if row.state is EngineState.NOT_INSTALLED:
            option = EngineOption(
                row.engine,
                installed=False,
                backend=row.backend,
                download_bytes=row.download_bytes,
            )
            confirm_and_install(self.app, option, self.reload, use_for_model=False)
        elif row.state is EngineState.INSTALLED:
            self._ask_remove(row)

    def _ask_remove(self, row: EngineRow) -> None:
        """Ask before the removal; when a role uses the engine, name the roles."""
        build = _build_label(row)
        title = msg.ENGINE_REMOVE_TITLE.format(build=build)

        def answered(yes: bool | None) -> None:
            if yes:
                self._remove(row)

        if row.roles:
            text = msg.ENGINE_REMOVE_IN_USE_MESSAGE.format(build=build, roles=", ".join(row.roles))
            self.app.push_screen(CancelFirstConfirmDialog(title, text), answered)
        else:
            text = msg.ENGINE_REMOVE_MESSAGE.format(build=build)
            self.app.push_screen(ConfirmDialog(title, text), answered)

    @work(thread=True, group="engines-remove")
    def _remove(self, row: EngineRow) -> None:
        """Remove the build off the UI thread; a refusal shows as a toast."""
        try:
            label = engine_service.remove_build(row.engine, row.version, row.backend)
        except engine_service.EngineRefusedError as exc:
            call_from_thread(self.app, self.app.notify, str(exc), severity="error")
            return
        call_from_thread(self.app, self.app.notify, engine_service.REMOVE_DONE.format(build=label))
        call_from_thread(self, self.reload)
