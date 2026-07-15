"""Shared session list panel: filter, select, resume, rename, delete, new.

Embedded by both the sessions drawer and the full-screen sessions view. The panel
owns everything self-contained (filtering, inline rename, delete confirmation) and
posts messages for the actions that need navigation (resume, new chat, close), so
each container decides how to leave.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Input, ListItem, ListView, Static

from lilbee.app.services import get_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.sessions import SessionMeta, SessionStore, TitleSource

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

_ROW_CSS = (Path(__file__).parent / "session_list.tcss").read_text(encoding="utf-8")


class SessionRow(ListItem):
    """One session: dot + title with a right-aligned age, and a meta line below."""

    def __init__(self, meta: SessionMeta, *, active: bool) -> None:
        super().__init__()
        self.meta = meta
        self._active = active

    def compose(self) -> ComposeResult:
        dot = "●" if self._active else "○"
        title = Content.assemble(
            (f"{dot} ", "$success" if self._active else "$text-muted"),
            (self.meta.title, "bold" if self._active else ""),
        )
        meta_line = msg.SESSIONS_ROW_META.format(
            count=self.meta.message_count, model=self.meta.model_ref
        )
        with Horizontal(classes="session-row-head"):
            yield Static(title, classes="session-row-title")
            yield Static(Content(self.meta.updated_at[:10]), classes="session-row-time")
        yield Static(Content.styled(meta_line, "$text-muted"), classes="session-row-meta")


class SessionListPanel(Vertical):
    """Filterable session list with resume / rename / delete / new actions."""

    app: LilbeeApp  # type: ignore[assignment]

    DEFAULT_CSS: ClassVar[str] = _ROW_CSS

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+n", "new_chat", "New", show=True, priority=True),
        Binding("ctrl+r", "rename", "Rename", show=True, priority=True),
        Binding("ctrl+d", "delete", "Delete", show=True, priority=True),
        Binding("escape", "close", "Close", show=True, priority=True),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("up", "cursor_up", "Up", show=False),
    ]

    class Resumed(Message):
        """A session was chosen to resume."""

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class NewChat(Message):
        """The user asked to start a new chat."""

    class CloseRequested(Message):
        """The user asked to close the panel."""

    def __init__(self, *, focus_filter: bool = True) -> None:
        super().__init__()
        self._renaming_id: str | None = None
        # The drawer focuses the filter for immediate type-to-switch. The
        # full-screen tab focuses the list instead, so the nav keys ([ ]) bubble
        # to the app instead of being typed into the filter.
        self._focus_filter = focus_filter

    def compose(self) -> ComposeResult:
        yield Static(id="sessions-title")
        yield Input(placeholder=msg.SESSIONS_FILTER_PLACEHOLDER, id="sessions-filter")
        yield ListView(id="sessions-list")
        yield Static(id="sessions-empty")
        yield Static(Content.styled(msg.SESSIONS_HINT, "$text-muted"), id="sessions-hint")

    def on_mount(self) -> None:
        self.refresh_list()
        target = "#sessions-filter" if self._focus_filter else "#sessions-list"
        self.query_one(target).focus()

    def _store(self) -> SessionStore:
        return get_services().session_store

    def refresh_list(self, query: str = "") -> None:
        lv = self.query_one("#sessions-list", ListView)
        lv.clear()
        needle = query.strip().lower()
        active_id = self.app.current_session_id()
        metas = [m for m in self._store().list() if needle in m.title.lower()]
        for meta in metas:
            lv.append(SessionRow(meta, active=meta.id == active_id))
        if metas:
            lv.index = 0
        title = Content.assemble(
            (msg.SESSIONS_VIEW, "bold"),
            (f"   {msg.SESSIONS_COUNT.format(count=len(metas))}", "$text-muted"),
        )
        self.query_one("#sessions-title", Static).update(title)
        self.query_one("#sessions-empty", Static).update(
            Content.styled(msg.SESSIONS_EMPTY, "$text-muted") if not metas else Content("")
        )

    def _selected(self) -> SessionMeta | None:
        item = self.query_one("#sessions-list", ListView).highlighted_child
        # highlighted_child is typed ListItem | None; every row we add is a
        # SessionRow, so narrow to read its meta.
        return item.meta if isinstance(item, SessionRow) else None

    @on(Input.Changed, "#sessions-filter")
    def _on_filter(self, event: Input.Changed) -> None:
        if self._renaming_id is None:
            self.refresh_list(event.value)

    @on(Input.Submitted, "#sessions-filter")
    def _on_submit(self, _event: Input.Submitted) -> None:
        if self._renaming_id is not None:
            self._commit_rename()
            return
        selected = self._selected()
        if selected is not None:
            self.post_message(self.Resumed(selected.id))

    def action_cursor_down(self) -> None:
        self.query_one("#sessions-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#sessions-list", ListView).action_cursor_up()

    def action_new_chat(self) -> None:
        self.post_message(self.NewChat())

    def action_close(self) -> None:
        if self._renaming_id is not None:
            self._cancel_rename()
            return
        self.post_message(self.CloseRequested())

    def action_rename(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        self._renaming_id = selected.id
        field = self.query_one("#sessions-filter", Input)
        field.value = selected.title
        field.placeholder = msg.SESSIONS_RENAME_PLACEHOLDER

    def _commit_rename(self) -> None:
        field = self.query_one("#sessions-filter", Input)
        title = field.value.strip()
        if self._renaming_id is not None and title:
            self._store().set_title(self._renaming_id, title, TitleSource.CUSTOM)
        self._finish_rename()

    def _cancel_rename(self) -> None:
        self._finish_rename()

    def _finish_rename(self) -> None:
        self._renaming_id = None
        field = self.query_one("#sessions-filter", Input)
        field.value = ""
        field.placeholder = msg.SESSIONS_FILTER_PLACEHOLDER
        self.refresh_list()

    def action_delete(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        dialog = ConfirmDialog(
            msg.SESSIONS_DELETE_CONFIRM_TITLE,
            msg.SESSIONS_DELETE_CONFIRM.format(title=selected.title),
        )
        self.app.push_screen(
            dialog, lambda confirmed: self._on_delete_confirmed(selected, confirmed)
        )

    def _on_delete_confirmed(self, meta: SessionMeta, confirmed: bool | None) -> None:
        if not confirmed:
            return
        self._store().delete(meta.id)
        self.refresh_list()
        self.app.notify(msg.SESSIONS_DELETED.format(title=meta.title))
