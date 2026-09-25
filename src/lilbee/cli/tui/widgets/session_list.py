"""Shared session list panel: filter, select, resume, rename, delete, new.

Embedded by both the sessions drawer and the full-screen sessions view. The panel
owns everything self-contained (filtering, inline rename, delete confirmation) and
resumes or starts a chat itself. It then posts a message, so each container
decides how to leave.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import Input, ListItem, ListView, Static

from lilbee.app.services import get_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.browse_bindings import BROWSE_LIST_BINDINGS
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.sessions import (
    HUMAN_ORIGINS,
    SessionMeta,
    SessionNotFoundError,
    SessionStore,
    TitleSource,
)

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

_ROW_CSS = (Path(__file__).parent / "session_list.tcss").read_text(encoding="utf-8")


class _RowText(Static):
    """Row text that never starts a text selection.

    Rows are rebuilt on every store mutation and every filter keystroke, and
    Textual's selection path takes ``content_widget.parent`` and dereferences
    ``container.region`` with no None check, so a click landing on a row that has
    just been unparented crashed the app with AttributeError on
    ``_MessagePump__parent``. Selecting text is not something a pick-list row
    needs, and switching it off makes that path unreachable here instead of
    relying on the removal winning the race against the click.
    """

    ALLOW_SELECT = False


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
            yield _RowText(title, classes="session-row-title")
            yield _RowText(Content(self.meta.updated_at[:10]), classes="session-row-time")
        yield _RowText(Content.styled(meta_line, "$text-muted"), classes="session-row-meta")


class _FilterInput(Input):
    """The filter box. Its editing keys run in its own queue, in order with the text typed.

    Textual runs a focused Input's bindings (Backspace, ctrl+w, the arrows) only
    once the key has bubbled back to the app, which is after any row key the app
    is running. Running them here makes the box's queue hold every key typed into it.
    """

    async def _on_key(self, event: events.Key) -> None:
        """Run the box's own binding for the key; a key the box declines goes on up as before."""
        for binding in self._bindings.key_to_bindings.get(event.key, ()):
            if await self.app.run_action(binding.action, self):
                event.stop()
                return


async def _caught_up(widget: Input) -> None:
    """Return once *widget* has handled every message already queued for it.

    Also returns when the widget's queue stops, so a handler that failed on one of
    those messages lets the app exit instead of leaving this wait unanswered.
    """
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    if widget.call_later(done.set_result, None):
        await asyncio.wait((done, widget.task), return_when=asyncio.FIRST_COMPLETED)


class SessionListPanel(Vertical):
    """Filterable session list with resume / rename / delete / new actions."""

    app: LilbeeApp  # type: ignore[assignment]

    DEFAULT_CSS: ClassVar[str] = _ROW_CSS

    # Every key that moves the cursor or acts on a row is a priority binding, so
    # it runs as the app reads it, before later keys are routed. Each first waits
    # for the filter box to handle the keys typed into it ahead of the row key.
    # A focused filter still types j / k / g / G: Textual drops a binding whose
    # key the focused Input consumes.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "select", "Resume", show=False, priority=True),
        Binding("ctrl+n", "new_chat", "New", show=True, priority=True),
        Binding("ctrl+r", "rename", "Rename", show=False, priority=True),
        Binding("ctrl+d", "delete", "Delete", show=False, priority=True),
        Binding("escape", "close", "Close", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        *(replace(binding, priority=True) for binding in BROWSE_LIST_BINDINGS),
    ]

    class ChatOpened(Message):
        """A session was resumed or a new chat was started, and chat is showing."""

    class CloseRequested(Message):
        """The user asked to close the panel."""

    def __init__(self, *, focus_filter: bool = True) -> None:
        super().__init__()
        self._renaming_id: str | None = None
        # Sessions as of the last store read, and the live filter text. Reading
        # the store replays every event of every session, so it happens on mount
        # and after a mutation only; keystrokes filter this list in memory.
        self._metas: list[SessionMeta] = []
        # The rows as last rendered, in list order; the selection reads these
        # rather than the ListView children, which mount a step later.
        self._shown: list[SessionMeta] = []
        self._query = ""
        # The drawer focuses the filter for immediate type-to-switch. The
        # full-screen tab focuses the list instead, so the nav keys ([ ]) bubble
        # to the app instead of being typed into the filter.
        self._focus_filter = focus_filter

    def compose(self) -> ComposeResult:
        yield Static(id="sessions-title")
        yield _FilterInput(placeholder=msg.SESSIONS_FILTER_PLACEHOLDER, id="sessions-filter")
        yield ListView(id="sessions-list")
        yield Static(id="sessions-empty")
        yield Static(Content.styled(msg.SESSIONS_HINT, "$text-muted"), id="sessions-hint")

    def on_mount(self) -> None:
        self.refresh_list()
        target = "#sessions-filter" if self._focus_filter else "#sessions-list"
        self.screen.set_focus(self.query_one(target))

    def _store(self) -> SessionStore:
        return get_services().session_store

    def refresh_list(self) -> None:
        """Re-read the store, then render. For mount and after a mutation.

        The filter text is not a parameter: it lives in _query and survives a
        reload, so deleting a row leaves the list filtered as the user left it.
        """
        # Agent (MCP) sessions are working state, not conversations; they
        # never appear here.
        self._metas = self._store().list(origins=HUMAN_ORIGINS)
        self._render_rows()

    def _render_rows(self) -> None:
        """Render rows from the sessions already loaded. Never touches the store.

        Not ``_render``: Textual's Widget defines that as its own visual hook.
        """
        lv = self.query_one("#sessions-list", ListView)
        lv.clear()
        needle = self._query.strip().lower()
        active_id = self.app.current_session_id()
        metas = [m for m in self._metas if needle in m.title.lower()]
        self._shown = metas
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
        """The session under the cursor, or the first one while the rows are still mounting."""
        index = self.query_one("#sessions-list", ListView).index
        if not self._shown:
            return None
        return self._shown[index if index is not None and index < len(self._shown) else 0]

    @on(Input.Changed, "#sessions-filter")
    def _on_filter(self, _event: Input.Changed) -> None:
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Render for the filter box's current text, unless the rows already show it.

        Reads the box rather than a Changed event's value: a burst leaves older
        Changed events queued behind a key that has already rendered the newer text.
        """
        value = self.query_one("#sessions-filter", Input).value
        if self._renaming_id is None and value != self._query:
            self._query = value
            self._render_rows()

    async def _filter_caught_up(self) -> None:
        """Wait for the filter box to apply the keys typed ahead, then render for them.

        The panel's bindings are priority bindings, so they run before the filter
        box has handled the keys queued for it.
        """
        await _caught_up(self.query_one("#sessions-filter", Input))
        self._apply_filter()

    @on(ListView.Selected, "#sessions-list")
    def _on_row_selected(self, event: ListView.Selected) -> None:
        """Resume a clicked row; enter never gets here, the panel's own binding takes it."""
        if self._renaming_id is None:
            self._resume(event.item.meta if isinstance(event.item, SessionRow) else None)

    def _resume(self, meta: SessionMeta | None) -> None:
        """Resume *meta* within this key's step, so focus is on the prompt before the next key."""
        if meta is None:
            return
        try:
            self.app.resume_session(meta.id)
        except SessionNotFoundError:
            self.app.notify(msg.SESSIONS_GONE, severity="warning")
            self.refresh_list()
            return
        self.post_message(self.ChatOpened())

    async def action_select(self) -> None:
        """Enter: finish a rename, else resume the highlighted session."""
        await self._filter_caught_up()
        if self._renaming_id is not None:
            self._commit_rename()
            return
        self._resume(self._selected())

    async def action_cursor_down(self) -> None:
        await self._filter_caught_up()
        self.query_one("#sessions-list", ListView).action_cursor_down()

    async def action_cursor_up(self) -> None:
        await self._filter_caught_up()
        self.query_one("#sessions-list", ListView).action_cursor_up()

    async def action_jump_top(self) -> None:
        await self._filter_caught_up()
        self.jump_to(0)

    async def action_jump_bottom(self) -> None:
        await self._filter_caught_up()
        self.jump_to(-1)

    def jump_to(self, index: int) -> None:
        """Move the list cursor to *index* (negative counts from the end)."""
        count = len(self._shown)
        if not count:
            return
        lv = self.query_one("#sessions-list", ListView)
        lv.index = count - 1 if index < 0 else min(index, count - 1)

    def action_new_chat(self) -> None:
        self.app.new_chat()
        self.post_message(self.ChatOpened())

    def action_close(self) -> None:
        if self._renaming_id is not None:
            self._cancel_rename()
            return
        self.post_message(self.CloseRequested())

    async def action_rename(self) -> None:
        await self._filter_caught_up()
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

    async def action_delete(self) -> None:
        await self._filter_caught_up()
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
