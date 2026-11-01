"""The Settings Profile tab: pick a profile, see your changes, and save, update or discard them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import DataTable, Select, Static

from lilbee.app import profiles
from lilbee.app.profiles import ActiveProfile, ChangeRow, DiscardResult, ProfileStatus
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.profile_dialogs import (
    SaveProfileDialog,
    SaveRequest,
    add_cost_column,
    credit_text,
    effect_cell,
    has_project_folder,
    run_profile_op,
    save_folders,
    start_profile_switch,
    value_text,
)
from lilbee.cli.tui.screens.profile_library import ProfileLibrary
from lilbee.cli.tui.screens.settings_widgets import user_pill
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmPill
from lilbee.core.profile_files import ProfileCatalog, ProfileFolder, ProfileStore

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

PROFILE_SELECT_ID = "profile-select"
_OWNED_FOLDERS = frozenset({ProfileFolder.PROJECT, ProfileFolder.GLOBAL})


class ProfileAction(StrEnum):
    """An action pill on the Profile tab."""

    UPDATE = "update"
    SAVE_AS = "save_as"
    DISCARD = "discard"
    REAPPLY = "reapply"
    MANAGE = "manage"


_ROW_ACTIONS = (
    ProfileAction.UPDATE,
    ProfileAction.SAVE_AS,
    ProfileAction.DISCARD,
    ProfileAction.REAPPLY,
)


@dataclass(frozen=True)
class ProfileSnapshot:
    """The active profile and your changes, every profile, and whether a project folder exists."""

    active: ActiveProfile
    catalog: ProfileCatalog
    has_project: bool


def load_snapshot() -> ProfileSnapshot:
    """Scan the profile folders and read config.toml; runs off the event loop."""
    store = ProfileStore()
    return ProfileSnapshot(
        active=profiles.active(store),
        catalog=profiles.list_profiles(store),
        has_project=has_project_folder(),
    )


class ProfileTab(Vertical):
    """The Profile tab body; ``show`` fills it from a snapshot the screen loads."""

    app: LilbeeApp  # type: ignore[assignment]

    class Stale(Message):
        """The profile state changed or a dialog closed; the screen reloads the snapshot."""

    def __init__(self) -> None:
        super().__init__(id="profile-tab")
        self._snapshot: ProfileSnapshot | None = None
        self._actions: dict[ProfileAction, Callable[[ProfileSnapshot], None]] = {
            ProfileAction.UPDATE: self._update,
            ProfileAction.SAVE_AS: self._save_as,
            ProfileAction.DISCARD: self._discard,
            ProfileAction.REAPPLY: self._reapply,
            ProfileAction.MANAGE: self._manage,
        }

    def compose(self) -> ComposeResult:
        yield Static(msg.PROFILE_TAB_TITLE, classes="setting-title")
        yield Static(msg.PROFILE_TAB_HELP, classes="setting-help")
        yield Select[str]([], id=PROFILE_SELECT_ID)
        yield Static("", id="profile-description", markup=False)
        yield Static("", id="profile-credit", markup=False)
        yield Static("", id="profile-folder", markup=False)
        yield Static("", id="profile-status-note", markup=False)
        yield Static(
            Content.assemble(msg.PROFILE_CHANGES_TITLE, "  ", user_pill()),
            classes="setting-title profile-section",
        )
        yield Static("", id="profile-changes-help", classes="setting-help", markup=False)
        table: DataTable[str | Content] = DataTable(id="profile-changes", cursor_type="none")
        table.add_columns(msg.PROFILE_COL_SETTING, msg.PROFILE_COL_YOURS, msg.PROFILE_COL_PROFILE)
        add_cost_column(table)
        yield table
        with Horizontal(id="profile-actions"):
            for action in _ROW_ACTIONS:
                yield ConfirmPill("", pill_id=f"profile-{action.value}", answer=action)
        yield Static(msg.PROFILE_ALL_TITLE, classes="setting-title profile-section")
        yield Static(msg.PROFILE_ALL_HELP, classes="setting-help")
        with Horizontal(id="profile-manage-row"):
            yield ConfirmPill(
                msg.PROFILE_MANAGE_LABEL, pill_id="profile-manage", answer=ProfileAction.MANAGE
            )

    def on_mount(self) -> None:
        self._stale()

    def show(self, snapshot: ProfileSnapshot) -> None:
        """Fill every part of the tab from *snapshot*."""
        self._snapshot = snapshot
        self._show_header(snapshot)
        self._show_changes(snapshot)
        self._show_actions(snapshot)

    def _show_header(self, snapshot: ProfileSnapshot) -> None:
        active = snapshot.active
        names = snapshot.catalog.usable_names()
        if active.name not in names:
            names = [active.name, *names]
        select = self.query_one(f"#{PROFILE_SELECT_ID}", Select)
        with select.prevent(Select.Changed):
            select.set_options((name, name) for name in names)
            select.value = active.name
        profile = active.entry.file if active.entry is not None else None
        description = profile.description if profile is not None else None
        credit = credit_text(profile) if profile is not None else ""
        folder = msg.PROFILE_FOLDER_TEXT[active.entry.folder] if active.entry is not None else ""
        note = profiles.status_note(active.status)
        for widget_id, text in (
            ("profile-description", description),
            ("profile-credit", credit),
            ("profile-folder", folder),
            ("profile-status-note", note),
        ):
            line = self.query_one(f"#{widget_id}", Static)
            line.update(text or "")
            line.display = bool(text)

    def _show_changes(self, snapshot: ProfileSnapshot) -> None:
        name = snapshot.active.name
        changes = snapshot.active.changes
        text = msg.PROFILE_CHANGES_HELP if changes else msg.PROFILE_CHANGES_NONE
        self.query_one("#profile-changes-help", Static).update(text.format(name=name))
        table = self.query_one("#profile-changes", DataTable)
        table.display = bool(changes)
        table.clear()
        table.add_rows(_change_cells(row) for row in changes)

    def _show_actions(self, snapshot: ProfileSnapshot) -> None:
        active = snapshot.active
        owned = active.entry is not None and active.entry.folder in _OWNED_FOLDERS
        current = active.status is ProfileStatus.CURRENT
        shown = {
            ProfileAction.UPDATE: owned and current and bool(active.changes),
            ProfileAction.SAVE_AS: True,
            ProfileAction.DISCARD: bool(active.changes),
            ProfileAction.REAPPLY: active.status is ProfileStatus.CHANGED,
        }
        labels = {
            ProfileAction.UPDATE: msg.PROFILE_ACTION_UPDATE.format(name=active.name),
            ProfileAction.SAVE_AS: msg.PROFILE_ACTION_SAVE_AS,
            ProfileAction.DISCARD: msg.PROFILE_ACTION_DISCARD,
            ProfileAction.REAPPLY: msg.PROFILE_ACTION_REAPPLY.format(name=active.name),
        }
        for action in _ROW_ACTIONS:
            pill = self.query_one(f"#profile-{action.value}", ConfirmPill)
            pill.update(labels[action])
            pill.display = shown[action]

    def focus_select(self) -> None:
        """Put focus on the profile dropdown."""
        self.query_one(f"#{PROFILE_SELECT_ID}", Select).focus()

    def _stale(self) -> None:
        self.post_message(self.Stale())

    @on(Select.Changed, f"#{PROFILE_SELECT_ID}")
    def _on_pick(self, event: Select.Changed) -> None:
        event.stop()
        if event.value == Select.NULL:
            # the blank entry picks no profile; show the active one again
            self._stale()
            return
        start_profile_switch(self.app, self, str(event.value), self._stale)

    @on(ConfirmPill.Picked)
    def _on_action(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        if self._snapshot is not None:
            self._actions[ProfileAction(str(event.answer))](self._snapshot)

    def _update(self, snapshot: ProfileSnapshot) -> None:
        def _done(result: profiles.SaveResult) -> None:
            self.app.publish_settings(result.absorbed)
            self.notify(msg.PROFILE_UPDATED.format(name=result.location.name))
            self.app.notify_warnings(result.warnings)
            self._stale()

        run_profile_op(self, lambda: profiles.update(ProfileStore()), _done)

    def _save_as(self, snapshot: ProfileSnapshot) -> None:
        dialog = SaveProfileDialog(
            title=msg.PROFILE_SAVE_TITLE,
            explain=msg.PROFILE_SAVE_EXPLAIN.format(
                name=snapshot.active.name, count=len(snapshot.active.changes)
            ),
            catalog=snapshot.catalog,
            folders=save_folders(snapshot.has_project),
        )
        self.app.push_screen(dialog, self._on_save_request)

    def _on_save_request(self, request: SaveRequest | None) -> None:
        if request is None:
            return

        def _done(result: profiles.SaveResult) -> None:
            self.app.publish_settings(result.absorbed)
            location = result.location
            self.notify(msg.PROFILE_SAVED.format(name=location.name, path=location.path))
            self.app.notify_warnings(result.warnings)
            self._stale()

        run_profile_op(self, lambda: profiles.save_as(request.name, request.folder), _done)

    def _discard(self, snapshot: ProfileSnapshot) -> None:
        name = snapshot.active.name

        def _done(result: DiscardResult) -> None:
            self.app.publish_settings(result.dropped)
            self.notify(msg.PROFILE_DISCARDED.format(name=name))
            self.app.notify_warnings(result.warnings)
            self._stale()
            if result.reindex_required:
                self.app.offer_rebuild(msg.PROFILE_DISCARD_REINDEX_MESSAGE.format(name=name))

        run_profile_op(self, profiles.discard, _done)

    def _reapply(self, snapshot: ProfileSnapshot) -> None:
        start_profile_switch(self.app, self, snapshot.active.name, self._stale)

    def _manage(self, snapshot: ProfileSnapshot) -> None:
        self.app.push_screen(ProfileLibrary(on_change=self._stale))


def _change_cells(row: ChangeRow) -> tuple[str | Content, ...]:
    return (row.key, value_text(row.yours), value_text(row.profile_value), effect_cell(row.effect))
