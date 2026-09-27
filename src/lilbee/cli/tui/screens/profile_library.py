"""The profile library: every profile with its folder, and the file operations on each."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar

from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import DataTable, OptionList, Static
from textual.widgets.option_list import Option

from lilbee.app import profiles
from lilbee.app.profiles import ProfileDiff
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.profile_dialogs import (
    PathRequest,
    ProfilePathDialog,
    SaveProfileDialog,
    SaveRequest,
    add_cost_column,
    credit_text,
    diff_cells,
    has_project_folder,
    run_profile_op,
    save_folders,
    start_profile_switch,
)
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.core.profile_files import ProfileCatalog, ProfileEntry, ProfileStore

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

R = TypeVar("R")

WIDE_COLUMNS = 100
NARROW_CLASS = "-narrow"
_APPLY_KEY = "enter"
_CLOSE_KEY = "esc"


class LibraryAction(StrEnum):
    """A file operation the library runs on a key."""

    DUPLICATE = "duplicate"
    RENAME = "rename"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"


@dataclass(frozen=True)
class _LibraryKey:
    key: str
    action: LibraryAction
    label: str


_KEYS = (
    _LibraryKey("d", LibraryAction.DUPLICATE, msg.PROFILE_KEY_DUPLICATE),
    _LibraryKey("r", LibraryAction.RENAME, msg.PROFILE_KEY_RENAME),
    _LibraryKey("x", LibraryAction.DELETE, msg.PROFILE_KEY_DELETE),
    _LibraryKey("e", LibraryAction.EXPORT, msg.PROFILE_KEY_EXPORT),
    _LibraryKey("i", LibraryAction.IMPORT, msg.PROFILE_KEY_IMPORT),
)


@dataclass(frozen=True)
class LibraryState:
    """Every profile file found, and whether there is a project folder to write to."""

    catalog: ProfileCatalog
    has_project: bool


def load_library() -> LibraryState:
    """Scan the profile folders; runs off the event loop."""
    return LibraryState(profiles.list_profiles(ProfileStore()), has_project_folder())


def entry_label(entry: ProfileEntry) -> Content:
    """A list row: the name, then its folder and whether it is shadowed or broken."""
    tags = [msg.PROFILE_FOLDER_TAG[entry.folder]]
    if entry.shadowed_by is not None:
        tags.append(msg.PROFILE_LIBRARY_SHADOWED)
    if entry.error is not None:
        tags.append(msg.PROFILE_LIBRARY_BROKEN.format(reason=entry.error))
    return Content.assemble(entry.name, "  ", Content.styled(", ".join(tags), "$text-muted"))


def _hints() -> Content:
    pairs = [
        (_APPLY_KEY, msg.PROFILE_KEY_APPLY),
        *((k.key, k.label) for k in _KEYS),
        (_CLOSE_KEY, msg.PROFILE_KEY_CLOSE),
    ]
    return Content("   ").join(
        Content.assemble(Content.styled(key, "$accent bold"), " ", label) for key, label in pairs
    )


def _without(catalog: ProfileCatalog, path: Path) -> ProfileCatalog:
    return replace(catalog, entries=tuple(e for e in catalog.entries if e.path != path))


class ProfileLibrary(ModalScreen[None]):
    """Lists every profile and applies, duplicates, renames, deletes, exports or imports one."""

    CSS_PATH = "profile_library.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", msg.PROFILE_KEY_CLOSE, show=False),
        *(Binding(k.key, f"run('{k.action.value}')", k.label, show=False) for k in _KEYS),
    ]

    app: LilbeeApp  # type: ignore[assignment]

    def __init__(self, *, on_change: Callable[[], None]) -> None:
        super().__init__()
        self._on_change = on_change
        self._state: LibraryState | None = None
        self._entries: list[ProfileEntry] = []
        self._credit: str | None = None
        self._handlers: dict[LibraryAction, Callable[[LibraryState], None]] = {
            LibraryAction.DUPLICATE: self._duplicate,
            LibraryAction.RENAME: self._rename,
            LibraryAction.DELETE: self._delete,
            LibraryAction.EXPORT: self._export,
            LibraryAction.IMPORT: self._import,
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="library-body"):
            yield Static(msg.PROFILE_LIBRARY_TITLE, id="library-title")
            with Horizontal(id="library-main"):
                yield OptionList(id="library-list")
                with VerticalScroll(id="library-detail"):
                    yield Static("", id="library-name", markup=False)
                    for widget_id in ("library-folder", "library-description", "library-credit"):
                        yield Static("", id=widget_id, classes="library-muted", markup=False)
                    yield Static("", id="library-problem", markup=False)
                    yield Static("", id="library-changes-title", classes="profile-section-title")
                    table: DataTable[str | Content] = DataTable(
                        id="library-changes", cursor_type="none"
                    )
                    table.add_columns(
                        msg.PROFILE_COL_SETTING, msg.PROFILE_COL_NOW, msg.PROFILE_COL_AFTER
                    )
                    add_cost_column(table)
                    yield table
            yield Static("", id="library-narrow-credit", classes="library-muted", markup=False)
            yield Static(_hints(), id="library-hints")

    def on_mount(self) -> None:
        self._fit(self.app.size.width)
        self.query_one("#library-list", OptionList).focus()
        self.reload()

    def on_resize(self, event: events.Resize) -> None:
        self._fit(event.size.width)

    def _fit(self, width: int) -> None:
        """Below WIDE_COLUMNS the library shows the list only, plus the selection's credit."""
        self.set_class(width < WIDE_COLUMNS, NARROW_CLASS)
        self._update_narrow_credit()

    @work(thread=True, exclusive=True, group="profile-library-load", exit_on_error=False)
    def reload(self) -> None:
        """Rescan the profile folders off the loop, then show the list."""
        call_from_thread(self, self.show, load_library())

    def show(self, state: LibraryState) -> None:
        """Fill the list from *state*, keeping the highlighted file when it is still there."""
        selected = self._selected()
        self._state = state
        self._entries = list(state.catalog.entries)
        listing = self.query_one("#library-list", OptionList)
        listing.set_options(Option(entry_label(entry)) for entry in self._entries)
        paths = [entry.path for entry in self._entries]
        if selected is not None and selected.path in paths:
            listing.highlighted = paths.index(selected.path)
        elif self._entries:
            listing.highlighted = 0

    def _selected(self) -> ProfileEntry | None:
        index = self.query_one("#library-list", OptionList).highlighted
        return None if index is None or index >= len(self._entries) else self._entries[index]

    @on(OptionList.OptionHighlighted, "#library-list")
    def _on_highlight(self) -> None:
        entry = self._selected()
        if entry is not None:
            self._show_detail(entry)

    def _show_detail(self, entry: ProfileEntry) -> None:
        profile = entry.file
        shadowed = entry.shadowed_by
        problem = entry.error
        if problem is None and shadowed is not None:
            problem = msg.PROFILE_LIBRARY_SHADOWED_NOTE.format(folder=shadowed.value)
        self._credit = credit_text(profile) if profile is not None else None
        for widget_id, text in (
            ("library-name", entry.name),
            ("library-folder", msg.PROFILE_FOLDER_TEXT[entry.folder]),
            ("library-description", profile.description if profile is not None else None),
            ("library-credit", self._credit),
            ("library-problem", problem),
        ):
            line = self.query_one(f"#{widget_id}", Static)
            line.update(text or "")
            line.display = bool(text)
        self._update_narrow_credit()
        self._show_diff(entry.path, None)
        if profile is not None and self._reaches(entry):
            self._load_diff(entry)

    def _update_narrow_credit(self) -> None:
        """Mirror the selection's credit outside the detail pane, for the list-only layout."""
        widget = self.query_one("#library-narrow-credit", Static)
        widget.update(self._credit or "")
        widget.display = bool(self._credit) and self.has_class(NARROW_CLASS)

    @work(thread=True, exclusive=True, group="profile-library-diff", exit_on_error=False)
    def _load_diff(self, entry: ProfileEntry) -> None:
        try:
            diff = profiles.diff(ProfileStore(), entry.name)
        except (ValueError, OSError):
            return
        call_from_thread(self, self._show_diff, entry.path, diff)

    def _show_diff(self, path: Path, diff: ProfileDiff | None) -> None:
        """Show *diff* for the file at *path*; None, or a file no longer highlighted, hides it."""
        selected = self._selected()
        current = diff if selected is not None and selected.path == path else None
        title = self.query_one("#library-changes-title", Static)
        table = self.query_one("#library-changes", DataTable)
        title.display = current is not None
        table.display = current is not None and bool(current.changes)
        if current is None:
            return
        changes = current.changes
        title.update(msg.PROFILE_LIBRARY_CHANGES if changes else msg.PROFILE_LIBRARY_NO_CHANGES)
        table.clear()
        table.add_rows(diff_cells(row) for row in changes)

    def _reaches(self, entry: ProfileEntry) -> bool:
        """Whether the name of *entry* picks its own file; operations act by name."""
        found = self._state.catalog.find(entry.name) if self._state is not None else None
        return found is not None and found.path == entry.path

    def _target(self) -> ProfileEntry | None:
        """The highlighted entry, or None with a toast when its name picks another file."""
        entry = self._selected()
        if entry is None or self._reaches(entry):
            return entry
        self.notify(
            msg.PROFILE_LIBRARY_UNREACHABLE.format(name=entry.name, path=entry.path),
            severity="error",
        )
        return None

    def _changed(self) -> None:
        self.reload()
        self._on_change()

    def action_run(self, action: str) -> None:
        """Run the file operation a key stands for."""
        if self._state is not None:
            self._handlers[LibraryAction(action)](self._state)

    def action_close(self) -> None:
        self.dismiss(None)

    @on(OptionList.OptionSelected, "#library-list")
    def _on_apply(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        entry = self._target()
        if entry is not None:
            start_profile_switch(self.app, self, entry.name, self._changed)

    def _duplicate(self, state: LibraryState) -> None:
        entry = self._target()
        if entry is None:
            return
        dialog = SaveProfileDialog(
            title=msg.PROFILE_DUPLICATE_TITLE.format(name=entry.name),
            explain=msg.PROFILE_DUPLICATE_EXPLAIN.format(name=entry.name),
            catalog=state.catalog,
            folders=save_folders(state.has_project),
            initial=msg.PROFILE_DUPLICATE_INITIAL.format(name=entry.name),
        )

        def _write(request: SaveRequest) -> profiles.ProfileLocation:
            return profiles.duplicate(ProfileStore(), entry.name, request.name, request.folder)

        def _done(location: profiles.ProfileLocation) -> None:
            self.notify(msg.PROFILE_SAVED.format(name=location.name, path=location.path))
            self._changed()

        self._ask(dialog, _write, _done)

    def _rename(self, state: LibraryState) -> None:
        entry = self._target()
        if entry is None:
            return

        def _ask_name(owned: ProfileEntry) -> None:
            dialog = SaveProfileDialog(
                title=msg.PROFILE_RENAME_TITLE.format(name=owned.name),
                explain=msg.PROFILE_RENAME_EXPLAIN,
                catalog=_without(state.catalog, owned.path),
                folders=(owned.folder,),
                initial=owned.name,
            )
            self._ask(dialog, _write, _done)

        def _write(request: SaveRequest) -> profiles.ProfileLocation:
            return profiles.rename(ProfileStore(), entry.name, request.name)

        def _done(location: profiles.ProfileLocation) -> None:
            self.notify(msg.PROFILE_RENAMED.format(old=entry.name, name=location.name))
            self._changed()

        run_profile_op(self, lambda: profiles.owned(ProfileStore(), entry.name), _ask_name)

    def _delete(self, state: LibraryState) -> None:
        entry = self._target()
        if entry is None:
            return

        def _confirm(owned: ProfileEntry) -> None:
            dialog = ConfirmDialog(
                msg.PROFILE_DELETE_TITLE.format(name=owned.name),
                msg.PROFILE_DELETE_MESSAGE.format(path=owned.path),
            )
            self.app.push_screen(dialog, _on_answer)

        def _on_answer(confirmed: bool | None) -> None:
            if confirmed:
                run_profile_op(self, lambda: profiles.delete(ProfileStore(), entry.name), _done)

        def _done(location: profiles.ProfileLocation) -> None:
            self.notify(msg.PROFILE_DELETED.format(path=location.path))
            self._changed()

        run_profile_op(self, lambda: profiles.owned(ProfileStore(), entry.name), _confirm)

    def _export(self, state: LibraryState) -> None:
        entry = self._target()
        if entry is None:
            return
        dialog = ProfilePathDialog(
            title=msg.PROFILE_EXPORT_TITLE.format(name=entry.name),
            explain=msg.PROFILE_EXPORT_EXPLAIN,
            action_label=msg.PROFILE_EXPORT_LABEL,
            initial=str(Path.cwd()),
        )

        def _write(request: PathRequest) -> profiles.ProfileLocation:
            return profiles.export(ProfileStore(), entry.name, request.path)

        def _done(location: profiles.ProfileLocation) -> None:
            self.notify(msg.PROFILE_EXPORTED.format(name=location.name, path=location.path))
            self._changed()

        self._ask(dialog, _write, _done)

    def _import(self, state: LibraryState) -> None:
        dialog = ProfilePathDialog(
            title=msg.PROFILE_IMPORT_TITLE,
            explain=msg.PROFILE_IMPORT_EXPLAIN,
            action_label=msg.PROFILE_IMPORT_LABEL,
            folders=save_folders(state.has_project),
        )

        def _write(request: PathRequest) -> profiles.ProfileLocation:
            assert request.folder is not None  # noqa: S101 -- import always offers a folder
            return profiles.import_profile(ProfileStore(), request.path, request.folder)

        def _done(location: profiles.ProfileLocation) -> None:
            self.notify(msg.PROFILE_IMPORTED.format(name=location.name, path=location.path))
            self._changed()

        self._ask(dialog, _write, _done)

    def _ask(
        self,
        dialog: ModalScreen[R | None],
        write: Callable[[R], profiles.ProfileLocation],
        done: Callable[[profiles.ProfileLocation], None],
    ) -> None:
        """Push *dialog*; on an answer, run *write* with it in a worker, then *done*."""

        def _on_answer(request: R | None) -> None:
            if request is not None:
                run_profile_op(self, lambda: write(request), done)

        self.app.push_screen(dialog, _on_answer)
