"""Profile dialogs: apply a profile, save the project's values as a new one, and their workers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from textual import on
from textual.app import ComposeResult, CSSPathType
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import DataTable, Input, Select, Static

from lilbee.app import profiles
from lilbee.app.profiles import DiffRow, ProfileDiff, ProfileEffect
from lilbee.app.services import get_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.settings_widgets import user_pill
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmPill
from lilbee.core.config import cfg
from lilbee.core.profile_files import (
    MAX_NAME_LENGTH,
    NameProblem,
    ProfileCatalog,
    ProfileFile,
    ProfileFolder,
    ProfileStore,
    name_problem,
    profile_folders,
)

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

T = TypeVar("T")

PROFILE_OP_GROUP = "profile-op"
DISABLED_CLASS = "-disabled"
COPY_NAME_TRIES = 100


class ApplyChoice(StrEnum):
    """How the Apply dialog closed."""

    APPLY = "apply"
    APPLY_REINDEX = "apply_reindex"
    CANCEL = "cancel"


class _DialogAnswer(StrEnum):
    SAVE = "save"
    CANCEL = "cancel"


@dataclass(frozen=True)
class SwitchPlan:
    """What applying a profile does: its file, the diff, and the files a reindex rebuilds."""

    profile: ProfileFile
    diff: ProfileDiff
    kept: tuple[tuple[str, Any], ...]
    file_count: int


@dataclass(frozen=True)
class SaveRequest:
    """The name and folder the Save-as dialog asks for."""

    name: str
    folder: ProfileFolder


@dataclass(frozen=True)
class PathRequest:
    """What the path dialog asks for: a path, and a folder when it offers one."""

    path: Path
    folder: ProfileFolder | None


def value_text(value: object) -> str:
    """A setting value as the profile surfaces show it: on/off, none, or a comma list."""
    # values are raw Config or TOML values of any setting type
    if isinstance(value, bool):
        return msg.PROFILE_VALUE_ON if value else msg.PROFILE_VALUE_OFF
    if value is None:
        return msg.PROFILE_VALUE_NONE
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value) or msg.PROFILE_VALUE_NONE
    return str(value)


def add_cost_column(table: DataTable[str | Content]) -> None:
    """Add the Cost column, sized to its longest pill; DataTable does not measure Content cells."""
    width = max(len(text) for text in msg.PROFILE_EFFECT_TEXT.values()) + 2
    table.add_column(msg.PROFILE_COL_COST, width=width)


def effect_cell(effect: ProfileEffect) -> Content:
    """The Cost cell: a loud pill for a reindex, plain text otherwise."""
    text = msg.PROFILE_EFFECT_TEXT[effect]
    return pill(text, "$error", "$text") if effect is ProfileEffect.REINDEX else Content(text)


def credit_text(profile: ProfileFile) -> str:
    """The credit line: the authors, then what the profile was tested on."""
    parts = [
        profiles.credit_line(profile.authors),
        profiles.tested_on_line(profile.tested_on, msg.PROFILE_TESTED_ON),
    ]
    return ". ".join(part for part in parts if part)


def has_project_folder() -> bool:
    """Whether the current data root has a project profile folder."""
    return ProfileFolder.PROJECT in dict(profile_folders(cfg.data_root))


def save_folders(has_project: bool) -> tuple[ProfileFolder, ...]:
    """The folders a new profile can go to, the default first."""
    if has_project:
        return (ProfileFolder.GLOBAL, ProfileFolder.PROJECT)
    return (ProfileFolder.GLOBAL,)


def _copy_candidate(name: str, number: int) -> str:
    """The copy name for *name* that fits the name limit; its base is cut to make room."""
    suffix = msg.PROFILE_DUPLICATE_INITIAL.format(name="")
    if number > 1:
        suffix = f"{suffix} {number}"
    return name[: MAX_NAME_LENGTH - len(suffix)].rstrip() + suffix


def free_copy_name(name: str, folder: ProfileFolder, catalog: ProfileCatalog) -> str:
    """The first copy name *folder* accepts, or the last one tried after COPY_NAME_TRIES."""
    candidate = _copy_candidate(name, 1)
    for number in range(2, COPY_NAME_TRIES + 1):
        if name_problem(candidate, folder, catalog) is None:
            break
        candidate = _copy_candidate(name, number)
    return candidate


def _folder_choice(select_id: str, folders: Sequence[ProfileFolder]) -> ComposeResult:
    """The "Save to" Select; nothing when there is no choice to make."""
    if len(folders) <= 1:
        return
    yield Static(msg.PROFILE_SAVE_TO, classes="profile-section-title")
    options = [(msg.PROFILE_SAVE_FOLDER_TEXT[folder], folder) for folder in folders]
    yield Select(options, value=next(iter(folders)), allow_blank=False, id=select_id)


def _chosen_folder(
    screen: Screen[Any], select_id: str, folders: Sequence[ProfileFolder]
) -> ProfileFolder:
    default, *others = folders
    if not others:
        return default
    return ProfileFolder(str(screen.query_one(f"#{select_id}", Select).value))


def run_profile_op(
    node: Widget,
    op: Callable[[], T],
    on_done: Callable[[T], None],
    on_error: Callable[[], None] | None = None,
) -> None:
    """Run *op* off the loop; on failure toast and call *on_error*, else call *on_done*."""

    def _work() -> None:
        try:
            result = op()
        except ValueError as exc:
            call_from_thread(node, node.notify, str(exc), severity="error")
            if on_error is not None:
                call_from_thread(node, on_error)
            return
        except OSError as exc:
            call_from_thread(
                node, node.notify, profiles.file_failure_message(exc), severity="error"
            )
            if on_error is not None:
                call_from_thread(node, on_error)
            return
        call_from_thread(node, on_done, result)

    node.run_worker(_work, thread=True, group=PROFILE_OP_GROUP, exclusive=True, exit_on_error=False)


def switch_plan(profile: ProfileFile, diff: ProfileDiff) -> SwitchPlan:
    """The Apply dialog's plan for *profile*: its diff, your kept values, the files to rebuild."""
    reindex = any(row.effect is ProfileEffect.REINDEX for row in diff.changes)
    files = len(get_services().store.get_sources()) if reindex else 0
    kept = tuple((key, getattr(cfg, key)) for key in diff.kept)
    return SwitchPlan(profile, diff, kept, files)


def _switch_plan(name: str) -> SwitchPlan:
    """Diff *name* against the project and count the files a reindex would rebuild."""
    store = ProfileStore()
    return switch_plan(profiles.usable(store, name), profiles.diff(store, name))


def start_switch(
    app: LilbeeApp,
    node: Widget,
    plan: Callable[[], SwitchPlan],
    apply: Callable[[], profiles.ApplyResult],
    *,
    on_close: Callable[[], None],
    on_applied: Callable[[], None] = lambda: None,
) -> None:
    """Build *plan* off the loop, ask with the Apply dialog, then *apply*; *on_close* runs after.

    *on_applied* runs only when the profile was applied.
    """

    def _on_choice(choice: ApplyChoice | None) -> None:
        if choice is None or choice is ApplyChoice.CANCEL:
            on_close()
            return

        def _applied(result: profiles.ApplyResult) -> None:
            app.publish_settings([row.key for row in result.changes])
            if choice is ApplyChoice.APPLY_REINDEX:
                app.start_rebuild()
            node.notify(msg.PROFILE_APPLIED.format(name=result.name))
            app.notify_warnings(result.warnings)
            on_close()
            on_applied()

        run_profile_op(node, apply, _applied)

    def _ask(switch: SwitchPlan) -> None:
        app.push_screen(ApplyProfileDialog(switch), _on_choice)

    run_profile_op(node, plan, _ask, on_error=on_close)


def start_profile_switch(
    app: LilbeeApp,
    node: Widget,
    name: str,
    on_close: Callable[[], None],
    on_applied: Callable[[], None] = lambda: None,
) -> None:
    """Diff *name* off the loop, ask with the Apply dialog, then apply; *on_close* runs after."""
    start_switch(
        app,
        node,
        lambda: _switch_plan(name),
        lambda: profiles.apply(ProfileStore(), name),
        on_close=on_close,
        on_applied=on_applied,
    )


class ApplyProfileDialog(ModalScreen[ApplyChoice]):
    """Shows what applying a profile changes and keeps; nothing changes until a pill is picked."""

    CSS_PATH: ClassVar[CSSPathType] = ["dialog.tcss", "profile_dialogs.tcss"]

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left", "step_pill(-1)", "Previous", show=False),
        Binding("right", "step_pill(1)", "Next", show=False),
    ]

    def __init__(self, plan: SwitchPlan) -> None:
        super().__init__()
        self._plan = plan
        reindex_rows = [row for row in plan.diff.changes if row.effect is ProfileEffect.REINDEX]
        self._rebuilds = len(reindex_rows) if plan.file_count else 0

    def compose(self) -> ComposeResult:
        profile = self._plan.profile
        with Vertical(id="apply-body", classes="dialog-body"):
            with VerticalScroll(id="apply-scroll"):
                yield Static(
                    msg.PROFILE_APPLY_TITLE.format(name=profile.name),
                    id="apply-title",
                    classes="dialog-title",
                    markup=False,
                )
                for text, widget_id in (
                    (profile.description, "apply-description"),
                    (credit_text(profile), "apply-credit"),
                ):
                    if text:
                        yield Static(text, id=widget_id, classes="dialog-note", markup=False)
                yield from self._compose_changes()
                yield from self._compose_kept()
                yield Static(
                    msg.PROFILE_APPLY_UNTOUCHED, id="apply-untouched", classes="dialog-note"
                )
                summary = self._summary()
                if summary:
                    yield Static(summary, id="apply-summary", markup=False)
            with Horizontal(id="apply-actions"):
                yield from self._action_pills()

    def _compose_changes(self) -> ComposeResult:
        if not self._plan.diff.changes:
            return
        yield Static(msg.PROFILE_APPLY_CHANGES, classes="profile-section-title")
        table: DataTable[str | Content] = DataTable(
            id="apply-changes", cursor_type="none", zebra_stripes=False
        )
        table.can_focus = False
        table.add_columns(msg.PROFILE_COL_SETTING, msg.PROFILE_COL_NOW, msg.PROFILE_COL_AFTER)
        add_cost_column(table)
        table.add_rows(diff_cells(row) for row in self._plan.diff.changes)
        yield table

    def _compose_kept(self) -> ComposeResult:
        if not self._plan.kept:
            return
        yield Static(msg.PROFILE_APPLY_KEEPS, classes="profile-section-title")
        kept = ", ".join(f"{key} {value_text(value)}" for key, value in self._plan.kept)
        yield Static(Content.assemble(kept, "  ", user_pill()), id="apply-kept")

    def _summary(self) -> str:
        if not self._plan.diff.changes:
            return msg.PROFILE_APPLY_NOTHING
        if not self._rebuilds:
            return ""
        return msg.profile_reindex_text(self._rebuilds, self._plan.file_count)

    def _action_pills(self) -> ComposeResult:
        if self._rebuilds:
            yield ConfirmPill(
                msg.PROFILE_APPLY_REINDEX_LABEL,
                pill_id="apply-reindex",
                answer=ApplyChoice.APPLY_REINDEX,
            )
        yield ConfirmPill(
            msg.PROFILE_APPLY_LABEL,
            pill_id="apply-apply",
            answer=ApplyChoice.APPLY,
            primary=True,
        )
        yield ConfirmPill(
            msg.PROFILE_CANCEL_LABEL, pill_id="apply-cancel", answer=ApplyChoice.CANCEL
        )

    def on_mount(self) -> None:
        self.query_one("#apply-cancel", ConfirmPill).focus()

    @on(ConfirmPill.Picked)
    def _on_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        self.dismiss(ApplyChoice(str(event.answer)))

    def action_step_pill(self, step: int) -> None:
        """Move focus *step* pills along the action row, wrapping; the scroll area is no stop."""
        pills = list(self.query("#apply-actions ConfirmPill"))
        current = pills.index(self.focused) if self.focused in pills else 0
        pills[(current + step) % len(pills)].focus()

    def action_cancel(self) -> None:
        self.dismiss(ApplyChoice.CANCEL)


def diff_cells(row: DiffRow) -> Sequence[str | Content]:
    """A diff row's cells: the setting, its value now and after, and the cost."""
    return (row.key, value_text(row.current), value_text(row.new), effect_cell(row.effect))


class SaveProfileDialog(ModalScreen[SaveRequest | None]):
    """Asks for a new profile's name and folder; Save stays off while the name is refused."""

    CSS_PATH: ClassVar[CSSPathType] = ["dialog.tcss", "profile_dialogs.tcss"]

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left", "app.focus_previous", "Previous", show=False),
        Binding("right", "app.focus_next", "Next", show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        explain: str,
        catalog: ProfileCatalog,
        folders: Sequence[ProfileFolder],
        initial: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._explain = explain
        self._catalog = catalog
        self._folders = folders
        self._initial = initial
        self._opened_with: SaveRequest | None = None
        self._edited = False

    def compose(self) -> ComposeResult:
        with Vertical(id="save-body", classes="dialog-body"):
            yield Static(self._title, id="save-title", classes="dialog-title", markup=False)
            yield Static(self._explain, id="save-explain", classes="dialog-note", markup=False)
            yield Input(self._initial, placeholder=msg.PROFILE_SAVE_PLACEHOLDER, id="save-name")
            yield Static("", id="save-error", classes="dialog-error", markup=False)
            yield from _folder_choice("save-folder", self._folders)
            with Horizontal(id="save-actions"):
                yield ConfirmPill(
                    msg.PROFILE_SAVE_LABEL,
                    pill_id="save-save",
                    answer=_DialogAnswer.SAVE,
                    primary=True,
                )
                yield ConfirmPill(
                    msg.PROFILE_CANCEL_LABEL, pill_id="save-cancel", answer=_DialogAnswer.CANCEL
                )

    def on_mount(self) -> None:
        self._opened_with = self._request()
        self._check()
        self.query_one("#save-name", Input).focus()

    def _request(self) -> SaveRequest:
        name = self.query_one("#save-name", Input).value.strip()
        return SaveRequest(name, _chosen_folder(self, "save-folder", self._folders))

    def _problem(self) -> NameProblem | None:
        request = self._request()
        return name_problem(request.name, request.folder, self._catalog)

    def _check(self) -> None:
        """Turn Save off while the name has a problem, and say what it is once the user edits."""
        problem = self._problem()
        error = self.query_one("#save-error", Static)
        text = msg.PROFILE_NAME_PROBLEM_TEXT[problem] if problem and self._edited else ""
        error.update(text)
        error.display = bool(text)
        self.query_one("#save-save", ConfirmPill).set_class(problem is not None, DISABLED_CLASS)

    @on(Input.Changed, "#save-name")
    @on(Select.Changed, "#save-folder")
    def _on_edit(self) -> None:
        if self._opened_with is not None and self._request() != self._opened_with:
            self._edited = True
        self._check()

    @on(Input.Submitted, "#save-name")
    def _on_submit(self) -> None:
        self._save()

    @on(ConfirmPill.Picked)
    def _on_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        if event.answer is _DialogAnswer.SAVE:
            self._save()
        else:
            self.dismiss(None)

    def _save(self) -> None:
        if self._problem() is None:
            self.dismiss(self._request())

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProfilePathDialog(ModalScreen[profiles.ProfileLocation | None]):
    """Asks for a path, and for a folder when it offers a choice.

    The dialog runs *write* with the answer and closes with its result; when *write* fails,
    the error shows as a toast and the dialog stays open for another path.
    """

    CSS_PATH: ClassVar[CSSPathType] = ["dialog.tcss", "profile_dialogs.tcss"]

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left", "app.focus_previous", "Previous", show=False),
        Binding("right", "app.focus_next", "Next", show=False),
    ]

    def __init__(
        self,
        *,
        title: str,
        explain: str,
        action_label: str,
        write: Callable[[PathRequest], profiles.ProfileLocation],
        initial: str = "",
        folders: Sequence[ProfileFolder] = (),
    ) -> None:
        super().__init__()
        self._write = write
        self._title = title
        self._explain = explain
        self._action_label = action_label
        self._initial = initial
        self._folders = folders
        self._opened_with: str | None = None
        self._edited = False

    def compose(self) -> ComposeResult:
        with Vertical(id="path-body", classes="dialog-body"):
            yield Static(self._title, id="path-title", classes="dialog-title", markup=False)
            with VerticalScroll(id="path-fields"):
                yield Static(self._explain, id="path-explain", classes="dialog-note", markup=False)
                yield Input(
                    self._initial, placeholder=msg.PROFILE_PATH_PLACEHOLDER, id="path-input"
                )
                yield from _folder_choice("path-folder", self._folders)
                yield Static("", id="path-error", classes="dialog-error", markup=False)
            with Horizontal(id="path-actions"):
                yield ConfirmPill(
                    self._action_label, pill_id="path-ok", answer=_DialogAnswer.SAVE, primary=True
                )
                yield ConfirmPill(
                    msg.PROFILE_CANCEL_LABEL, pill_id="path-cancel", answer=_DialogAnswer.CANCEL
                )

    def on_mount(self) -> None:
        self._opened_with = self._path()
        self._check()
        self.query_one("#path-input", Input).focus()

    def _path(self) -> str:
        return self.query_one("#path-input", Input).value.strip()

    def _check(self) -> None:
        """Turn the action off while the path is empty, and say so once the user edits."""
        empty = not self._path()
        error = self.query_one("#path-error", Static)
        text = msg.PROFILE_PATH_EMPTY if empty and self._edited else ""
        error.update(text)
        error.display = bool(text)
        self.query_one("#path-ok", ConfirmPill).set_class(empty, DISABLED_CLASS)

    def _request(self) -> PathRequest:
        folder = _chosen_folder(self, "path-folder", self._folders) if self._folders else None
        return PathRequest(Path(self._path()).expanduser(), folder)

    @on(Input.Changed, "#path-input")
    def _on_edit(self) -> None:
        if self._opened_with is not None and self._path() != self._opened_with:
            self._edited = True
        self._check()

    @on(Input.Submitted, "#path-input")
    def _on_submit(self) -> None:
        self._confirm()

    @on(ConfirmPill.Picked)
    def _on_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        if event.answer is _DialogAnswer.SAVE:
            self._confirm()
        else:
            self.dismiss(None)

    def _confirm(self) -> None:
        if self._path():
            request = self._request()
            run_profile_op(self, lambda: self._write(request), self._finish)

    def _finish(self, location: profiles.ProfileLocation) -> None:
        self.dismiss(location)

    def action_cancel(self) -> None:
        self.dismiss(None)
