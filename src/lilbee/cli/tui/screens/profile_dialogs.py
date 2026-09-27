"""Profile dialogs: apply a profile, save the project's values as a new one, and their workers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
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
    NameProblem,
    ProfileCatalog,
    ProfileFile,
    ProfileFolder,
    ProfileStore,
    name_problem,
)

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

T = TypeVar("T")

PROFILE_OP_GROUP = "profile-op"
DISABLED_CLASS = "-disabled"


class ApplyChoice(StrEnum):
    """How the Apply dialog closed."""

    APPLY = "apply"
    APPLY_REINDEX = "apply_reindex"
    CANCEL = "cancel"


class _SaveAnswer(StrEnum):
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
    parts = [profiles.credit_line(profile.authors)]
    if profile.tested_on is not None:
        parts.append(msg.PROFILE_TESTED_ON.format(text=profile.tested_on))
    return ". ".join(part for part in parts if part)


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


def _switch_plan(name: str) -> SwitchPlan:
    """Diff *name* against the project and count the files a reindex would rebuild."""
    store = ProfileStore()
    profile = profiles.show(store, name).file
    diff = profiles.diff(store, name)
    reindex = any(row.effect is ProfileEffect.REINDEX for row in diff.changes)
    files = len(get_services().store.get_sources()) if reindex else 0
    kept = tuple((key, getattr(cfg, key)) for key in diff.kept)
    assert profile is not None  # noqa: S101 -- diff() refuses a broken profile first
    return SwitchPlan(profile, diff, kept, files)


def start_profile_switch(
    app: LilbeeApp, node: Widget, name: str, on_close: Callable[[], None]
) -> None:
    """Diff *name* off the loop, ask with the Apply dialog, then apply; *on_close* runs after."""

    def _on_choice(choice: ApplyChoice | None) -> None:
        if choice is None or choice is ApplyChoice.CANCEL:
            on_close()
            return

        def _applied(result: profiles.ApplyResult) -> None:
            app.publish_settings([row.key for row in result.changes])
            if choice is ApplyChoice.APPLY_REINDEX:
                app.start_rebuild()
            node.notify(msg.PROFILE_APPLIED.format(name=result.name))
            on_close()

        run_profile_op(node, lambda: profiles.apply(ProfileStore(), name), _applied)

    def _ask(plan: SwitchPlan) -> None:
        app.push_screen(ApplyProfileDialog(plan), _on_choice)

    run_profile_op(node, lambda: _switch_plan(name), _ask, on_error=on_close)


class ApplyProfileDialog(ModalScreen[ApplyChoice]):
    """Shows what applying a profile changes and keeps; nothing changes until a pill is picked."""

    CSS_PATH = "profile_dialogs.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("left", "app.focus_previous", "Previous", show=False),
        Binding("right", "app.focus_next", "Next", show=False),
    ]

    def __init__(self, plan: SwitchPlan) -> None:
        super().__init__()
        self._plan = plan
        self._reindex_rows = [
            row for row in plan.diff.changes if row.effect is ProfileEffect.REINDEX
        ]

    def compose(self) -> ComposeResult:
        profile = self._plan.profile
        with Vertical(id="apply-body"):
            with VerticalScroll(id="apply-scroll"):
                yield Static(
                    msg.PROFILE_APPLY_TITLE.format(name=profile.name),
                    id="apply-title",
                    markup=False,
                )
                for text, widget_id in (
                    (profile.description, "apply-description"),
                    (credit_text(profile), "apply-credit"),
                ):
                    if text:
                        yield Static(text, id=widget_id, markup=False)
                yield from self._compose_changes()
                yield from self._compose_kept()
                yield Static(msg.PROFILE_APPLY_UNTOUCHED, id="apply-untouched")
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
        table.add_columns(msg.PROFILE_COL_SETTING, msg.PROFILE_COL_NOW, msg.PROFILE_COL_AFTER)
        add_cost_column(table)
        table.add_rows(_diff_cells(row) for row in self._plan.diff.changes)
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
        if not self._reindex_rows:
            return ""
        return msg.profile_reindex_text(len(self._reindex_rows), self._plan.file_count)

    def _action_pills(self) -> ComposeResult:
        if self._reindex_rows:
            yield ConfirmPill(
                msg.PROFILE_APPLY_REINDEX_LABEL,
                pill_id="apply-reindex",
                answer=ApplyChoice.APPLY_REINDEX,
            )
        yield ConfirmPill(msg.PROFILE_APPLY_LABEL, pill_id="apply-apply", answer=ApplyChoice.APPLY)
        yield ConfirmPill(
            msg.PROFILE_CANCEL_LABEL, pill_id="apply-cancel", answer=ApplyChoice.CANCEL
        )

    def on_mount(self) -> None:
        self.query_one("#apply-actions ConfirmPill", ConfirmPill).focus()

    @on(ConfirmPill.Picked)
    def _on_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        self.dismiss(ApplyChoice(str(event.answer)))

    def action_cancel(self) -> None:
        self.dismiss(ApplyChoice.CANCEL)


def _diff_cells(row: DiffRow) -> Sequence[str | Content]:
    return (row.key, value_text(row.current), value_text(row.new), effect_cell(row.effect))


class SaveProfileDialog(ModalScreen[SaveRequest | None]):
    """Asks for a new profile's name and folder; Save stays off while the name is refused."""

    CSS_PATH = "profile_dialogs.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(
        self, *, active_name: str, change_count: int, catalog: ProfileCatalog, has_project: bool
    ) -> None:
        super().__init__()
        self._active_name = active_name
        self._change_count = change_count
        self._catalog = catalog
        self._has_project = has_project

    def compose(self) -> ComposeResult:
        folders = [(msg.PROFILE_SAVE_GLOBAL, ProfileFolder.GLOBAL)]
        if self._has_project:
            folders.append((msg.PROFILE_SAVE_PROJECT, ProfileFolder.PROJECT))
        with Vertical(id="save-body"):
            yield Static(msg.PROFILE_SAVE_TITLE, id="save-title")
            yield Input(placeholder=msg.PROFILE_SAVE_PLACEHOLDER, id="save-name")
            yield Static("", id="save-error", markup=False)
            yield Static(msg.PROFILE_SAVE_TO, classes="profile-section-title")
            yield Select(folders, value=ProfileFolder.GLOBAL, allow_blank=False, id="save-folder")
            yield Static(
                msg.PROFILE_SAVE_EXPLAIN.format(name=self._active_name, count=self._change_count),
                id="save-explain",
                markup=False,
            )
            with Horizontal(id="save-actions"):
                yield ConfirmPill(
                    msg.PROFILE_SAVE_LABEL, pill_id="save-save", answer=_SaveAnswer.SAVE
                )
                yield ConfirmPill(
                    msg.PROFILE_CANCEL_LABEL, pill_id="save-cancel", answer=_SaveAnswer.CANCEL
                )

    def on_mount(self) -> None:
        self._check()
        self.query_one("#save-name", Input).focus()

    def _request(self) -> SaveRequest:
        name = self.query_one("#save-name", Input).value.strip()
        folder = ProfileFolder(str(self.query_one("#save-folder", Select).value))
        return SaveRequest(name, folder)

    def _problem(self) -> NameProblem | None:
        request = self._request()
        return name_problem(request.name, request.folder, self._catalog)

    def _check(self) -> None:
        """Show the name's problem, and turn Save off while there is one."""
        problem = self._problem()
        error = self.query_one("#save-error", Static)
        error.update("" if problem is None else msg.PROFILE_NAME_PROBLEM_TEXT[problem])
        self.query_one("#save-save", ConfirmPill).set_class(problem is not None, DISABLED_CLASS)

    @on(Input.Changed, "#save-name")
    @on(Select.Changed, "#save-folder")
    def _on_edit(self) -> None:
        self._check()

    @on(Input.Submitted, "#save-name")
    def _on_submit(self) -> None:
        self._save()

    @on(ConfirmPill.Picked)
    def _on_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        if event.answer is _SaveAnswer.SAVE:
            self._save()
        else:
            self.dismiss(None)

    def _save(self) -> None:
        if self._problem() is None:
            self.dismiss(self._request())

    def action_cancel(self) -> None:
        self.dismiss(None)
