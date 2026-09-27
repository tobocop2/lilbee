"""The /analyze task and the report it opens: findings, the recommended profile, Apply or save."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NoReturn

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static

from lilbee.app import profiles
from lilbee.app.analyze import (
    PICK_REASON_KEY,
    AnalyzeReport,
    AnalyzeRequest,
    LanguageRow,
    Recommendation,
    run_analysis,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.browse_bindings import browse_back_bindings
from lilbee.cli.tui.screens.profile_dialogs import (
    add_cost_column,
    diff_cells,
    run_profile_op,
    start_profile_switch,
    start_switch,
    switch_plan,
)
from lilbee.cli.tui.task_queue import TaskType
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.bottom_bars import BottomBars
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmPill
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.cli.tui.widgets.task_bar import TaskBar
from lilbee.cli.tui.widgets.top_bars import TopBars
from lilbee.core.profile_files import ProfileStore
from lilbee.data.analyze import CorpusSignals
from lilbee.runtime import asyncio_loop
from lilbee.runtime.progress import AnalyzeEvent, EventType

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
    from lilbee.runtime.progress import ProgressEvent

_PERCENT = 100


class ReportAction(StrEnum):
    """The report's action pills."""

    APPLY = "apply"
    SAVE = "save"
    CLOSE = "close"


def start_analysis(app: LilbeeApp, directory: Path | None) -> None:
    """Queue analyze of *directory*, or of the indexed corpus when None; the report opens after."""
    folder = str(directory) if directory is not None else msg.ANALYZE_YOUR_DOCUMENTS

    def _target(reporter: ProgressReporter) -> None:
        report = _analyze(app, directory, reporter)
        call_from_thread(app, app.push_screen, AnalyzeReportScreen(report, folder))

    name = msg.TASK_NAME_ANALYZE.format(folder=folder)
    app.task_bar.start_task(name, TaskType.ANALYZE, _target, indeterminate=True)


def _analyze(app: LilbeeApp, directory: Path | None, reporter: ProgressReporter) -> AnalyzeReport:
    """Run analyze on the task worker; a refusal toasts its reason and fails the task."""
    reporter.update(0, msg.ANALYZE_STATUS_FINDING, indeterminate=True)
    request = AnalyzeRequest(directory=directory)

    def on_progress(event_type: EventType, data: ProgressEvent) -> None:
        # the callback carries every event type; only analyze events move the bar
        if event_type is EventType.ANALYZE and isinstance(data, AnalyzeEvent):
            percent = data.done * _PERCENT // data.total if data.total else 0
            detail = msg.ANALYZE_STATUS_READING.format(
                done=data.done, total=data.total, file=data.file
            )
            reporter.update(percent, detail, indeterminate=False)

    run = run_analysis(ProfileStore(), request, on_progress=on_progress, cancel=reporter)
    try:
        return asyncio_loop.run(run)
    except ValueError as exc:
        _refused(app, str(exc), exc)
    except OSError as exc:
        _refused(app, profiles.file_failure_message(exc), exc)


def _refused(app: LilbeeApp, text: str, exc: Exception) -> NoReturn:
    """Toast why analyze stopped, then fail the task with the same reason."""
    call_from_thread(app, app.notify, msg.ANALYZE_FAILED.format(error=text), severity="error")
    raise RuntimeError(text) from exc


def reading_text(signals: CorpusSignals) -> str:
    """How many documents analyze read and counted, and whether it sampled them."""
    template = msg.ANALYZE_READ_COUNTED if signals.files_counted else msg.ANALYZE_READ
    text = template.format(
        read=signals.files_read, total=signals.documents_total, counted=signals.files_counted
    )
    if signals.files_read + len(signals.failed) < signals.documents_total:
        return f"{text} {msg.ANALYZE_SAMPLED.format(cap=signals.cap)}"
    return text


def _types_text(file_types: Mapping[str, int]) -> str:
    ordered = sorted(file_types.items(), key=lambda item: (-item[1], item[0]))
    counts = [msg.ANALYZE_TYPE_COUNT.format(count=count, kind=kind) for kind, count in ordered]
    return ", ".join(counts) or msg.PROFILE_VALUE_NONE


def _languages_text(rows: Sequence[LanguageRow]) -> str:
    shares = [
        msg.ANALYZE_LANGUAGE_SHARE.format(name=row.fts_language or row.code, share=row.share)
        for row in rows
    ]
    if not shares:
        return msg.ANALYZE_LANGUAGES_NONE
    return msg.ANALYZE_LANGUAGES_TEXT.format(languages=", ".join(shares))


def fact_rows(report: AnalyzeReport) -> list[tuple[str, str]]:
    """The Found table: each signal analyze read, left out when there is nothing to show."""
    signals = report.signals
    pdf = signals.pdf
    rows: list[tuple[str, str | None]] = [
        (msg.ANALYZE_FACT_TYPES, _types_text(signals.file_types)),
        (
            msg.ANALYZE_FACT_CODE,
            msg.ANALYZE_SHARE_OF_FILES.format(share=signals.code_share)
            if signals.code_share
            else None,
        ),
        (
            msg.ANALYZE_FACT_SCANS,
            msg.ANALYZE_SHARE_OF_PAGES.format(share=pdf.scanned_share)
            if pdf.pages or signals.image_files
            else None,
        ),
        (
            msg.ANALYZE_FACT_TABLES,
            msg.ANALYZE_TABLES_VALUE.format(tables=pdf.files_with_tables, files=pdf.files)
            if pdf.files
            else None,
        ),
        (msg.ANALYZE_FACT_LANGUAGES, _languages_text(report.languages)),
        (
            msg.ANALYZE_FACT_PDF_LENGTH,
            None
            if pdf.median_pages is None
            else msg.ANALYZE_PAGES_VALUE.format(pages=pdf.median_pages),
        ),
        (
            msg.ANALYZE_FACT_TEXT_LENGTH,
            None
            if signals.median_chars is None
            else msg.ANALYZE_CHARS_VALUE.format(chars=signals.median_chars),
        ),
    ]
    return [(label, value) for label, value in rows if value is not None]


def _table(table_id: str, *columns: str) -> DataTable[str | Content]:
    table: DataTable[str | Content] = DataTable(
        id=table_id, cursor_type="none", zebra_stripes=False, show_cursor=False
    )
    table.add_columns(*columns)
    # read-only tables: focus moves between the scroll area and the action pills
    table.can_focus = False
    return table


def change_rows(recommendation: Recommendation) -> list[Sequence[str | Content]]:
    """The changes table: each setting applying changes, its cost, and why analyze set it."""
    why = {reason.key: reason.text for reason in recommendation.reasons}
    # every adjustment carries a reason, so a change without one is the built-in's own value
    builtin = msg.ANALYZE_WHY_BUILTIN.format(builtin=recommendation.builtin)
    return [(*diff_cells(row), why.get(row.key, builtin)) for row in recommendation.changes]


def _recommend_text(recommendation: Recommendation) -> str:
    template = (
        msg.ANALYZE_RECOMMEND_DERIVED if recommendation.name else msg.ANALYZE_RECOMMEND_BUILTIN
    )
    text = template.format(builtin=recommendation.builtin)
    picked = [r.text for r in recommendation.reasons if r.key == PICK_REASON_KEY]
    return " ".join([text, *(msg.ANALYZE_PICKED.format(reason=reason) for reason in picked)])


class AnalyzeReportScreen(Screen[None]):
    """What analyze found and the profile it recommends; nothing changes until Apply or Save."""

    CSS_PATH = "analyze_report.tcss"
    AUTO_FOCUS = "#analyze-apply"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("a", "apply", "Apply"),
        Binding("s", "save", "Save only", show=False),
        *browse_back_bindings(),
        Binding("left", "app.focus_previous", "Previous", show=False),
        Binding("right", "app.focus_next", "Next", show=False),
    ]

    app: LilbeeApp  # type: ignore[assignment]

    def __init__(self, report: AnalyzeReport, folder: str) -> None:
        super().__init__()
        self._report = report
        self._folder = folder
        self._actions: dict[ReportAction, Callable[[], None]] = {
            ReportAction.APPLY: self.action_apply,
            ReportAction.SAVE: self.action_save,
            ReportAction.CLOSE: self.action_go_back,
        }

    def compose(self) -> ComposeResult:
        with TopBars():
            yield ViewTabs()
        with VerticalScroll(id="analyze-report"):
            yield Static(
                msg.ANALYZE_TITLE.format(folder=self._folder), classes="analyze-heading first"
            )
            yield Static(reading_text(self._report.signals), classes="analyze-muted")
            facts = _table("analyze-facts", msg.ANALYZE_FOUND, "")
            facts.add_rows(fact_rows(self._report))
            yield facts
            yield from self._compose_failures()
            yield from self._compose_recommendation()
            yield from self._compose_notes()
        with Horizontal(id="analyze-actions"):
            yield ConfirmPill(
                msg.PROFILE_APPLY_LABEL, pill_id="analyze-apply", answer=ReportAction.APPLY
            )
            if self._report.recommendation.name is not None:
                yield ConfirmPill(
                    msg.ANALYZE_SAVE_LABEL, pill_id="analyze-save", answer=ReportAction.SAVE
                )
            yield ConfirmPill(
                msg.PROFILE_KEY_CLOSE, pill_id="analyze-close", answer=ReportAction.CLOSE
            )
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def _compose_failures(self) -> ComposeResult:
        failed = self._report.signals.failed
        if not failed:
            return
        yield Static(msg.ANALYZE_FAILED_TITLE, classes="analyze-heading")
        table = _table("analyze-failed", msg.ANALYZE_COL_FILE, msg.ANALYZE_COL_REASON)
        table.add_rows((failure.file, failure.error) for failure in failed)
        yield table

    def _compose_recommendation(self) -> ComposeResult:
        rec = self._report.recommendation
        title = msg.ANALYZE_RECOMMEND_TITLE.format(name=rec.name or rec.builtin)
        yield Static(title, classes="analyze-heading", markup=False)
        yield Static(_recommend_text(rec), classes="analyze-muted", markup=False)
        if not rec.changes:
            yield Static(msg.ANALYZE_NO_CHANGES, classes="analyze-muted")
        else:
            table = _table(
                "analyze-changes",
                msg.PROFILE_COL_SETTING,
                msg.PROFILE_COL_NOW,
                msg.PROFILE_COL_AFTER,
            )
            add_cost_column(table)
            table.add_column(msg.ANALYZE_COL_WHY)
            table.add_rows(change_rows(rec))
            yield table
        if rec.kept:
            keys = ", ".join(rec.kept)
            yield Static(msg.ANALYZE_KEEPS.format(keys=keys), classes="analyze-muted")

    def _compose_notes(self) -> ComposeResult:
        notes = self._report.recommendation.notes
        if not notes:
            return
        yield Static(msg.ANALYZE_NOTES_TITLE, classes="analyze-heading")
        for note in notes:
            yield Static(note, classes="analyze-muted", markup=False)

    @on(ConfirmPill.Picked)
    def _on_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        self._actions[ReportAction(str(event.answer))]()

    def action_apply(self) -> None:
        """Ask with the Apply dialog, then save the recommended profile and switch to it."""
        rec = self._report.recommendation
        name = rec.name
        if name is None:
            start_profile_switch(self.app, self, rec.builtin, _ignore, self._applied)
            return
        values = rec.values
        start_switch(
            self.app,
            self,
            lambda: switch_plan(
                profiles.recommended_file(name, values), profiles.preview(name, values)
            ),
            lambda: profiles.apply_recommended(
                ProfileStore(), name, values, profiles.default_save_folder()
            ),
            on_close=_ignore,
            on_applied=self._applied,
        )

    def action_save(self) -> None:
        """Save the recommended profile without switching to it."""
        rec = self._report.recommendation
        name = rec.name
        if name is None:
            return
        values = rec.values
        run_profile_op(
            self,
            lambda: profiles.save_recommended(
                ProfileStore(), name, values, profiles.default_save_folder(), switch=False
            ),
            self._saved,
        )

    def _saved(self, result: profiles.SaveResult) -> None:
        location = result.location
        self.notify(msg.PROFILE_SAVED.format(name=location.name, path=location.path))

    def _applied(self) -> None:
        self.dismiss()

    def action_go_back(self) -> None:
        self.dismiss()


def _ignore() -> None:
    """Keep the report open when the Apply dialog closes without applying."""
