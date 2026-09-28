"""/analyze, the analyze report and its actions, and the analyze tip on the /add hint."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from textual.binding import Binding
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import DataTable

from lilbee.app import profiles
from lilbee.app.analyze import AnalyzeReport, LanguageRow, recommend, tip_shows
from lilbee.app.ingest import register_sources
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.command_registry import get_command
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.screens.analyze_report import AnalyzeReportScreen, fact_rows, reading_text
from lilbee.cli.tui.screens.profile_dialogs import ApplyProfileDialog
from lilbee.cli.tui.task_queue import TaskStatus, TaskType
from lilbee.cli.tui.widgets.arg_hint import ArgHintLine
from lilbee.cli.tui.widgets.autocomplete import get_completions
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmPill
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.cli.tui.widgets.slash_command_catalog import CATALOG_GROUPS
from lilbee.core.config import cfg
from lilbee.core.config.enums import FtsLanguage
from lilbee.core.profile_files import ProfileFolder, ProfileStore
from lilbee.core.project_state import mark_analyzed, read_state
from lilbee.data.analyze import CorpusSignals, FileFailure, LanguageShare, PdfSignals
from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.progress import AnalyzeEvent, EventType
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, await_chat
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
    sources_totaling,
)

# 300 pauses still missed a ConfirmPill focus event on a loaded Windows runner (#958).
_PAUSES = 900
_TIP_SHOWS = "lilbee.cli.tui.widgets.arg_hint.tip_shows"
_NARROW = (80, 24)
_WIDE = (120, 40)


def _screen_text(app: LilbeeApp) -> str:
    """The screen as rendered, one line per row."""
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


async def _until(pilot: Pilot, predicate: Callable[[], bool]) -> bool:
    return await wait_until(pilot, predicate, max_pauses=_PAUSES)


async def _press(pilot: Pilot, widget: Widget, key: str = "enter") -> None:
    widget.focus()
    assert await _until(pilot, lambda: widget.has_focus), widget
    await pilot.press(key)


def _signals(**overrides: Any) -> CorpusSignals:
    pdf = PdfSignals(
        files=10,
        pages=140,
        scanned_pages=60,
        scanned_share=60 / 140,
        files_with_tables=3,
        tables=7,
        median_pages=14.0,
    )
    base: dict[str, Any] = {
        "files_total": 12,
        "documents_total": 12,
        "files_read": 11,
        "cap": 500,
        "failed": (FileFailure("broken.pdf", "not a PDF"),),
        "file_types": {"pdf": 10, "docx": 2},
        "code_share": 0.0,
        "pdf": pdf,
        "median_chars": None,
        "languages": (LanguageShare("deu", 0.61), LanguageShare("eng", 0.39)),
        "image_files": 0,
    }
    return CorpusSignals(**{**base, **overrides})


def _rows(signals: CorpusSignals) -> tuple[LanguageRow, ...]:
    return tuple(
        LanguageRow(lang.code, lang.share, fts, lang.code, True)
        for lang, fts in zip(
            signals.languages, (FtsLanguage.GERMAN, FtsLanguage.ENGLISH), strict=False
        )
    )


def _report(signals: CorpusSignals | None = None) -> AnalyzeReport:
    signals = signals or _signals()
    rows = _rows(signals)
    return AnalyzeReport(signals, rows, recommend(ProfileStore(), signals, rows), None)


def _fits_default() -> AnalyzeReport:
    signals = _signals(
        failed=(),
        files_read=12,
        pdf=PdfSignals(0, 0, 0, 0.0, 0, 0, None),
        file_types={"docx": 12},
        languages=(),
    )
    return AnalyzeReport(signals, (), recommend(ProfileStore(), signals, ()), None)


class _ReportApp(LilbeeAppHost):
    def __init__(self, report: AnalyzeReport) -> None:
        super().__init__()
        self._report = report

    def on_mount(self) -> None:
        self.push_screen(AnalyzeReportScreen(self._report, "~/archive/city-records"))


async def _report_open(app: LilbeeAppHost, pilot: Pilot) -> AnalyzeReportScreen:
    assert await _until(
        pilot,
        lambda: isinstance(app.screen, AnalyzeReportScreen) and app.focused is not None,
    )
    screen = app.screen
    assert isinstance(screen, AnalyzeReportScreen)
    return screen


async def _dialog(app: LilbeeAppHost, pilot: Pilot) -> ApplyProfileDialog:
    assert await _until(
        pilot,
        lambda: (
            isinstance(app.screen, ApplyProfileDialog)
            and app.focused is not None
            and app.focused.screen is app.screen
        ),
    )
    screen = app.screen
    assert isinstance(screen, ApplyProfileDialog)
    return screen


@pytest.fixture
def chat_app():
    """The real app with a ready chat screen and no model scan."""
    with (
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True),
        mock.patch.object(ModelBar, "_scan_models"),
    ):
        yield LilbeeApp()


@pytest.fixture
def sources():
    with sources_totaling(0) as services:
        yield services


def _analyze_tasks(app: LilbeeApp) -> list[Any]:
    queue = app.task_bar.queue
    tasks = queue.history + queue.active_tasks + queue.queued_tasks
    return [t for t in tasks if t.task_type == TaskType.ANALYZE.value]


# The report


@pytest.mark.parametrize("size", [_NARROW, _WIDE])
async def test_report_shows_findings_changes_and_actions_on_screen(size) -> None:
    app = _ReportApp(_report())
    async with app.run_test(size=size) as pilot:
        screen = await _report_open(app, pilot)
        facts = screen.query_one("#analyze-facts", DataTable)
        assert facts.row_count == len(fact_rows(screen._report)) > 0
        changes = screen.query_one("#analyze-changes", DataTable)
        why = {
            str(changes.get_row_at(i)[0]): str(changes.get_row_at(i)[-1])
            for i in range(changes.row_count)
        }
        assert why["fts_language"] == "61% of text files are German"
        assert why["layout_detection"] == "Scanned archive sets it"
        failed = screen.query_one("#analyze-failed", DataTable)
        assert [str(c) for c in failed.get_row_at(0)] == ["broken.pdf", "not a PDF"]
        text = _screen_text(app)
        assert "What's in ~/archive/city-records" in text
        assert "Recommendation: Scanned archive (" in text
        flat = "".join(text.split())
        for line in (reading_text(screen._report.signals), "Scanned archive, with these changes."):
            assert "".join(line.split()) in flat, line
        for label in (msg.PROFILE_APPLY_LABEL, msg.ANALYZE_SAVE_LABEL, msg.PROFILE_KEY_CLOSE):
            assert label in text
        assert "None" not in text
        region = screen.region
        for pill in screen.query(ConfirmPill):
            assert region.contains_region(pill.region), pill.id
        assert app.focused is screen.query_one("#analyze-report")


async def test_the_changes_scroll_into_view_by_keyboard_at_80x24() -> None:
    app = _ReportApp(_report())
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        assert "fts_language" not in _screen_text(app)
        assert screen.query_one("#analyze-report").has_focus
        await pilot.press("pagedown")
        assert await _until(pilot, lambda: "fts_language" in _screen_text(app))
        assert msg.PROFILE_APPLY_LABEL in _screen_text(app)


def test_fact_rows_read_like_the_mockup() -> None:
    rows = dict(fact_rows(_report()))
    assert rows[msg.ANALYZE_FACT_TYPES] == "10 pdf, 2 docx"
    assert rows[msg.ANALYZE_FACT_SCANS] == "43% of pages"
    assert rows[msg.ANALYZE_FACT_TABLES] == "in 3 of 10 PDFs"
    assert rows[msg.ANALYZE_FACT_LANGUAGES] == "German 61%, English 39% (text files)"
    assert rows[msg.ANALYZE_FACT_PDF_LENGTH] == "14 pages"
    assert msg.ANALYZE_FACT_CODE not in rows
    assert msg.ANALYZE_FACT_TEXT_LENGTH not in rows


def test_fact_rows_without_pdfs_or_languages() -> None:
    rows = dict(fact_rows(_fits_default()))
    assert rows[msg.ANALYZE_FACT_LANGUAGES] == msg.ANALYZE_LANGUAGES_NONE
    assert msg.ANALYZE_FACT_SCANS not in rows
    assert msg.ANALYZE_FACT_TABLES not in rows
    signals = _signals(code_share=0.5, median_chars=3200.0, file_types={}, languages=())
    report = AnalyzeReport(signals, (), recommend(ProfileStore(), signals, ()), None)
    rows = dict(fact_rows(report))
    assert rows[msg.ANALYZE_FACT_CODE] == "50% of files"
    assert rows[msg.ANALYZE_FACT_TEXT_LENGTH] == "3,200 characters"
    assert rows[msg.ANALYZE_FACT_TYPES] == msg.PROFILE_VALUE_NONE


def test_reading_text_says_what_was_counted_and_sampled() -> None:
    assert reading_text(_signals(failed=(), files_read=12)) == msg.ANALYZE_READ.format(
        read=12, total=12
    )
    counted = reading_text(_signals(files_total=15, failed=(), files_read=12))
    assert "counted 3 code, image and archive files" in counted
    sampled = reading_text(_signals(documents_total=900, files_read=500, cap=500, failed=()))
    assert sampled.endswith(msg.ANALYZE_SAMPLED.format(cap=500))
    assert msg.ANALYZE_SAMPLED.format(cap=500) not in reading_text(_signals())


async def test_apply_asks_then_saves_and_switches_to_the_derived_profile(sources) -> None:
    report = _report()
    name = report.recommendation.name
    assert name is not None
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        await _press(pilot, screen.query_one("#analyze-apply", ConfirmPill))
        dialog = await _dialog(app, pilot)
        assert ProfileStore().scan().find(name) is None, "nothing is saved before the confirm"
        assert dialog._plan.profile.name == name
        await _press(pilot, dialog.query_one("#apply-apply", ConfirmPill))
        assert await _until(pilot, lambda: not isinstance(app.screen, AnalyzeReportScreen))
    assert profiles.active(ProfileStore()).name == name
    assert cfg.fts_language == FtsLanguage.GERMAN


async def test_apply_keeps_your_own_values(sources) -> None:
    config = cfg.data_root / "config.toml"
    config.write_text('fts_language = "French"\n', encoding="utf-8")
    report = _report()
    name = report.recommendation.name
    assert name is not None and "fts_language" in report.recommendation.kept
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        await _press(pilot, screen.query_one("#analyze-apply", ConfirmPill))
        dialog = await _dialog(app, pilot)
        await _press(pilot, dialog.query_one("#apply-apply", ConfirmPill))
        assert await _until(pilot, lambda: not isinstance(app.screen, AnalyzeReportScreen))
    assert profiles.active(ProfileStore()).name == name
    assert cfg.layout_detection is True, "the profile's own values apply"
    assert cfg.fts_language == FtsLanguage.FRENCH
    assert 'fts_language = "French"' in config.read_text(encoding="utf-8")


async def test_the_apply_dialog_starts_on_cancel_so_enter_applies_nothing(sources) -> None:
    report = _report()
    name = report.recommendation.name
    assert name is not None
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        await _press(pilot, screen.query_one("#analyze-apply", ConfirmPill))
        dialog = await _dialog(app, pilot)
        assert app.focused is dialog.query_one("#apply-cancel", ConfirmPill)
        await pilot.press("enter")
        assert await _until(pilot, lambda: app.screen is screen)
    assert ProfileStore().scan().find(name) is None
    assert profiles.active(ProfileStore()).name == "Default"
    assert app.task_bar.queue.is_empty


async def test_cancelling_the_apply_dialog_saves_nothing_and_keeps_the_report(sources) -> None:
    report = _report()
    name = report.recommendation.name
    assert name is not None
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        await _press(pilot, screen.query_one("#analyze-apply", ConfirmPill))
        dialog = await _dialog(app, pilot)
        await _press(pilot, dialog.query_one("#apply-cancel", ConfirmPill))
        assert await _until(pilot, lambda: app.screen is screen)
    assert ProfileStore().scan().find(name) is None
    assert profiles.active(ProfileStore()).name == "Default"


async def test_save_only_saves_without_switching() -> None:
    report = _report()
    name = report.recommendation.name
    assert name is not None
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        await _press(pilot, screen.query_one("#analyze-save", ConfirmPill))
        assert await _until(
            pilot, lambda: any(f"Saved {name}" in n.message for n in app._notifications)
        )
        assert app.screen is screen
    entry = ProfileStore().scan().find(name)
    assert entry is not None and entry.file is not None
    assert entry.file.description == profiles.ANALYZE_DESCRIPTION
    assert profiles.active(ProfileStore()).name == "Default"


async def test_stray_keys_on_the_open_report_act_on_nothing(sources) -> None:
    report = _report()
    name = report.recommendation.name
    assert name is not None
    config = cfg.data_root / "config.toml"
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        await pilot.press("enter", "space", "a", "s", "enter")
        await pilot.pause()
        assert app.screen is screen
    assert ProfileStore().scan().find(name) is None
    assert profiles.active(ProfileStore()).name == "Default"
    assert not config.exists()
    assert app.task_bar.queue.is_empty


async def test_a_refused_save_toasts_its_reason() -> None:
    report = _report()
    app = _ReportApp(report)
    refusal = ValueError("A profile named x already exists in the project folder")
    with mock.patch.object(profiles, "save_recommended", side_effect=refusal):
        async with app.run_test(size=_NARROW) as pilot:
            screen = await _report_open(app, pilot)
            await _press(pilot, screen.query_one("#analyze-save", ConfirmPill))
            assert await _until(
                pilot, lambda: any("already exists" in n.message for n in app._notifications)
            )


async def test_when_default_fits_apply_offers_default_and_there_is_no_save(sources) -> None:
    profiles.apply(ProfileStore(), "Research papers")
    app = _ReportApp(_fits_default())
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        assert not screen.query("#analyze-save")
        assert "Default fits these files" in _screen_text(app)
        await _press(pilot, screen.query_one("#analyze-apply", ConfirmPill))
        dialog = await _dialog(app, pilot)
        assert dialog._plan.profile.name == "Default"
        await _press(pilot, dialog.query_one("#apply-apply", ConfirmPill))
        assert await _until(pilot, lambda: not isinstance(app.screen, AnalyzeReportScreen))
    assert profiles.active(ProfileStore()).name == "Default"


async def test_a_report_with_no_changes_says_so() -> None:
    app = _ReportApp(_fits_default())
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        assert not screen.query("#analyze-changes")
        assert not screen.query("#analyze-failed")
        assert msg.ANALYZE_NO_CHANGES in _screen_text(app)


async def test_a_report_lists_your_kept_values(sources) -> None:
    (cfg.data_root / "config.toml").write_text('fts_language = "French"\n', encoding="utf-8")
    report = _report()
    assert "fts_language" in report.recommendation.kept
    app = _ReportApp(report)
    async with app.run_test(size=_WIDE) as pilot:
        await _report_open(app, pilot)
        assert "Keeps your values of fts_language" in _screen_text(app)


@pytest.mark.parametrize("close", ["q", "escape", "pill"])
async def test_close_leaves_the_report_and_saves_nothing(close) -> None:
    report = _report()
    app = _ReportApp(report)
    async with app.run_test(size=_NARROW) as pilot:
        screen = await _report_open(app, pilot)
        if close == "pill":
            await _press(pilot, screen.query_one("#analyze-close", ConfirmPill))
        else:
            await pilot.press(close)
        assert await _until(pilot, lambda: not isinstance(app.screen, AnalyzeReportScreen))
    assert report.recommendation.name is not None
    assert ProfileStore().scan().find(report.recommendation.name) is None


def test_report_binds_no_key_that_writes() -> None:
    bindings = [b for b in AnalyzeReportScreen.BINDINGS if isinstance(b, Binding)]
    assert bindings
    assert {b.action for b in bindings} == {"go_back", "app.focus_previous", "app.focus_next"}
    assert [b.description for b in bindings if b.show] == ["Back"]


# /analyze and the task


def _write_notes(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        text = f"# Note {i}\n\nPlain English text.\n"
        (folder / f"note{i}.md").write_text(text, encoding="utf-8")
    return folder


async def _finished_report(app: LilbeeApp, pilot: Pilot, chat: Any) -> AnalyzeReportScreen:
    """Wait for the finished toast, check nothing opened unasked, then open the report."""
    assert await _until(pilot, lambda: app.last_analysis is not None)
    assert await _until(
        pilot, lambda: any("/analyze report" in n.message for n in app._notifications)
    )
    assert app.screen is chat
    chat.run_command("/analyze report")
    return await _report_open(app, pilot)


async def test_slash_analyze_reads_a_folder_and_its_report_opens_on_request(
    chat_app, tmp_path
) -> None:
    folder = _write_notes(tmp_path / "notes")
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command(f"/analyze {folder}")
        screen = await _finished_report(chat_app, pilot, chat)
        assert screen._report.signals.files_total == 3
        flat = "".join(_screen_text(chat_app).split())
        assert "".join(f"What's in {folder}".split()) in flat
        tasks = _analyze_tasks(chat_app)
        assert [t.status for t in tasks] == [TaskStatus.DONE]
        assert tasks[0].name == msg.TASK_NAME_ANALYZE.format(folder=folder)
    assert read_state(cfg.data_root).analyzed_at is not None


async def test_typed_slash_analyze_takes_a_quoted_folder_with_spaces(chat_app, tmp_path) -> None:
    folder = _write_notes(tmp_path / "city records")
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat_input = chat.query_one("#chat-input", ChatInput)
        chat_input.focus()
        chat_input.value = f'/analyze "{folder}"'
        await pilot.press("enter")
        screen = await _finished_report(chat_app, pilot, chat)
        assert screen._report.signals.files_total == 3
        assert _analyze_tasks(chat_app)[0].name == msg.TASK_NAME_ANALYZE.format(folder=folder)


async def test_slash_analyze_alone_reads_the_indexed_documents(chat_app) -> None:
    _write_notes(cfg.documents_dir)
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command("/analyze")
        screen = await _finished_report(chat_app, pilot, chat)
        assert screen._report.signals.files_total == 3
        assert "What's in your documents" in _screen_text(chat_app)


async def test_palette_analyze_documents_runs_analyze(chat_app) -> None:
    _write_notes(cfg.documents_dir)
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        provider = LilbeeCommandProvider(chat, match_style=None)
        entry = next(c for c in provider._get_commands() if c[0] == "Analyze documents")
        entry[2]()
        screen = await _finished_report(chat_app, pilot, chat)
        assert screen._report.signals.files_total == 3


async def test_keys_typed_into_chat_as_analyze_finishes_stay_in_chat(
    chat_app, tmp_path, sources
) -> None:
    folder = _write_notes(tmp_path / "notes")
    config = cfg.data_root / "config.toml"
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat_input = chat.query_one("#chat-input", ChatInput)
        chat_input.focus()
        chat.run_command(f"/analyze {folder}")
        assert await _until(
            pilot, lambda: [t.status for t in _analyze_tasks(chat_app)] == [TaskStatus.DONE]
        )
        await pilot.press("enter", "enter", "space", "a", "s", "q")
        assert await _until(pilot, lambda: chat_input.value == " asq")
        assert chat_app.screen is chat
        assert chat_input.has_focus
    assert not config.exists()
    saved = [e for e in ProfileStore().scan().entries if e.folder is not ProfileFolder.BUILTIN]
    assert saved == []
    assert profiles.active(ProfileStore()).name == "Default"
    queue = chat_app.task_bar.queue
    assert queue.active_tasks == queue.queued_tasks == []
    assert [t.task_type for t in queue.history] == [TaskType.ANALYZE.value]


async def test_analyze_report_before_any_run_says_there_is_none(chat_app) -> None:
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command("/analyze report")
        assert await _until(
            pilot, lambda: any(msg.ANALYZE_NO_REPORT in n.message for n in chat_app._notifications)
        )
        assert chat_app.screen is chat
        assert _analyze_tasks(chat_app) == []


async def test_slash_analyze_on_a_missing_folder_toasts_and_starts_nothing(
    chat_app, tmp_path
) -> None:
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command(f"/analyze {tmp_path / 'nope'}")
        assert await _until(
            pilot, lambda: any("is not a folder" in n.message for n in chat_app._notifications)
        )
        assert _analyze_tasks(chat_app) == []


def _stalled_collect(started: list[bool]):
    async def _collect(files, *, on_progress, cancel):
        on_progress(EventType.ANALYZE, AnalyzeEvent(done=1, total=4, file="a.pdf"))
        started.append(True)
        while not cancel.is_set():
            await asyncio.sleep(0.01)
        raise TaskCancelledError

    return _collect


async def test_analyze_shows_progress_and_a_cancel_stops_it_without_a_report(
    chat_app, tmp_path
) -> None:
    folder = _write_notes(tmp_path / "notes")
    started: list[bool] = []
    with mock.patch("lilbee.app.analyze.collect_signals", _stalled_collect(started)):
        async with chat_app.run_test(size=_NARROW) as pilot:
            chat = await await_chat(chat_app, pilot)
            chat.run_command(f"/analyze {folder}")
            assert await _until(pilot, lambda: bool(started))
            (task,) = _analyze_tasks(chat_app)
            assert task.progress == 25
            assert task.detail == msg.ANALYZE_STATUS_READING.format(done=1, total=4, file="a.pdf")
            chat_app.task_bar.cancel_task(task.task_id)
            assert await _until(
                pilot, lambda: _analyze_tasks(chat_app)[0].status is TaskStatus.CANCELLED
            )
            await pilot.pause()
            assert not isinstance(chat_app.screen, AnalyzeReportScreen)
    assert read_state(cfg.data_root).analyzed_at is None


async def test_a_failed_analyze_toasts_and_fails_the_task(chat_app, tmp_path) -> None:
    folder = _write_notes(tmp_path / "notes")
    failure = mock.AsyncMock(side_effect=PermissionError(13, "Permission denied"))
    with mock.patch("lilbee.app.analyze.collect_signals", failure):
        async with chat_app.run_test(size=_NARROW) as pilot:
            chat = await await_chat(chat_app, pilot)
            chat.run_command(f"/analyze {folder}")
            assert await _until(
                pilot,
                lambda: any("Analyze failed" in n.message for n in chat_app._notifications),
            )
            assert await _until(
                pilot, lambda: _analyze_tasks(chat_app)[0].status is TaskStatus.FAILED
            )


async def test_a_refused_analyze_names_the_reason(chat_app, tmp_path) -> None:
    folder = _write_notes(tmp_path / "notes")
    refusal = mock.AsyncMock(side_effect=ValueError("The built-in profile x is missing"))
    with mock.patch("lilbee.app.analyze.collect_signals", refusal):
        async with chat_app.run_test(size=_NARROW) as pilot:
            chat = await await_chat(chat_app, pilot)
            chat.run_command(f"/analyze {folder}")
            assert await _until(
                pilot,
                lambda: any(
                    "Analyze failed: The built-in profile x is missing" in n.message
                    for n in chat_app._notifications
                ),
            )


# /analyze off, completion and help


async def test_slash_analyze_off_hides_the_tip(chat_app) -> None:
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command("/analyze off")
        assert await _until(
            pilot, lambda: any(msg.ANALYZE_TIP_HIDDEN in n.message for n in chat_app._notifications)
        )
    assert read_state(cfg.data_root).tip_dismissed is True
    assert _analyze_tasks(chat_app) == []


def test_analyze_completion_offers_off_ahead_of_folders(tmp_path, monkeypatch) -> None:
    (tmp_path / "old-scans").mkdir()
    (tmp_path / "papers").mkdir()
    monkeypatch.chdir(tmp_path)
    assert get_completions("/analyze ")[:4] == ["off", "report", "old-scans/", "papers/"]
    assert get_completions("/analyze o") == ["off", "old-scans/"]
    assert get_completions("/analyze r") == ["report"]
    assert get_completions("/analyze p") == ["papers/"]
    assert get_completions("/add o") == ["old-scans/"]


def test_analyze_is_registered_and_listed_in_the_help_catalog() -> None:
    names = [name for group in CATALOG_GROUPS for name in group.members]
    assert names.index("/analyze") == names.index("/profile") + 1
    assert get_command("/analyze").args_hint == "[dir|report|off]"


# The tip on the /add hint


def _hint_text(chat: Any) -> str:
    hint = chat.query_one("#arg-hint", ArgHintLine)
    return str(hint.render()) if hint.display else ""


async def _type_add(chat_app: LilbeeApp, pilot: Pilot, path: Path) -> Any:
    chat = await await_chat(chat_app, pilot)
    chat_input = chat.query_one("#chat-input", ChatInput)
    chat_input.focus()
    await pilot.press(*"/add ")
    chat_input.value = f"/add {path}"
    return chat


@pytest.mark.parametrize("size", [_NARROW, _WIDE])
async def test_the_tip_shows_under_the_add_hint_before_anything_is_added(
    chat_app, tmp_path, size
) -> None:
    folder = _write_notes(tmp_path / "city-records")
    with mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._submit_add") as submit:
        async with chat_app.run_test(size=size) as pilot:
            chat = await _type_add(chat_app, pilot, folder)
            assert await _until(pilot, lambda: msg.ANALYZE_TIP_LABEL in _hint_text(chat))
            text = _screen_text(chat_app)
            assert "This project uses the Default profile" in text
            assert "/analyze off to hide this tip" in text.replace("\n", " ").replace("  ", " ")
            submit.assert_not_called()
            assert chat_app.task_bar.queue.is_empty
            await pilot.press("enter")
            assert await _until(pilot, lambda: submit.called)


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(lambda: mark_analyzed(cfg.data_root), id="analyzed"),
        pytest.param(lambda: profiles.apply(ProfileStore(), "Research papers"), id="on-a-profile"),
        pytest.param(lambda: _write_notes(cfg.documents_dir), id="has-documents"),
        pytest.param(lambda: register_sources([_write_notes(cfg.data_root / "src")]), id="added"),
    ],
)
async def test_the_tip_stays_away_when_the_project_does_not_need_it(
    chat_app, tmp_path, setup
) -> None:
    setup()
    folder = _write_notes(tmp_path / "docs")
    with mock.patch(_TIP_SHOWS, wraps=tip_shows) as read:
        async with chat_app.run_test(size=_NARROW) as pilot:
            chat = await _type_add(chat_app, pilot, folder)
            assert await _until(pilot, lambda: read.called and "/add" in _hint_text(chat))
            await _tip_read_settled(chat_app, pilot)
            assert msg.ANALYZE_TIP_LABEL not in _hint_text(chat)
    assert [call.args for call in read.call_args_list] == [(cfg.data_root,)]


async def _tip_read_settled(app: LilbeeApp, pilot: Pilot) -> None:
    """Wait until no tip read is still running, so the hint shows what it read."""
    assert await _until(
        pilot, lambda: not any(w.group == "analyze-tip" and w.is_running for w in app.workers)
    )
    await pilot.pause()


async def test_analyze_off_takes_the_tip_off_the_add_hint(chat_app, tmp_path) -> None:
    folder = _write_notes(tmp_path / "docs")
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await _type_add(chat_app, pilot, folder)
        assert await _until(pilot, lambda: msg.ANALYZE_TIP_LABEL in _hint_text(chat))
        chat.run_command("/analyze off")
        assert await _until(pilot, lambda: msg.ANALYZE_TIP_LABEL not in _hint_text(chat))
        assert "/add" in _hint_text(chat)


async def test_other_commands_get_no_tip(chat_app) -> None:
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.query_one("#chat-input").focus()
        await pilot.press(*"/import x")
        assert await _until(pilot, lambda: "/import" in _hint_text(chat))
        await _tip_read_settled(chat_app, pilot)
        assert msg.ANALYZE_TIP_LABEL not in _hint_text(chat)


async def test_the_tip_leaves_when_the_command_is_no_longer_add(chat_app, tmp_path) -> None:
    folder = _write_notes(tmp_path / "docs")
    async with chat_app.run_test(size=_NARROW) as pilot:
        chat = await _type_add(chat_app, pilot, folder)
        assert await _until(pilot, lambda: msg.ANALYZE_TIP_LABEL in _hint_text(chat))
        chat.query_one("#chat-input", ChatInput).value = f"/import {folder}"
        assert await _until(pilot, lambda: "/import" in _hint_text(chat))
        assert msg.ANALYZE_TIP_LABEL not in _hint_text(chat)


async def test_the_tip_goes_once_an_add_is_queued_for_indexing(chat_app, tmp_path) -> None:
    folder = _write_notes(tmp_path / "docs")
    release = threading.Event()

    async def _held_sync(**_kwargs: Any) -> None:
        await asyncio.to_thread(release.wait, 30)
        raise RuntimeError("released")

    with mock.patch("lilbee.data.ingest.sync", _held_sync):
        async with chat_app.run_test(size=_NARROW) as pilot:
            chat = await _type_add(chat_app, pilot, folder)
            assert await _until(pilot, lambda: msg.ANALYZE_TIP_LABEL in _hint_text(chat))
            await pilot.press("enter")
            try:
                assert await _until(pilot, lambda: bool(cfg.linked_roots))
                (task,) = chat_app.task_bar.queue.active_tasks
                assert task.task_type == TaskType.ADD.value
                chat_input = chat.query_one("#chat-input", ChatInput)
                chat_input.value = f"/add {folder}"
                assert await _until(pilot, lambda: "/add" in _hint_text(chat))
                await _tip_read_settled(chat_app, pilot)
                assert msg.ANALYZE_TIP_LABEL not in _hint_text(chat)
            finally:
                release.set()
