"""The analyze command: read the corpus, recommend a profile, and the analyze tip."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.table import Table
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.commands._shared import fail, line
from lilbee.cli.commands.profile import render_changes
from lilbee.cli.helpers import json_output, sigint_cancel
from lilbee.core.config import cfg
from lilbee.core.profile_files import ProfileFolder, ProfileStore
from lilbee.runtime.console import PlainConsole
from lilbee.runtime.progress import AnalyzeEvent, EventType

if TYPE_CHECKING:
    from lilbee.app.analyze import AnalyzeRequest
    from lilbee.runtime.progress import DetailedProgressCallback, ProgressEvent
    from lilbee.server.models import AnalyzeResponse

READING_LABEL = "Reading files"
CANCELLED_MESSAGE = "Analyze was cancelled; nothing was saved."
OFF_ALONE_MESSAGE = "--off hides the tip and takes no folder, --apply, --save or --target."
TIP_HIDDEN_MESSAGE = "The analyze tip is hidden for this project."
MAX_FAILURES_SHOWN = 5

_directory_argument = typer.Argument(
    None,
    metavar="[DIR]",
    help="A folder to analyze; omit it to analyze the files lilbee indexes.",
    show_default=False,
)
_apply_option = typer.Option(
    False, "--apply", help="Save the recommended profile and switch this project to it."
)
_save_option = typer.Option(
    None,
    "--save",
    metavar="NAME",
    help="Save the recommended profile as NAME without switching to it.",
)
_target_option = typer.Option(
    None,
    "--target",
    help="Where to save with --apply or --save: project (the default when there is one) or global.",
    show_default=False,
)
_off_option = typer.Option(False, "--off", help="Hide the analyze tip; reads no files.")


def print_tip_if_shown(root: Path) -> None:
    """Print the analyze tip in text mode when the project at *root* should see it."""
    from lilbee.app.analyze import TIP_TEXT, tip_shows

    if not cfg.json_mode and tip_shows(root):
        console.print(Text(TIP_TEXT, style=theme.MUTED), soft_wrap=True)


@contextmanager
def _progress() -> Iterator[DetailedProgressCallback]:
    """A progress bar over the files analyze extracts; off in JSON mode."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress

    from lilbee.runtime.progress.columns import literal_text_column

    with Progress(
        literal_text_column("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
        console=PlainConsole(stderr=True),
        disable=cfg.json_mode,
    ) as progress:
        task = progress.add_task(READING_LABEL, total=None)

        def on_progress(event_type: EventType, data: ProgressEvent) -> None:
            # the callback carries every event type; only analyze events move this bar
            if event_type is EventType.ANALYZE and isinstance(data, AnalyzeEvent):
                progress.update(task, completed=data.done, total=data.total)

        yield on_progress


def _reading_line(report: AnalyzeResponse) -> str:
    read = f"Read {report.files_read} of {report.documents_total} documents"
    if report.files_counted:
        return f"{read}; counted {report.files_counted} code, image and archive files."
    return f"{read}."


def _render_reading(report: AnalyzeResponse) -> None:
    line(_reading_line(report), theme.ACCENT)
    if report.files_read + len(report.failed) < report.documents_total:
        line(f"The documents are sampled evenly; analyze_max_files is {report.cap}.", theme.MUTED)
    types = Table("File type", "Files")
    for kind, count in report.file_types.items():
        types.add_row(Text(kind), str(count))
    console.print(types)
    pdf = report.pdf
    line(f"Code files: {report.code_share:.0%}")
    line(f"Scanned pages: {pdf.scanned_pages} of {pdf.pages} PDF pages")
    line(f"Scanned share, image files included: {pdf.scanned_share:.0%}")
    line(f"PDFs with tables: {pdf.files_with_tables} of {pdf.files}")
    if pdf.median_pages is not None:
        line(f"Median PDF length: {pdf.median_pages:g} pages")
    if report.median_chars is not None:
        line(f"Median length of other files: {report.median_chars:g} characters")


def _render_languages(report: AnalyzeResponse) -> None:
    if not report.languages:
        line("No language detected: no file has enough text.")
        return
    table = Table("Language", "Share", "Search stemmer", "Tesseract")
    for lang in report.languages:
        stemmer = lang.fts_language.value if lang.fts_language else "none"
        tesseract = "installed" if lang.ocr_supported else "not installed"
        table.add_row(lang.code, f"{lang.share:.0%}", stemmer, tesseract)
    console.print(table)


def _render_failures(report: AnalyzeResponse) -> None:
    if not report.failed:
        return
    line(f"{len(report.failed)} files could not be read:", theme.WARNING)
    for failure in report.failed[:MAX_FAILURES_SHOWN]:
        line(f"  {failure.file}: {failure.error}")
    hidden = len(report.failed) - MAX_FAILURES_SHOWN
    if hidden > 0:
        line(f"  and {hidden} more; --json lists them all")


def _render_recommendation(report: AnalyzeResponse) -> None:
    rec = report.recommendation
    line(f"Recommended: {rec.name or rec.builtin}", theme.ACCENT)
    for reason in rec.reasons:
        line(f"  {reason.key}: {reason.text}")
    render_changes(rec.changes)
    if rec.kept:
        line(f"Keeps your values of: {', '.join(rec.kept)}")
    for note in rec.notes:
        line(note, theme.MUTED)


def _render_saved(report: AnalyzeResponse) -> None:
    saved = report.saved
    if saved is not None and saved.applied:
        line(f"This project now uses {saved.name}: {saved.path}", theme.ACCENT)
    elif saved is not None:
        line(f"Saved {saved.name}: {saved.path}")
    elif report.recommendation.name is not None:
        line("Run lilbee analyze --apply to save it and switch to it.", theme.MUTED)
    if saved is not None:
        for warning in saved.warnings:
            line(warning, theme.WARNING)


def _render(report: AnalyzeResponse) -> None:
    _render_reading(report)
    _render_languages(report)
    _render_failures(report)
    _render_recommendation(report)
    _render_saved(report)


def _analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run analyze with a progress bar and Ctrl+C as a clean cancel; a refusal exits 1."""
    from lilbee.app.analyze import run_analysis
    from lilbee.app.profiles import file_failure_message
    from lilbee.runtime.cancellation import TaskCancelledError
    from lilbee.server.models import AnalyzeResponse

    try:
        with _progress() as on_progress, sigint_cancel() as cancel:
            report = asyncio.run(
                run_analysis(ProfileStore(), request, on_progress=on_progress, cancel=cancel)
            )
    except TaskCancelledError:
        fail(CANCELLED_MESSAGE)
    except ValueError as exc:
        fail(str(exc))
    except OSError as exc:
        fail(file_failure_message(exc))
    return AnalyzeResponse.from_report(report)


def _hide_tip() -> None:
    from lilbee.app.analyze import hide_tip
    from lilbee.app.profiles import file_failure_message

    try:
        hide_tip(cfg.data_root)
    except OSError as exc:
        fail(file_failure_message(exc))
    if cfg.json_mode:
        json_output({"tip_dismissed": True})
    else:
        line(TIP_HIDDEN_MESSAGE)


def analyze_cmd(
    directory: Path | None = _directory_argument,
    apply: bool = _apply_option,
    save: str | None = _save_option,
    target: ProfileFolder | None = _target_option,
    off: bool = _off_option,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Read your documents and recommend a profile; --apply saves it and switches to it."""
    from lilbee.app.analyze import AnalyzeRequest

    apply_overrides(data_dir=data_dir, use_global=use_global)
    if off:
        if directory is not None or apply or save is not None or target is not None:
            fail(OFF_ALONE_MESSAGE)
        _hide_tip()
        return
    request = AnalyzeRequest(directory=directory, apply=apply, save=save, target=target)
    response = _analyze(request)
    if cfg.json_mode:
        json_output(response.model_dump(mode="json"))
        return
    _render(response)
