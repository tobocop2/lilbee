"""The analyze command: read the corpus, recommend a profile, and the analyze tip."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer
from rich.table import Table
from rich.text import Text

from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.commands.profile import render_changes
from lilbee.cli.helpers import json_output, print_prefixed, sigint_cancel
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
    help="Where to save: project (the default when there is one) or global.",
    show_default=False,
)
_off_option = typer.Option(False, "--off", help="Hide the analyze tip; reads no files.")


def print_tip_if_shown(root: Path) -> None:
    """Print the analyze tip in text mode when the project at *root* should see it."""
    from lilbee.app.analyze import TIP_TEXT, tip_shows

    if not cfg.json_mode and tip_shows(root):
        console.print(Text(TIP_TEXT, style=theme.MUTED), soft_wrap=True)


def _fail(message: str) -> NoReturn:
    if cfg.json_mode:
        json_output({"error": message})
    else:
        print_prefixed(console, "Error: ", message, style=theme.ERROR)
    raise typer.Exit(1)


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


def _line(text: str, style: str | None = None) -> None:
    console.print(Text(text, style=style or ""), soft_wrap=True)


def _render_reading(report: AnalyzeResponse) -> None:
    _line(f"Read {report.files_read} of {report.files_total} files.", theme.ACCENT)
    if report.files_read < report.files_total:
        _line(f"The files are sampled evenly; analyze_max_files is {report.cap}.", theme.MUTED)
    types = Table("File type", "Files")
    for kind, count in report.file_types.items():
        types.add_row(Text(kind), str(count))
    console.print(types)
    pdf = report.pdf
    _line(f"Code files: {report.code_share:.0%}")
    _line(f"Scanned pages: {pdf.scanned_pages} of {pdf.pages} PDF pages")
    _line(f"Scanned share, image files included: {pdf.scanned_share:.0%}")
    _line(f"PDFs with tables: {pdf.files_with_tables} of {pdf.files}")
    if pdf.median_pages is not None:
        _line(f"Median PDF length: {pdf.median_pages:g} pages")
    if report.median_chars is not None:
        _line(f"Median length of other files: {report.median_chars:g} characters")


def _render_languages(report: AnalyzeResponse) -> None:
    if not report.languages:
        _line("No language detected: no file has enough text.")
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
    _line(f"{len(report.failed)} files could not be read:", theme.WARNING)
    for failure in report.failed[:MAX_FAILURES_SHOWN]:
        _line(f"  {failure.file}: {failure.error}")
    hidden = len(report.failed) - MAX_FAILURES_SHOWN
    if hidden > 0:
        _line(f"  and {hidden} more; --json lists them all")


def _render_recommendation(report: AnalyzeResponse) -> None:
    rec = report.recommendation
    _line(f"Recommended: {rec.name or rec.builtin}", theme.ACCENT)
    for reason in rec.reasons:
        _line(f"  {reason.key}: {reason.text}")
    render_changes(rec.changes)
    if rec.kept:
        _line(f"Keeps your values of: {', '.join(rec.kept)}")
    for note in rec.notes:
        _line(note, theme.MUTED)


def _render_saved(report: AnalyzeResponse) -> None:
    saved = report.saved
    if saved is not None and saved.applied:
        _line(f"This project now uses {saved.name}: {saved.path}", theme.ACCENT)
    elif saved is not None:
        _line(f"Saved {saved.name}: {saved.path}")
    elif report.recommendation.name is not None:
        _line("Run lilbee analyze --apply to save it and switch to it.", theme.MUTED)


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
        _fail(CANCELLED_MESSAGE)
    except ValueError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(file_failure_message(exc))
    return AnalyzeResponse.from_report(report)


def _hide_tip() -> None:
    from lilbee.app.analyze import hide_tip
    from lilbee.app.profiles import file_failure_message

    try:
        hide_tip(cfg.data_root)
    except OSError as exc:
        _fail(file_failure_message(exc))
    if cfg.json_mode:
        json_output({"tip_dismissed": True})
    else:
        _line(TIP_HIDDEN_MESSAGE)


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
            _fail(OFF_ALONE_MESSAGE)
        _hide_tip()
        return
    request = AnalyzeRequest(directory=directory, apply=apply, save=save, target=target)
    response = _analyze(request)
    if cfg.json_mode:
        json_output(response.model_dump(mode="json"))
        return
    _render(response)
