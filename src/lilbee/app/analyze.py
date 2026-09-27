"""Corpus analysis use case: signals to a recommended profile, saving it, and the analyze tip."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lilbee.app.profiles import (
    DiffRow,
    apply,
    default_save_folder,
    preview,
    save_recommended,
    show,
)
from lilbee.core.config import cfg
from lilbee.core.config.enums import FtsLanguage
from lilbee.core.config.resolve import read_layers, read_profile_table, resolve
from lilbee.core.profile_files import (
    DEFAULT_PROFILE_NAME,
    ProfileFolder,
    ProfileStore,
    profile_key,
)
from lilbee.core.project_state import dismiss_tip, mark_analyzed, read_state
from lilbee.core.system import LOCAL_ROOT_DIRNAME
from lilbee.data.analyze import CorpusSignals, collect_signals, ocr_language_supported
from lilbee.data.ingest.discovery import discover_corpus, discover_dir
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError
from lilbee.runtime.progress import DetailedProgressCallback, noop_callback

log = logging.getLogger(__name__)

# Built-in pick thresholds, first match wins. Unmeasured: a run on real corpora tunes them.
SCANNED_ARCHIVE_SHARE = 0.30
CODE_REPOSITORY_SHARE = 0.50
NOTES_SHARE = 0.50
RESEARCH_TABLES_SHARE = 0.20
# Languages below this share of text files are left out of ocr_language. Unmeasured.
OCR_LANGUAGE_FLOOR = 0.10
MAX_OCR_LANGUAGES = 3

SCANNED_ARCHIVE = "Scanned archive"
CODE_REPOSITORY = "Code repository"
NOTES_AND_MARKDOWN = "Notes and markdown"
RESEARCH_PAPERS = "Research papers"
PICK_REASON_KEY = "profile"
MAX_PROFILE_NAME_LENGTH = 40
FALLBACK_PROJECT_NAME = "project"

NOTE_CONTENT_TYPES = frozenset({"md", "markdown", "mdx", "txt"})
OCR_LANGUAGE_REASON = "assumed from the text pages"
OCR_MODEL_NOTE = "The catalog lists OCR models that fit this machine."
DEFAULT_FITS_NOTE = "Default fits this corpus."
IMAGE_SCAN_NOTE = (
    "Image files are counted, not read: each adds one scanned page to the scanned share, "
    "scaled by the share of documents sampled."
)
TARGET_WITHOUT_SAVE_MESSAGE = "target takes effect only with apply or save; nothing was read."
RELATIVE_DIRECTORY_MESSAGE = "directory must be an absolute path on the lilbee server: {value!r}"
TIP_TEXT = (
    "This project uses the Default profile. Run lilbee analyze to get a profile "
    "recommendation, or lilbee analyze --off to hide this tip."
)

# ISO 639-3 codes xberg detects, mapped to the Snowball stemmers of full-text search.
FTS_LANGUAGE_BY_CODE: Mapping[str, FtsLanguage] = {
    "ara": FtsLanguage.ARABIC,
    "dan": FtsLanguage.DANISH,
    "nld": FtsLanguage.DUTCH,
    "eng": FtsLanguage.ENGLISH,
    "fin": FtsLanguage.FINNISH,
    "fra": FtsLanguage.FRENCH,
    "deu": FtsLanguage.GERMAN,
    "ell": FtsLanguage.GREEK,
    "hun": FtsLanguage.HUNGARIAN,
    "ita": FtsLanguage.ITALIAN,
    "nob": FtsLanguage.NORWEGIAN,
    "nno": FtsLanguage.NORWEGIAN,
    "nor": FtsLanguage.NORWEGIAN,
    "por": FtsLanguage.PORTUGUESE,
    "ron": FtsLanguage.ROMANIAN,
    "rus": FtsLanguage.RUSSIAN,
    "spa": FtsLanguage.SPANISH,
    "swe": FtsLanguage.SWEDISH,
    "tam": FtsLanguage.TAMIL,
    "tur": FtsLanguage.TURKISH,
}

# ISO 639-3 codes xberg detects whose Tesseract language data goes by another name.
TESSERACT_LANGUAGE_BY_CODE: Mapping[str, str] = {
    "cmn": "chi_sim",
    "nob": "nor",
    "pes": "fas",
}

_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9 _-]+")


@dataclass(frozen=True)
class LanguageRow:
    """A detected language, its share, its stemmer, and its Tesseract name and support."""

    code: str
    share: float
    fts_language: FtsLanguage | None
    ocr_code: str
    ocr_supported: bool


@dataclass(frozen=True)
class Reason:
    """Why the recommendation sets *key*; ``profile`` explains the built-in pick."""

    key: str
    text: str


@dataclass(frozen=True)
class Recommendation:
    """The picked built-in, the derived profile's name and values, and what applying it changes.

    ``name`` is None when Default fits and there is nothing to save.
    """

    builtin: str
    name: str | None
    values: Mapping[str, Any]
    changes: tuple[DiffRow, ...]
    kept: tuple[str, ...]
    reasons: tuple[Reason, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class SavedProfile:
    """The profile analyze saved or switched to, its file, and whether the project now uses it."""

    name: str
    folder: ProfileFolder
    path: Path
    applied: bool


@dataclass(frozen=True)
class AnalyzeReport:
    """Everything one analyze run read and recommends."""

    signals: CorpusSignals
    languages: tuple[LanguageRow, ...]
    recommendation: Recommendation
    saved: SavedProfile | None


@dataclass(frozen=True)
class AnalyzeRequest:
    """What to analyze and what to do with the recommendation.

    No *directory* reads the corpus ingest reads. *save* names the saved profile; *apply* also
    switches to it. *target* picks the folder and needs *apply* or *save*; without it the
    profile goes to the project folder when there is one, else the global one.
    """

    directory: Path | None = None
    apply: bool = False
    save: str | None = None
    target: ProfileFolder | None = None


@dataclass(frozen=True)
class _PickRule:
    builtin: str
    threshold: float
    share: Callable[[CorpusSignals], float]
    reason: str


def _notes_share(signals: CorpusSignals) -> float:
    notes = sum(n for kind, n in signals.file_types.items() if kind in NOTE_CONTENT_TYPES)
    return notes / signals.files_total if signals.files_total else 0.0


def _tables_share(signals: CorpusSignals) -> float:
    pdf = signals.pdf
    return pdf.files_with_tables / pdf.files if pdf.files else 0.0


_PICK_RULES: tuple[_PickRule, ...] = (
    _PickRule(
        SCANNED_ARCHIVE, SCANNED_ARCHIVE_SHARE, lambda s: s.pdf.scanned_share, "of pages are scans"
    ),
    _PickRule(CODE_REPOSITORY, CODE_REPOSITORY_SHARE, lambda s: s.code_share, "of files are code"),
    _PickRule(NOTES_AND_MARKDOWN, NOTES_SHARE, _notes_share, "of files are Markdown or text"),
    _PickRule(RESEARCH_PAPERS, RESEARCH_TABLES_SHARE, _tables_share, "of PDFs have tables"),
)


def pick_builtin(signals: CorpusSignals) -> tuple[str, Reason | None]:
    """The first built-in whose share reaches its threshold, else Default, with the reason."""
    for rule in _PICK_RULES:
        share = rule.share(signals)
        if share >= rule.threshold:
            return rule.builtin, Reason(PICK_REASON_KEY, f"{share:.0%} {rule.reason}")
    return DEFAULT_PROFILE_NAME, None


def _language_row(code: str, share: float) -> LanguageRow:
    ocr_code = TESSERACT_LANGUAGE_BY_CODE.get(code, code)
    return LanguageRow(
        code, share, FTS_LANGUAGE_BY_CODE.get(code), ocr_code, ocr_language_supported(ocr_code)
    )


def language_rows(signals: CorpusSignals) -> tuple[LanguageRow, ...]:
    """Each detected language with its stemmer and Tesseract support."""
    return tuple(_language_row(lang.code, lang.share) for lang in signals.languages)


def project_label(data_root: Path) -> str:
    """The project's folder name: the parent of a ``.lilbee`` folder, else the data root's own."""
    folder = data_root.parent if data_root.name == LOCAL_ROOT_DIRNAME else data_root
    return folder.name


def derived_name(builtin: str, project: str) -> str:
    """The derived profile's name, "<built-in> (<project>)", cut to fit the profile name rules."""
    room = MAX_PROFILE_NAME_LENGTH - len(builtin) - len(" ()")
    safe = _NAME_UNSAFE.sub("-", project).strip(" -_")[:room].strip(" -_")
    return f"{builtin} ({safe or FALLBACK_PROJECT_NAME[:room]})"


class _Plan:
    """The adjustments, reasons and notes a recommendation collects on top of a built-in."""

    def __init__(self, builtin_values: Mapping[str, Any]) -> None:
        layers = read_layers(cfg.data_root)
        self._layers = replace(layers, profile=dict(builtin_values))
        self.adjustments: dict[str, Any] = {}
        self.reasons: list[Reason] = []
        self.notes: list[str] = []

    def baseline(self, key: str) -> Any:
        """*key*'s value under the built-in alone, with user and env values on top."""
        return resolve(key, self._layers).value

    def adjust(self, key: str, value: Any, reason: str) -> None:
        """Set *key* to *value* when the built-in alone gives something else."""
        if value != self.baseline(key):
            self.adjustments[key] = value
            self.reasons.append(Reason(key, reason))


def _plan_fts(plan: _Plan, rows: tuple[LanguageRow, ...]) -> None:
    if not rows:
        return
    top = rows[0]
    if top.fts_language is None:
        plan.notes.append(
            f"No stemmer for {top.code}; fts_language stays {plan.baseline('fts_language')}."
        )
        return
    plan.adjust(
        "fts_language", top.fts_language, f"{top.share:.0%} of text files are {top.fts_language}"
    )


def _plan_ocr_language(plan: _Plan, signals: CorpusSignals, rows: tuple[LanguageRow, ...]) -> None:
    scans = signals.pdf.scanned_pages + signals.image_files
    if scans == 0 or plan.baseline("enable_ocr") is False:
        return
    common = [row for row in rows if row.share >= OCR_LANGUAGE_FLOOR][:MAX_OCR_LANGUAGES]
    for row in common:
        if not row.ocr_supported:
            plan.notes.append(f"Tesseract has no language data for {row.ocr_code} on this machine.")
    supported = [row.ocr_code for row in common if row.ocr_supported]
    if supported:
        plan.adjust("ocr_language", supported, OCR_LANGUAGE_REASON)


def _plan_ocr_model(plan: _Plan, signals: CorpusSignals) -> None:
    if signals.pdf.scanned_share >= SCANNED_ARCHIVE_SHARE and not cfg.vision_model:
        plan.notes.append(OCR_MODEL_NOTE)


def _builtin_values(store: ProfileStore, builtin: str) -> Mapping[str, Any]:
    entry = store.scan().find(builtin)
    if entry is None or entry.file is None:
        raise ValueError(f"The built-in profile {builtin} is missing or broken")
    return entry.file.values


def recommend(
    store: ProfileStore, signals: CorpusSignals, rows: tuple[LanguageRow, ...]
) -> Recommendation:
    """The built-in that fits *signals*, adjusted for the corpus, and what applying it changes."""
    builtin, pick_reason = pick_builtin(signals)
    builtin_values = _builtin_values(store, builtin)
    plan = _Plan(builtin_values)
    _plan_fts(plan, rows)
    _plan_ocr_language(plan, signals, rows)
    _plan_ocr_model(plan, signals)
    if signals.image_files:
        plan.notes.append(IMAGE_SCAN_NOTE)
    values = {**builtin_values, **plan.adjustments}
    fits_default = builtin == DEFAULT_PROFILE_NAME and not plan.adjustments
    name = None if fits_default else derived_name(builtin, project_label(cfg.data_root))
    if fits_default:
        plan.notes.append(DEFAULT_FITS_NOTE)
    diff = preview(name or builtin, values)
    reasons = ([pick_reason] if pick_reason else []) + plan.reasons
    return Recommendation(
        builtin, name, values, diff.changes, diff.kept, tuple(reasons), tuple(plan.notes)
    )


def _save(
    store: ProfileStore, recommendation: Recommendation, request: AnalyzeRequest
) -> SavedProfile | None:
    """Save the recommendation when the request asks; applying a fitting Default switches to it."""
    if not request.apply and request.save is None:
        return None
    name = request.save if request.save is not None else recommendation.name
    if name is None:
        entry = show(store, apply(store, recommendation.builtin).name)
        return SavedProfile(entry.name, entry.folder, entry.path, applied=True)
    target = request.target or default_save_folder()
    result = save_recommended(store, name, recommendation.values, target, switch=request.apply)
    location = result.location
    return SavedProfile(location.name, location.folder, location.path, request.apply)


def validate_request(request: AnalyzeRequest) -> None:
    """Refuse a *target* without *apply* or *save*, and a *directory* that is not a folder."""
    if request.target is not None and not request.apply and request.save is None:
        raise ValueError(TARGET_WITHOUT_SAVE_MESSAGE)
    if request.directory is not None and not request.directory.is_dir():
        raise ValueError(f"{request.directory} is not a folder")


def server_directory(value: str | None) -> Path | None:
    """A directory named by a remote caller: absolute on the server, or None for the corpus."""
    if value is None:
        return None
    directory = Path(value)
    if not directory.is_absolute():
        raise ValueError(RELATIVE_DIRECTORY_MESSAGE.format(value=value))
    return directory


def _files(directory: Path | None) -> Mapping[str, Path]:
    if directory is None:
        return discover_corpus().files
    return discover_dir(directory).files


def _record_run() -> None:
    """Record the completed run; a failed write must not fail a run that already saved."""
    try:
        mark_analyzed(cfg.data_root)
    except OSError as exc:
        log.warning("Could not record that analyze ran: %s", exc)


def _finish(
    store: ProfileStore,
    signals: CorpusSignals,
    request: AnalyzeRequest,
    cancel: CancelSignal | None,
) -> AnalyzeReport:
    rows = language_rows(signals)
    recommendation = recommend(store, signals, rows)
    if cancel is not None and cancel.is_set():
        raise TaskCancelledError
    saved = _save(store, recommendation, request)
    _record_run()
    return AnalyzeReport(signals, rows, recommendation, saved)


async def run_analysis(
    store: ProfileStore,
    request: AnalyzeRequest,
    *,
    on_progress: DetailedProgressCallback = noop_callback,
    cancel: CancelSignal | None = None,
) -> AnalyzeReport:
    """Read the corpus, recommend a profile, save it when asked, and record the run.

    Raises ``TaskCancelledError`` on cancel, before anything is saved or recorded;
    raises ``ValueError`` when *directory* is not a folder, a *target* comes without
    *apply* or *save*, or the save is refused.
    """
    await asyncio.to_thread(validate_request, request)
    files = await asyncio.to_thread(_files, request.directory)
    signals = await collect_signals(files, on_progress=on_progress, cancel=cancel)
    return await asyncio.to_thread(_finish, store, signals, request, cancel)


@dataclass(frozen=True)
class TipState:
    """Whether the project was analyzed, hid the tip, and would see the tip now."""

    analyzed: bool
    tip_dismissed: bool
    tip_shows: bool


def tip_state(root: Path) -> TipState:
    """The analyze tip state of the project at *root*."""
    state = read_state(root)
    analyzed = state.analyzed_at is not None
    name = read_profile_table(root).name or DEFAULT_PROFILE_NAME
    on_default = profile_key(name) == profile_key(DEFAULT_PROFILE_NAME)
    shows = on_default and not analyzed and not state.tip_dismissed
    return TipState(analyzed, state.tip_dismissed, shows)


def tip_shows(root: Path) -> bool:
    """True when the project at *root* is on Default, never analyzed, and the tip is not hidden."""
    return tip_state(root).tip_shows


def hide_tip(root: Path) -> None:
    """Stop showing the analyze tip for the project at *root*."""
    dismiss_tip(root)
