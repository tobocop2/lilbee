"""Analyze rules, the recommended profile, saving it, and the analyze tip."""

import asyncio
import threading
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from lilbee.app import analyze, profiles
from lilbee.app.analyze import (
    _PICK_RULES,
    DEFAULT_FITS_NOTE,
    IMAGE_SCAN_NOTE,
    OCR_LANGUAGE_REASON,
    OCR_MODEL_NOTE,
    AnalyzeRequest,
    LanguageRow,
    Reason,
    derived_name,
    language_rows,
    pick_builtin,
    project_label,
    recommend,
    run_analysis,
    server_directory,
    tip_shows,
    tip_state,
)
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import FtsLanguage
from lilbee.core.profile_files import (
    DEFAULT_PROFILE_NAME,
    PROFILES_DIRNAME,
    ProfileCatalog,
    ProfileFolder,
    ProfileStore,
    profile_key,
)
from lilbee.core.project_state import STATE_FILE_NAME, read_state
from lilbee.core.system import default_data_dir
from lilbee.data.analyze import CorpusSignals, LanguageShare, PdfSignals
from lilbee.data.ingest.ignore import IGNORE_FILENAME
from lilbee.runtime.cancellation import TaskCancelledError

_PDF = PdfSignals(
    files=10,
    pages=100,
    scanned_pages=0,
    scanned_share=0.0,
    files_with_tables=0,
    tables=0,
    median_pages=10.0,
)
_BASE = CorpusSignals(
    files_total=10,
    documents_total=10,
    files_read=10,
    cap=500,
    failed=(),
    file_types={"pdf": 10},
    code_share=0.0,
    pdf=_PDF,
    median_chars=None,
    languages=(),
    image_files=0,
)


@pytest.fixture(autouse=True)
def _tesseract_has(monkeypatch):
    """Tesseract has eng and deu installed; nothing else."""
    monkeypatch.setattr(analyze, "ocr_language_supported", lambda code: code in {"eng", "deu"})


@pytest.fixture
def store() -> ProfileStore:
    return ProfileStore()


def _signals(**changes) -> CorpusSignals:
    pdf_changes = changes.pop("pdf", {})
    return replace(_BASE, pdf=replace(_PDF, **pdf_changes), **changes)


def _langs(*pairs: tuple[str, float]) -> tuple[LanguageShare, ...]:
    return tuple(LanguageShare(code, share) for code, share in pairs)


def _scanned(share: float = 0.4) -> dict:
    return {"scanned_pages": int(share * 100), "scanned_share": share}


def _recommend(store, signals):
    return recommend(store, signals, language_rows(signals))


def _write_config(text: str) -> None:
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    (cfg.data_root / "config.toml").write_text(text, encoding="utf-8")
    settings.overlay_persisted_settings(cfg.data_root)


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@pytest.mark.parametrize(
    ("signals", "builtin"),
    [
        (_signals(pdf=_scanned(0.30)), analyze.SCANNED_ARCHIVE),
        (_signals(pdf=_scanned(0.29)), "Default"),
        (_signals(code_share=0.50), analyze.CODE_REPOSITORY),
        (_signals(code_share=0.49), "Default"),
        (_signals(file_types={"md": 3, "txt": 2, "pdf": 5}), analyze.NOTES_AND_MARKDOWN),
        (_signals(file_types={"md": 4, "pdf": 6}), "Default"),
        (_signals(pdf={"files_with_tables": 2}), analyze.RESEARCH_PAPERS),
        (_signals(pdf={"files_with_tables": 1}), "Default"),
    ],
)
def test_each_builtin_is_picked_at_its_threshold_and_not_below(signals, builtin):
    assert pick_builtin(signals)[0] == builtin


def test_the_first_matching_rule_wins_and_names_its_share():
    builtin, reason = pick_builtin(_signals(pdf=_scanned(0.5), code_share=0.9))
    assert builtin == analyze.SCANNED_ARCHIVE
    assert reason == Reason("profile", "50% of pages are scans")


def test_default_has_no_pick_reason():
    assert pick_builtin(_signals()) == ("Default", None)


def test_no_files_picks_default():
    empty = _signals(files_total=0, file_types={}, pdf={"files": 0, "pages": 0})
    assert pick_builtin(empty) == ("Default", None)


def test_the_most_common_mapped_language_becomes_fts_language(store):
    rec = _recommend(store, _signals(languages=_langs(("deu", 0.61), ("eng", 0.39))))
    assert rec.values["fts_language"] is FtsLanguage.GERMAN
    assert Reason("fts_language", "61% of text files are German") in rec.reasons
    assert rec.name == derived_name("Default", project_label(cfg.data_root))
    row = next(r for r in rec.changes if r.key == "fts_language")
    assert (row.current, row.new) == (FtsLanguage.ENGLISH, FtsLanguage.GERMAN)


def test_language_rows_carry_every_share_stemmer_and_tesseract_support():
    rows = language_rows(_signals(languages=_langs(("deu", 0.6), ("zho", 0.3), ("nob", 0.1))))
    assert rows == (
        LanguageRow("deu", 0.6, FtsLanguage.GERMAN, "deu", True),
        LanguageRow("zho", 0.3, None, "zho", False),
        LanguageRow("nob", 0.1, FtsLanguage.NORWEGIAN, "nor", False),
    )


@pytest.mark.parametrize(
    ("detected", "tesseract"), [("nob", "nor"), ("cmn", "chi_sim"), ("pes", "fas")]
)
def test_detected_codes_map_to_tesseract_language_names(store, monkeypatch, detected, tesseract):
    monkeypatch.setattr(analyze, "ocr_language_supported", lambda code: code == tesseract)
    signals = _signals(pdf=_scanned(0.1), languages=_langs((detected, 1.0)))
    (row,) = language_rows(signals)
    assert (row.ocr_code, row.ocr_supported) == (tesseract, True)
    assert _recommend(store, signals).values["ocr_language"] == [tesseract]


def test_a_missing_tesseract_language_is_named_as_tesseract_names_it(store):
    signals = _signals(pdf=_scanned(0.1), languages=_langs(("nob", 0.5), ("deu", 0.5)))
    rec = _recommend(store, signals)
    assert rec.values["ocr_language"] == ["deu"]
    assert "Tesseract has no language data for nor on this machine." in rec.notes


def test_an_unmapped_top_language_keeps_fts_language_and_says_why(store):
    rec = _recommend(store, _signals(languages=_langs(("zho", 0.8), ("eng", 0.2))))
    assert "fts_language" not in rec.values
    assert "No stemmer for zho; fts_language stays English." in rec.notes
    assert rec.name is None
    assert DEFAULT_FITS_NOTE in rec.notes


def test_a_corpus_in_the_current_language_needs_no_fts_change(store):
    rec = _recommend(store, _signals(languages=_langs(("eng", 1.0))))
    assert rec.values == {}
    assert rec.name is None
    assert rec.changes == ()


def test_your_own_fts_language_counts_as_the_baseline(store):
    _write_config('fts_language = "German"\n')
    rec = _recommend(store, _signals(languages=_langs(("deu", 1.0))))
    assert "fts_language" not in rec.values


def test_a_language_the_current_profile_sets_is_still_recommended(store):
    """The derived profile replaces the current one, so its language must be restated."""
    _write_config('[profile]\nname = "Mine"\n[profile.values]\nfts_language = "German"\n')
    assert cfg.fts_language is FtsLanguage.GERMAN
    rec = _recommend(store, _signals(languages=_langs(("deu", 1.0))))
    assert rec.values["fts_language"] is FtsLanguage.GERMAN
    assert all(row.key != "fts_language" for row in rec.changes)


def test_scans_take_ocr_languages_from_the_text_pages(store):
    signals = _signals(
        pdf=_scanned(0.1), languages=_langs(("deu", 0.6), ("eng", 0.35), ("fra", 0.05))
    )
    rec = _recommend(store, signals)
    assert rec.values["ocr_language"] == ["deu", "eng"]
    assert Reason("ocr_language", OCR_LANGUAGE_REASON) in rec.reasons


def test_an_unsupported_ocr_language_is_left_out_and_noted(store):
    signals = _signals(pdf=_scanned(0.1), languages=_langs(("fra", 0.5), ("deu", 0.5)))
    rec = _recommend(store, signals)
    assert rec.values["ocr_language"] == ["deu"]
    assert "Tesseract has no language data for fra on this machine." in rec.notes


def test_at_most_three_ocr_languages(store):
    langs = _langs(("deu", 0.3), ("eng", 0.3), ("ita", 0.2), ("fra", 0.2))
    signals = _signals(pdf=_scanned(0.1), languages=langs)
    rows = tuple(replace(row, ocr_supported=True) for row in language_rows(signals))
    rec = recommend(store, signals, rows)
    assert rec.values["ocr_language"] == ["deu", "eng", "ita"]


def test_no_scans_means_no_ocr_language(store):
    rec = _recommend(store, _signals(languages=_langs(("deu", 1.0))))
    assert "ocr_language" not in rec.values


def test_an_image_file_counts_as_a_scan_for_ocr_language(store):
    rec = _recommend(store, _signals(image_files=1, languages=_langs(("deu", 1.0))))
    assert rec.values["ocr_language"] == ["deu"]
    assert IMAGE_SCAN_NOTE in rec.notes


def test_a_builtin_that_turns_ocr_off_gets_no_ocr_language(store):
    signals = _signals(code_share=0.9, pdf=_scanned(0.1), languages=_langs(("deu", 1.0)))
    rec = _recommend(store, signals)
    assert rec.builtin == analyze.CODE_REPOSITORY
    assert "ocr_language" not in rec.values


def test_scan_heavy_corpora_without_a_vision_model_get_the_catalog_pointer(store):
    rec = _recommend(store, _signals(pdf=_scanned(0.4)))
    assert OCR_MODEL_NOTE in rec.notes
    cfg.vision_model = "org/repo/vision.gguf"
    assert OCR_MODEL_NOTE not in _recommend(store, _signals(pdf=_scanned(0.4))).notes


def test_a_builtin_pick_shows_its_changes_and_keeps_your_values(store):
    _write_config("layout_detection = false\n")
    rec = _recommend(store, _signals(pdf=_scanned(0.4)))
    assert rec.builtin == analyze.SCANNED_ARCHIVE
    assert rec.values == {"ocr_strategy": "scanned_pages", "layout_detection": True}
    assert rec.kept == ("layout_detection",)
    assert [row.key for row in rec.changes] == ["ocr_strategy"]
    assert rec.reasons[0] == Reason("profile", "40% of pages are scans")


def test_derived_names_are_sanitized_and_cut_to_forty_characters():
    assert derived_name("Scanned archive", "city-records") == "Scanned archive (city-records)"
    assert derived_name("Default", "Q3/reports: v2!") == "Default (Q3-reports- v2)"
    long = derived_name("Notes and markdown", "x" * 60)
    assert len(long) == 40
    assert long == f"Notes and markdown ({'x' * 19})"
    assert derived_name("Default", "!!!") == "Default (project)"


def test_the_project_label_is_the_folder_holding_dot_lilbee(tmp_path):
    assert project_label(tmp_path / "city-records" / ".lilbee") == "city-records"
    assert project_label(tmp_path / "somewhere") == "somewhere"


def _run(store, request=None, signals=None, monkeypatch=None, seen=None):
    async def _collect(files, *, on_progress, cancel):
        if seen is not None:
            seen.append(dict(files))
        return signals or _signals(pdf=_scanned(0.4))

    monkeypatch.setattr(analyze, "collect_signals", _collect)
    return asyncio.run(run_analysis(store, request or AnalyzeRequest()))


def test_a_plain_run_saves_nothing_and_records_the_run(store, monkeypatch):
    report = _run(store, monkeypatch=monkeypatch)
    assert report.saved is None
    assert not (cfg.data_root / PROFILES_DIRNAME).exists()
    assert "profile" not in _stored()
    state = read_state(cfg.data_root)
    assert state.analyzed_at is not None
    assert state.tip_dismissed is False


def test_apply_saves_the_derived_profile_switches_and_keeps_your_values(store, monkeypatch):
    _write_config("layout_detection = false\n")
    report = _run(store, AnalyzeRequest(apply=True), monkeypatch=monkeypatch)
    saved = report.saved
    assert saved is not None and saved.applied
    assert saved.name == report.recommendation.name
    assert saved.folder is ProfileFolder.PROJECT
    assert saved.path.parent == cfg.data_root / PROFILES_DIRNAME
    written = tomllib.loads(saved.path.read_text(encoding="utf-8"))
    assert written["profile"]["description"] == profiles.ANALYZE_DESCRIPTION
    stored = _stored()
    assert stored["profile"]["name"] == saved.name
    assert stored["layout_detection"] is False
    assert cfg.layout_detection is False
    assert cfg.ocr_strategy == "scanned_pages"
    assert read_state(cfg.data_root).tip_dismissed is True


def test_apply_saves_the_derived_profile_and_carries_the_ocr_warning(store, monkeypatch):
    _write_config("enable_ocr = false\n")
    profiles.apply(store, "Notes and markdown")
    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    report = _run(store, AnalyzeRequest(apply=True), monkeypatch=monkeypatch)
    saved = report.saved
    assert saved is not None and saved.applied
    assert len(saved.warnings) == 1
    assert "enable_ocr" in saved.warnings[0]


def test_save_writes_under_the_name_without_switching(store, monkeypatch):
    report = _run(store, AnalyzeRequest(save="Records"), monkeypatch=monkeypatch)
    assert report.saved is not None
    assert (report.saved.name, report.saved.applied) == ("Records", False)
    assert report.saved.path.name == "records.toml"
    assert "profile" not in _stored()
    assert cfg.ocr_strategy == "auto"


def test_a_second_run_replaces_its_own_earlier_file(store, monkeypatch):
    first = _run(store, AnalyzeRequest(apply=True), monkeypatch=monkeypatch)
    second = _run(store, AnalyzeRequest(apply=True), monkeypatch=monkeypatch)
    assert first.saved is not None and second.saved is not None
    assert second.saved.path == first.saved.path
    assert sorted(p.name for p in (cfg.data_root / PROFILES_DIRNAME).glob("*.toml")) == [
        first.saved.path.name
    ]


def test_a_hand_made_profile_with_the_name_is_refused_and_nothing_is_recorded(store, monkeypatch):
    profiles.new(store, "Records", ProfileFolder.PROJECT)
    with pytest.raises(ValueError, match="--save NAME"):
        _run(store, AnalyzeRequest(save="Records"), monkeypatch=monkeypatch)
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


def test_a_broken_file_with_the_name_is_refused(store, monkeypatch):
    folder = cfg.data_root / PROFILES_DIRNAME
    folder.mkdir(parents=True)
    (folder / "records.toml").write_text("not toml [", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists in the project folder"):
        _run(store, AnalyzeRequest(save="Records"), monkeypatch=monkeypatch)


def test_the_global_data_root_saves_to_the_global_folder(store, monkeypatch):
    cfg.data_root = default_data_dir()
    report = _run(store, AnalyzeRequest(save="Records"), monkeypatch=monkeypatch)
    assert report.saved is not None
    assert report.saved.folder is ProfileFolder.GLOBAL
    assert report.saved.path.parent == default_data_dir() / PROFILES_DIRNAME


def test_default_fitting_saves_nothing_on_a_plain_run(store, monkeypatch):
    report = _run(store, signals=_signals(), monkeypatch=monkeypatch)
    assert report.recommendation.name is None
    assert report.saved is None


def test_applying_a_fitting_default_switches_back_to_default(store, monkeypatch):
    profiles.apply(store, "Research papers")
    assert cfg.table_extraction is True
    report = _run(store, AnalyzeRequest(apply=True), signals=_signals(), monkeypatch=monkeypatch)
    assert report.recommendation.name is None
    changed = {row.key for row in report.recommendation.changes}
    assert changed == {"table_extraction", "layout_detection"}
    assert report.saved is not None
    assert (report.saved.name, report.saved.folder, report.saved.applied) == (
        "Default",
        ProfileFolder.BUILTIN,
        True,
    )
    assert _stored()["profile"]["name"] == "Default"
    assert cfg.table_extraction is False


def test_applying_a_fitting_default_carries_the_ocr_warning(store, monkeypatch):
    _write_config("enable_ocr = false\n")
    profiles.apply(store, "Notes and markdown")
    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    report = _run(store, AnalyzeRequest(apply=True), signals=_signals(), monkeypatch=monkeypatch)
    assert report.recommendation.name is None
    assert report.saved is not None
    assert len(report.saved.warnings) == 1
    assert "enable_ocr" in report.saved.warnings[0]


def test_a_cancelled_run_records_nothing(store, monkeypatch):
    async def _cancelled(files, *, on_progress, cancel):
        raise TaskCancelledError

    monkeypatch.setattr(analyze, "collect_signals", _cancelled)
    with pytest.raises(TaskCancelledError):
        asyncio.run(run_analysis(store, AnalyzeRequest(apply=True)))
    assert not (cfg.data_root / STATE_FILE_NAME).exists()
    assert "profile" not in _stored()


def test_a_directory_is_walked_instead_of_the_corpus(store, monkeypatch, tmp_path):
    folder = tmp_path / "elsewhere"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.md").write_text("x", encoding="utf-8")
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (cfg.documents_dir / "owned.md").write_text("x", encoding="utf-8")
    seen: list[dict] = []
    _run(store, AnalyzeRequest(directory=folder), monkeypatch=monkeypatch, seen=seen)
    _run(store, AnalyzeRequest(), monkeypatch=monkeypatch, seen=seen)
    assert [sorted(files) for files in seen] == [["sub/a.md"], ["owned.md"]]


def test_no_directory_reads_linked_folders_with_the_ignore_rules(store, monkeypatch, tmp_path):
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    (cfg.documents_dir / "owned.md").write_text("x", encoding="utf-8")
    (cfg.documents_dir / "draft.md").write_text("x", encoding="utf-8")
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    (cfg.data_root / IGNORE_FILENAME).write_text("draft.md\n", encoding="utf-8")
    linked = tmp_path / "linked"
    (linked / "build").mkdir(parents=True)
    (linked / "paper.md").write_text("x", encoding="utf-8")
    (linked / "build" / "out.md").write_text("x", encoding="utf-8")
    (linked / IGNORE_FILENAME).write_text("build/\n", encoding="utf-8")
    cfg.linked_roots = {"linked": str(linked)}
    seen: list[dict] = []
    _run(store, AnalyzeRequest(), monkeypatch=monkeypatch, seen=seen)
    assert [sorted(files) for files in seen] == [["linked/paper.md", "owned.md"]]


def test_a_target_without_apply_or_save_is_refused_before_reading(store, monkeypatch):
    seen: list[dict] = []
    with pytest.raises(ValueError, match="target takes effect only with apply or save"):
        _run(store, AnalyzeRequest(target=ProfileFolder.GLOBAL), monkeypatch=monkeypatch, seen=seen)
    assert seen == []
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


def test_every_builtin_analyze_can_pick_is_a_shipped_builtin(store):
    names = {rule.builtin for rule in _PICK_RULES} | {DEFAULT_PROFILE_NAME}
    assert len(names) == 5
    catalog = store.scan()
    for name in names:
        entry = catalog.find(name)
        assert entry is not None, name
        assert entry.folder is ProfileFolder.BUILTIN, name
        assert entry.file is not None, name


def test_a_directory_that_is_not_a_folder_is_refused(store, monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="is not a folder"):
        _run(store, AnalyzeRequest(directory=tmp_path / "missing"), monkeypatch=monkeypatch)


def test_a_fresh_default_project_gets_the_tip_until_it_is_hidden():
    root = cfg.data_root
    assert tip_shows(root) is True
    analyze.hide_tip(root)
    assert tip_shows(root) is False


def test_a_project_with_a_document_gets_no_tip():
    assert tip_shows(cfg.data_root) is True
    (cfg.documents_dir / "notes").mkdir(parents=True)
    assert tip_shows(cfg.data_root) is True, "an empty folder is not a document"
    (cfg.documents_dir / "notes" / "a.md").write_text("# A\n", encoding="utf-8")
    assert tip_shows(cfg.data_root) is False


def test_a_project_with_an_added_root_gets_no_tip(tmp_path):
    assert tip_shows(cfg.data_root) is True
    settings.set_value(cfg.data_root, "linked_roots", {"papers": str(tmp_path / "papers")})
    assert tip_shows(cfg.data_root) is False


def test_another_project_is_judged_by_its_own_documents(tmp_path):
    (cfg.documents_dir).mkdir(parents=True, exist_ok=True)
    (cfg.documents_dir / "a.md").write_text("# A\n", encoding="utf-8")
    other = tmp_path / "other" / ".lilbee"
    other.mkdir(parents=True)
    assert tip_shows(cfg.data_root) is False
    assert tip_shows(other) is True
    (other / "documents").mkdir()
    (other / "documents" / "b.md").write_text("# B\n", encoding="utf-8")
    assert tip_shows(other) is False


def test_an_analyzed_project_gets_no_tip(store, monkeypatch):
    _run(store, monkeypatch=monkeypatch)
    assert tip_shows(cfg.data_root) is False


def test_a_project_on_another_profile_gets_no_tip(store):
    profiles.apply(store, "Research papers")
    (cfg.data_root / STATE_FILE_NAME).unlink()
    assert tip_shows(cfg.data_root) is False


def test_a_project_that_names_default_still_gets_the_tip():
    _write_config('[profile]\nname = "default"\n')
    assert profile_key("default") == profile_key("Default")
    assert tip_shows(cfg.data_root) is True


def test_every_profile_apply_hides_the_tip(store):
    profiles.apply(store, "Research papers")
    assert read_state(cfg.data_root).tip_dismissed is True


def test_save_as_hides_the_tip(store):
    profiles.save_as("Mine", ProfileFolder.PROJECT)
    assert read_state(cfg.data_root).tip_dismissed is True


def test_a_failed_tip_write_does_not_fail_the_apply(store, monkeypatch, caplog):
    def _refuse(root: Path) -> None:
        raise OSError("read-only")

    monkeypatch.setattr("lilbee.app.settings.dismiss_tip", _refuse)
    profiles.apply(store, "Research papers")
    assert _stored()["profile"]["name"] == "Research papers"
    assert "Could not record that the analyze tip is hidden" in caplog.text


def test_a_missing_builtin_profile_is_an_error_not_a_guess():
    class _Empty(ProfileStore):
        def scan(self) -> ProfileCatalog:
            return ProfileCatalog(entries=())

    with pytest.raises(ValueError, match="built-in profile Default is missing or broken"):
        _recommend(_Empty(), _signals())


def test_an_empty_save_name_is_refused_not_ignored(store, monkeypatch):
    with pytest.raises(ValueError, match="Bad name"):
        _run(store, AnalyzeRequest(save=""), monkeypatch=monkeypatch)


def test_a_cancel_after_reading_saves_and_records_nothing(store, monkeypatch):
    cancel = threading.Event()

    async def _collect(files, *, on_progress, cancel):
        cancel.set()
        return _signals(pdf=_scanned(0.4))

    monkeypatch.setattr(analyze, "collect_signals", _collect)
    with pytest.raises(TaskCancelledError):
        asyncio.run(run_analysis(store, AnalyzeRequest(apply=True), cancel=cancel))
    assert not (cfg.data_root / PROFILES_DIRNAME).exists()
    assert "profile" not in _stored()
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


def test_a_server_directory_must_be_absolute(tmp_path):
    assert server_directory(None) is None
    assert server_directory(str(tmp_path)) == tmp_path
    for value in ("notes", "", "../notes"):
        with pytest.raises(ValueError, match="must be an absolute path on the lilbee server"):
            server_directory(value)
    with pytest.raises(ValueError, match="'notes'"):
        server_directory("notes")


def test_tip_state_follows_analyze_dismiss_and_the_active_profile(store, monkeypatch):
    root = cfg.data_root
    assert tip_state(root) == analyze.TipState(analyzed=False, tip_dismissed=False, tip_shows=True)
    analyze.hide_tip(root)
    assert tip_state(root) == analyze.TipState(analyzed=False, tip_dismissed=True, tip_shows=False)
    (root / STATE_FILE_NAME).unlink()
    _run(store, monkeypatch=monkeypatch)
    assert tip_state(root) == analyze.TipState(analyzed=True, tip_dismissed=False, tip_shows=False)
    (root / STATE_FILE_NAME).unlink()
    _write_config('[profile]\nname = "Research papers"\n')
    assert tip_state(root) == analyze.TipState(analyzed=False, tip_dismissed=False, tip_shows=False)
