"""The ``lilbee analyze`` command, and the analyze tip on ``init`` and ``add``."""

import json
import signal
import tomllib
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.app import analyze
from lilbee.app import services as svc_mod
from lilbee.app.analyze import TIP_TEXT
from lilbee.cli.app import app
from lilbee.cli.commands.analyze import CANCELLED_MESSAGE, OFF_ALONE_MESSAGE
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME
from lilbee.core.project_state import STATE_FILE_NAME, read_state
from lilbee.runtime.cancellation import TaskCancelledError
from tests.conftest import make_mock_services, make_pdf

runner = CliRunner()

GERMAN = "Die Bundesregierung hat heute beschlossen, dass die neuen Regeln für alle gelten. " * 8
REPORT_KEYS = {
    "files_total",
    "files_read",
    "cap",
    "failed",
    "file_types",
    "code_share",
    "pdf",
    "median_chars",
    "languages",
    "recommendation",
    "saved",
}
RECOMMENDATION_KEYS = {"builtin", "name", "values", "changes", "kept", "reasons", "notes"}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    svc_mod.set_services(make_mock_services())
    monkeypatch.setattr(analyze, "ocr_language_supported", lambda code: code == "eng")
    yield
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def project(tmp_path) -> Path:
    root = tmp_path / "project" / ".lilbee"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def notes(tmp_path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    for i in range(3):
        (folder / f"note{i}.md").write_text(f"# Notiz {i}\n\n{GERMAN}", encoding="utf-8")
    return folder


def _invoke(project: Path, *args: str, json_mode: bool = False):
    prefix = ["--json"] if json_mode else []
    return runner.invoke(app, [*prefix, "analyze", *args, "--data-dir", str(project)])


def _stored(project: Path) -> dict:
    path = project / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_json_output_is_one_object_in_the_report_shape(project, notes):
    result = _invoke(project, str(notes), json_mode=True)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert set(report) == REPORT_KEYS
    assert set(report["recommendation"]) == RECOMMENDATION_KEYS
    assert (report["files_total"], report["files_read"], report["cap"]) == (3, 3, 500)
    assert report["file_types"] == {"md": 3}
    assert report["languages"] == [
        {"code": "deu", "share": 1.0, "fts_language": "German", "ocr_supported": False}
    ]
    rec = report["recommendation"]
    assert rec["builtin"] == "Notes and markdown"
    assert rec["name"] == "Notes and markdown (project)"
    assert rec["values"]["fts_language"] == "German"
    change = next(c for c in rec["changes"] if c["key"] == "fts_language")
    assert change == {
        "key": "fts_language",
        "current": "English",
        "current_source": "built_in",
        "new": "German",
        "effect": "reindex",
    }
    assert report["saved"] is None
    assert read_state(project).analyzed_at is not None


def test_text_output_shows_the_reading_and_the_recommendation(project, notes):
    result = _invoke(project, str(notes))
    assert result.exit_code == 0, result.output
    assert "Read 3 of 3 files." in result.output
    assert "Recommended: Notes and markdown (project)" in result.output
    assert "fts_language" in result.output
    assert "Run lilbee analyze --apply" in result.output
    assert not (project / PROFILES_DIRNAME).exists()


def test_a_sampled_run_says_so(project, notes):
    cfg_file = project / "config.toml"
    cfg_file.write_text("analyze_max_files = 2\n", encoding="utf-8")
    result = _invoke(project, str(notes))
    assert "Read 2 of 3 files." in result.output
    assert "analyze_max_files is 2" in result.output


def test_apply_saves_and_switches(project, notes):
    report = json.loads(_invoke(project, str(notes), "--apply", json_mode=True).output)
    saved = report["saved"]
    assert saved["applied"] is True
    assert saved["folder"] == "project"
    assert Path(saved["path"]).is_file()
    assert _stored(project)["profile"]["name"] == "Notes and markdown (project)"
    assert read_state(project).tip_dismissed is True


def test_apply_text_names_the_switch(project, notes):
    result = _invoke(project, str(notes), "--apply")
    assert "This project now uses Notes and markdown (project):" in result.output


def test_save_writes_without_switching(project, notes):
    result = _invoke(project, str(notes), "--save", "Vault")
    assert result.exit_code == 0, result.output
    assert "Saved Vault:" in result.output
    assert (project / PROFILES_DIRNAME / "vault.toml").is_file()
    assert "profile" not in _stored(project)


def test_save_to_the_global_folder(project, notes):
    report = json.loads(
        _invoke(project, str(notes), "--save", "Vault", "--target", "global", json_mode=True).output
    )
    assert report["saved"]["folder"] == "global"


def test_a_refused_save_exits_1_with_the_reason(project, notes):
    runner.invoke(
        app, ["profile", "new", "Vault", "--target", "project", "--data-dir", str(project)]
    )
    result = _invoke(project, str(notes), "--save", "Vault", json_mode=True)
    assert result.exit_code == 1
    assert "--save NAME" in json.loads(result.output)["error"]


def test_a_folder_that_does_not_exist_exits_1(project, tmp_path):
    result = _invoke(project, str(tmp_path / "missing"))
    assert result.exit_code == 1
    assert "is not a folder" in result.output


def test_off_hides_the_tip_and_reads_nothing(project):
    with mock.patch.object(analyze, "collect_signals") as collect:
        result = _invoke(project, "--off", json_mode=True)
    assert result.exit_code == 0
    assert json.loads(result.output) == {"tip_dismissed": True}
    assert read_state(project).tip_dismissed is True
    assert read_state(project).analyzed_at is None
    collect.assert_not_called()


def test_off_text(project):
    result = _invoke(project, "--off")
    assert "The analyze tip is hidden for this project." in result.output


def test_off_refuses_other_options(project):
    result = _invoke(project, "--off", "--apply")
    assert result.exit_code == 1
    assert OFF_ALONE_MESSAGE in result.output
    assert not (project / STATE_FILE_NAME).exists()


def test_off_reports_a_failed_write(project):
    with mock.patch.object(analyze, "dismiss_tip", side_effect=OSError("read-only")):
        result = _invoke(project, "--off", json_mode=True)
    assert result.exit_code == 1
    assert "read-only" in json.loads(result.output)["error"]


def test_ctrl_c_cancels_between_batches_and_saves_nothing(project, notes, monkeypatch):
    async def _interrupted(files, *, on_progress, cancel):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        if cancel.is_set():
            raise TaskCancelledError
        raise AssertionError("Ctrl+C did not set the cancel token")

    monkeypatch.setattr(analyze, "collect_signals", _interrupted)
    result = _invoke(project, str(notes), "--apply")
    assert result.exit_code == 1
    assert CANCELLED_MESSAGE in result.output
    assert not (project / STATE_FILE_NAME).exists()
    assert "profile" not in _stored(project)


def test_a_failed_profile_write_exits_1(project, notes, monkeypatch):
    monkeypatch.setattr(analyze, "save_recommended", mock.Mock(side_effect=OSError("disk full")))
    result = _invoke(project, str(notes), "--apply")
    assert result.exit_code == 1
    assert "Could not save the change: disk full" in result.output


def test_a_failed_run_record_warns_and_keeps_the_saved_profile(project, notes, monkeypatch, caplog):
    monkeypatch.setattr(analyze, "mark_analyzed", mock.Mock(side_effect=OSError("disk full")))
    result = _invoke(project, str(notes), "--apply")
    assert result.exit_code == 0, result.output
    assert _stored(project)["profile"]["name"] == "Notes and markdown (project)"
    assert "Could not record that analyze ran: disk full" in caplog.text


def test_more_failures_than_shown_are_counted(project, tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    for i in range(7):
        (folder / f"bad{i}.pdf").write_bytes(b"%PDF-1.4 not a pdf")
    result = _invoke(project, str(folder))
    assert "7 files could not be read:" in result.output
    assert "and 2 more; --json lists them all" in result.output


def test_text_output_lists_failures_languages_and_notes(project, tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "bad.pdf").write_bytes(b"%PDF-1.4 not a pdf")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n")
    result = _invoke(project, str(folder))
    assert result.exit_code == 0, result.output
    assert "1 files could not be read:" in result.output
    assert "bad.pdf" in result.output
    assert "No language detected" in result.output
    assert "Each image file counts as one scanned page." in result.output


def test_progress_bar_moves_only_on_analyze_events():
    from rich.progress import Progress

    from lilbee.cli.commands import analyze as analyze_cli
    from lilbee.runtime.progress import AnalyzeEvent, EventType, FileStartEvent

    with mock.patch.object(Progress, "update") as update, analyze_cli._progress() as on_progress:
        on_progress(EventType.FILE_START, FileStartEvent(file="x", total_files=1, current_file=1))
        on_progress(EventType.ANALYZE, AnalyzeEvent(done=1, total=2, file="x"))
    assert update.call_count == 1
    assert update.call_args.kwargs == {"completed": 1, "total": 2}


def test_init_prints_the_tip_in_text_mode_only(tmp_path):
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        text = runner.invoke(app, ["init"])
    assert TIP_TEXT in " ".join(text.output.split())
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        again = runner.invoke(app, ["init"])
    assert "Already initialized" in again.output
    assert TIP_TEXT in " ".join(again.output.split())


def test_init_json_stays_one_object(tmp_path):
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["--json", "init"])
    assert json.loads(result.output)["created"] is True


def test_init_skips_the_tip_for_a_dismissed_project(tmp_path):
    root = tmp_path / ".lilbee"
    root.mkdir()
    analyze.hide_tip(root)
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init"])
    assert "Already initialized" in result.output
    assert "lilbee analyze" not in result.output


def _fake_add_paths(paths, console, **kwargs):
    console.print("INGEST STARTED")


def test_add_prints_the_tip_before_ingest_starts(project, notes):
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        result = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert TIP_TEXT in flat
    assert flat.index(TIP_TEXT) < flat.index("INGEST STARTED")


def test_add_after_analyze_prints_no_tip(project, notes):
    _invoke(project, str(notes))
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        result = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert "INGEST STARTED" in result.output
    assert "lilbee analyze" not in result.output


def test_add_json_mode_prints_no_tip(project, notes):
    def _fake_json(file_paths, crawled_paths, *, force):
        json_output({"command": "add"})

    with mock.patch("lilbee.cli.commands.ingest_sync._add_json_mode", _fake_json):
        result = runner.invoke(app, ["--json", "add", str(notes), "--data-dir", str(project)])
    assert json.loads(result.output) == {"command": "add"}


def test_text_output_shows_pdf_length_and_the_values_you_keep(project, tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    for i in range(2):
        (folder / f"paper{i}.pdf").write_bytes(make_pdf(pages=4))
    (project / "config.toml").write_text('fts_language = "German"\n', encoding="utf-8")
    (folder / "de.md").write_text(GERMAN, encoding="utf-8")
    result = _invoke(project, str(folder), "--save", "Papers")
    assert result.exit_code == 0, result.output
    assert "Median PDF length: 4 pages" in result.output
    assert "Median length of other files:" in result.output


def test_text_output_names_the_values_you_keep(project, notes):
    (project / "config.toml").write_text("chunk_size = 700\n", encoding="utf-8")
    result = _invoke(project, str(notes), "--apply")
    assert result.exit_code == 0, result.output
    assert "Keeps your values of: chunk_size" in result.output
    assert _stored(project)["chunk_size"] == 700
