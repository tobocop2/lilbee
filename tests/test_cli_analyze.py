"""The ``lilbee analyze`` command, and the analyze tip on ``init``, ``add`` and ``sync``."""

import json
import signal
import sys
import tomllib
from pathlib import Path
from unittest import mock

import pytest
from rich.text import Text
from typer.testing import CliRunner

from lilbee.app import analyze
from lilbee.app import services as svc_mod
from lilbee.app.analyze import IMAGE_SCAN_NOTE, TIP_TEXT
from lilbee.cli.app import app
from lilbee.cli.commands.analyze import CANCELLED_MESSAGE, OFF_ALONE_MESSAGE
from lilbee.cli.helpers import json_output
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME
from lilbee.core.project_state import STATE_FILE_NAME, read_state
from lilbee.data.types import SyncResult
from lilbee.runtime.cancellation import TaskCancelledError
from tests._unreadable import deny_listing
from tests.conftest import make_mock_services, make_pdf

runner = CliRunner()

GERMAN = "Die Bundesregierung hat heute beschlossen, dass die neuen Regeln für alle gelten. " * 8
REPORT_KEYS = {
    "files_total",
    "documents_total",
    "files_read",
    "files_counted",
    "cap",
    "failed",
    "unreadable_folders",
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


@pytest.mark.skipif(sys.platform == "win32", reason="chmod 000 does not lock a Windows folder")
def test_analyze_names_the_folder_it_could_not_open(project, notes):
    locked = notes / "locked"
    locked.mkdir()
    (locked / "hidden.md").write_text("hidden", encoding="utf-8")
    locked.chmod(0)
    try:
        text = _invoke(project, str(notes))
        data = json.loads(_invoke(project, str(notes), json_mode=True).stdout)
    finally:
        locked.chmod(0o700)
    assert text.exit_code == 0, text.output
    assert f"Could not open folder: {locked}" in text.output.replace("\n", "")
    assert data["unreadable_folders"] == [str(locked)]
    assert data["documents_total"] == 3


def test_analyze_names_a_folder_whose_listing_is_denied(project, notes, monkeypatch):
    """The same report as the chmod test above, with the denied listing stubbed for Windows."""
    locked = notes / "locked"
    locked.mkdir()
    deny_listing(monkeypatch, locked)
    text = _invoke(project, str(notes))
    data = json.loads(_invoke(project, str(notes), json_mode=True).stdout)
    assert text.exit_code == 0, text.output
    assert f"Could not open folder: {locked}" in text.output.replace("\n", "")
    assert data["unreadable_folders"] == [str(locked)]
    assert data["documents_total"] == 3


def _stored(project: Path) -> dict:
    path = project / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_json_output_is_one_object_in_the_report_shape(project, notes):
    result = _invoke(project, str(notes), json_mode=True)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert set(report) == REPORT_KEYS
    assert set(report["recommendation"]) == RECOMMENDATION_KEYS
    assert (report["files_total"], report["documents_total"], report["files_read"]) == (3, 3, 3)
    assert (report["files_counted"], report["cap"]) == (0, 500)
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
    assert "Read 3 of 3 documents." in result.output
    assert "counted" not in result.output
    assert "Recommended: Notes and markdown (project)" in result.output
    assert "fts_language" in result.output
    assert "Run lilbee analyze --apply" in result.output
    assert not (project / PROFILES_DIRNAME).exists()


def test_a_sampled_run_says_so(project, notes):
    cfg_file = project / "config.toml"
    cfg_file.write_text("analyze_max_files = 2\n", encoding="utf-8")
    result = _invoke(project, str(notes))
    assert "Read 2 of 3 documents." in result.output
    assert "analyze_max_files is 2" in result.output


def test_code_and_images_are_counted_and_not_reported_as_read(project, tmp_path):
    folder = tmp_path / "repo"
    folder.mkdir()
    for i in range(3):
        (folder / f"mod{i}.py").write_text("x = 1\n", encoding="utf-8")
    (folder / "scan.png").write_bytes(b"\x89PNG\r\n")
    (folder / "readme.md").write_text(GERMAN, encoding="utf-8")
    result = _invoke(project, str(folder))
    assert result.exit_code == 0, result.output
    assert "Read 1 of 1 documents; counted 4 code, image and archive files." in result.output
    report = json.loads(_invoke(project, str(folder), json_mode=True).output)
    assert (report["files_total"], report["documents_total"], report["files_read"]) == (5, 1, 1)
    assert report["files_counted"] == 4


def test_text_output_shows_the_changes_table(project, notes):
    result = _invoke(project, str(notes))
    assert result.exit_code == 0, result.output
    assert "Takes effect" in result.output
    row = next(
        line for line in result.output.splitlines() if "fts_language" in line and "│" in line
    )
    assert "English" in row
    assert "German" in row


def test_target_without_apply_or_save_is_refused(project, notes):
    with mock.patch.object(analyze, "collect_signals") as collect:
        result = _invoke(project, str(notes), "--target", "global")
    assert result.exit_code == 1
    assert "target takes effect only with apply or save" in result.output
    collect.assert_not_called()
    assert not (project / STATE_FILE_NAME).exists()


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


def test_apply_warns_when_it_leaves_ocr_off_with_a_vision_model(project, notes):
    vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    (project / "config.toml").write_text(f'vision_model = "{vision_model}"\n', encoding="utf-8")
    report = json.loads(_invoke(project, str(notes), "--apply", json_mode=True).output)
    warnings = report["saved"]["warnings"]
    assert len(warnings) == 1
    assert "ocr" in warnings[0]
    (project / "config.toml").write_text(f'vision_model = "{vision_model}"\n', encoding="utf-8")
    result = _invoke(project, str(notes), "--apply")
    assert vision_model in result.output


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
    answer = json.loads(result.output)
    assert set(answer) == {"analyzed", "tip_dismissed", "tip_shows"}
    assert answer["tip_dismissed"] is True
    assert answer["tip_shows"] is False
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
    assert "Read 0 of 7 documents." in result.output
    assert "sampled evenly" not in result.output
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
    assert IMAGE_SCAN_NOTE in result.output


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


class _Shown(SyncResult):
    """A sync result that prints as one known line, whatever it holds."""

    def __rich__(self) -> Text:
        return Text("SYNC RESULT")


_NEW_CONTENT = {"added": ["a.md"]}


def _fake_ingest(cancel_event, *, before_sync, **kwargs):
    from lilbee.cli.commands import ingest_sync

    before_sync()
    ingest_sync.console.print("INGEST STARTED")
    return _Shown(**_NEW_CONTENT)


def _fake_add_paths(paths, console, *, run_sync, **kwargs):
    """Register nothing, then run the add's own sync over a fake ingest and print its result."""
    from lilbee.cli.commands import ingest_sync

    with mock.patch("lilbee.cli.commands.ingest_sync._run_sync", _fake_ingest):
        console.print(run_sync(ingest_sync.RegisterResult()))


def _refused_ingest(cancel_event, *, before_sync, **kwargs):
    before_sync()
    raise RuntimeError("refused")


def _cancelled_ingest(cancel_event, *, before_sync, **kwargs):
    before_sync()
    raise SystemExit(130)


def _fake_refused_add_paths(ingest):
    def _add_paths(paths, console, *, run_sync, **kwargs):
        from lilbee.cli.commands import ingest_sync

        with mock.patch("lilbee.cli.commands.ingest_sync._run_sync", ingest):
            run_sync(ingest_sync.RegisterResult())

    return _add_paths


def test_add_prints_the_tip_after_the_sync_result(project, notes):
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        result = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert flat.index("SYNC RESULT") < flat.index(TIP_TEXT)
    assert flat.count(TIP_TEXT) == 1


@pytest.mark.parametrize("ingest", [_refused_ingest, _cancelled_ingest])
def test_an_add_that_fails_or_cancels_prints_no_tip(project, notes, ingest):
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_refused_add_paths(ingest)):
        result = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert result.exit_code != 0
    assert "lilbee analyze" not in result.output


def test_a_second_add_prints_the_tip_again_until_it_is_hidden(project, notes, tmp_path):
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        first = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert TIP_TEXT in " ".join(first.output.split())
    more = tmp_path / "more"
    more.mkdir()
    (more / "c.md").write_text("# C\n", encoding="utf-8")
    settings.set_value(project, "linked_roots", {"notes": str(notes)})
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        second = runner.invoke(app, ["add", str(more), "--data-dir", str(project)])
    assert second.exit_code == 0, second.output
    assert "INGEST STARTED" in second.output
    assert TIP_TEXT in " ".join(second.output.split())


def test_init_of_a_project_with_documents_prints_the_tip(tmp_path):
    documents = tmp_path / ".lilbee" / "documents"
    documents.mkdir(parents=True)
    (documents / "a.md").write_text("# A\n", encoding="utf-8")
    with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
        result = runner.invoke(app, ["init"])
    assert "Already initialized" in result.output
    assert TIP_TEXT in " ".join(result.output.split())


def _fake_sync(**kwargs):
    from lilbee.cli.commands import ingest_sync

    ingest_sync.console.print("SYNC STARTED")
    return _Shown(**_NEW_CONTENT)


@pytest.mark.parametrize("failure", [RuntimeError("refused"), SystemExit(130)])
def test_a_sync_that_fails_or_cancels_prints_no_tip(project, failure):
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel", side_effect=failure
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert result.exit_code == getattr(failure, "code", 1)
    assert "lilbee analyze" not in result.output


def test_sync_prints_the_tip_once_after_the_sync_result(project):
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel", side_effect=_fake_sync
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert result.exit_code == 0, result.output
    flat = " ".join(result.output.split())
    assert flat.index("SYNC RESULT") < flat.index(TIP_TEXT)
    assert flat.count(TIP_TEXT) == 1


_NOTHING_CHANGED = [
    {},
    {"unchanged": 5},
    {"unchanged": 2, "relocated": ["moved.md"]},
    {"failed": ["bad.md"], "skipped": ["scan.pdf"]},
]


@pytest.mark.parametrize("counts", _NOTHING_CHANGED)
def test_sync_that_added_updated_and_removed_nothing_prints_no_tip(project, counts):
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel",
        return_value=_Shown(**counts),
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "SYNC RESULT" in result.output
    assert "lilbee analyze" not in result.output


@pytest.mark.parametrize(
    "counts", [{"added": ["a.md"]}, {"updated": ["a.md"]}, {"removed": ["a.md"]}]
)
def test_sync_that_changed_a_file_prints_the_tip(project, counts):
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel",
        return_value=_Shown(**counts),
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert TIP_TEXT in " ".join(result.output.split())


@pytest.mark.parametrize("counts", _NOTHING_CHANGED)
def test_add_whose_sync_changed_nothing_prints_no_tip(project, notes, counts, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_NEW_CONTENT", counts)
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        result = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert result.exit_code == 0, result.output
    assert "SYNC RESULT" in result.output
    assert "lilbee analyze" not in result.output


def test_sync_prints_no_tip_after_the_tip_is_hidden(project):
    analyze.hide_tip(project)
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel", side_effect=_fake_sync
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert "SYNC STARTED" in result.output
    assert "lilbee analyze" not in result.output


def test_sync_prints_no_tip_after_analyze(project, notes):
    _invoke(project, str(notes))
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel", side_effect=_fake_sync
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert "SYNC STARTED" in result.output
    assert "lilbee analyze" not in result.output


def test_sync_prints_no_tip_on_another_profile(project):
    (project / "config.toml").write_text('[profile]\nname = "Research papers"\n', encoding="utf-8")
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel", side_effect=_fake_sync
    ):
        result = runner.invoke(app, ["sync", "--data-dir", str(project)])
    assert "SYNC STARTED" in result.output
    assert "lilbee analyze" not in result.output


def test_sync_json_mode_prints_no_tip(project):
    with mock.patch(
        "lilbee.cli.commands.ingest_sync.run_sync_with_signal_cancel", side_effect=_fake_sync
    ):
        result = runner.invoke(app, ["--json", "sync", "--data-dir", str(project)])
    assert TIP_TEXT not in " ".join(result.output.split())


def test_add_after_analyze_prints_no_tip(project, notes):
    _invoke(project, str(notes))
    with mock.patch("lilbee.cli.commands.ingest_sync.add_paths", _fake_add_paths):
        result = runner.invoke(app, ["add", str(notes), "--data-dir", str(project)])
    assert "INGEST STARTED" in result.output
    assert "lilbee analyze" not in result.output


def test_add_json_mode_prints_no_tip(project, notes):
    def _fake_json(file_paths, crawled_paths, *, force, run_sync):
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
