"""The ``lilbee profile`` command group: every profile operation, as text and as --json."""

import json
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.cli.app import app
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME
from lilbee.core.system import default_data_dir
from tests.conftest import make_mock_services

runner = CliRunner()

CREDITED = (
    '[profile]\nname = "Court filings"\ndescription = "Scanned court PDFs."\n'
    'authors = [{ name = "Jane Doe", github = "janedoe" }]\n'
    'tested_on = "4,000 county court filings"\n[values]\nchunk_size = 768\n'
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    svc_mod.set_services(make_mock_services())
    yield
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def project(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _invoke(project: Path, *args: str, json_mode: bool = False):
    prefix = ["--json"] if json_mode else []
    return runner.invoke(app, [*prefix, "profile", *args, "--data-dir", str(project)])


def _json(project: Path, *args: str) -> dict:
    result = _invoke(project, *args, json_mode=True)
    return json.loads(result.output)


def _global_dir() -> Path:
    return default_data_dir() / PROFILES_DIRNAME


def _write(folder: Path, stem: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _stored(project: Path) -> dict:
    path = project / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_show_without_a_name_shows_this_projects_profile(project):
    result = _invoke(project, "show")
    assert result.exit_code == 0, result.output
    assert "Profile: Default" in result.output
    assert _json(project, "show")["name"] == "Default"


def test_show_a_name_prints_its_credit_tested_on_and_values(project):
    _write(_global_dir(), "court-filings", CREDITED)
    result = _invoke(project, "show", "court-filings")
    assert result.exit_code == 0, result.output
    assert "Court filings" in result.output
    assert "by Jane Doe (@janedoe)" in result.output
    assert "Tested on 4,000 county court filings" in result.output
    assert "chunk_size = 768" in result.output
    shown = _json(project, "show", "court filings")
    assert shown["credit"] == "by Jane Doe (@janedoe)"
    assert shown["tested_on"] == "4,000 county court filings"


def test_list_marks_the_active_profile_and_shows_credit_and_broken_files(project):
    _write(_global_dir(), "court-filings", CREDITED)
    _write(_global_dir(), "broken", "[values]\nchat_model = 'x'\n")
    assert _invoke(project, "apply", "Court filings").exit_code == 0
    result = _invoke(project, "list")
    assert result.exit_code == 0, result.output
    marked = [line for line in result.output.splitlines() if " * " in line]
    assert len(marked) == 1 and "Court filings" in marked[0]
    assert "by Jane Doe (@janedoe)" in result.output
    assert "Broken: Profiles cannot set chat_model" in result.output
    names = {p["name"]: p for p in _json(project, "list")["profiles"]}
    assert names["Court filings"]["credit"] == "by Jane Doe (@janedoe)"
    assert names["broken"]["valid"] is False


def test_diff_prints_values_as_a_profile_file_writes_them(project):
    result = _invoke(project, "diff", "scanned-archive")
    assert result.exit_code == 0, result.output
    assert '"auto" (built in)' in result.output
    assert '"scanned_pages"' in result.output
    assert "OcrPageStrategy" not in result.output
    rows = {r["key"]: r for r in _json(project, "diff", "Scanned archive")["changes"]}
    assert rows["ocr_strategy"] == {
        "key": "ocr_strategy",
        "current": "auto",
        "current_source": "built_in",
        "new": "scanned_pages",
        "effect": "new_files_only",
    }


def test_apply_records_the_profile_and_says_to_rebuild(project, monkeypatch):
    rebuilds = MagicMock()
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync.rebuild_or_raise", rebuilds)
    _write(_global_dir(), "court-filings", CREDITED)
    result = _invoke(project, "apply", "court filings")
    assert result.exit_code == 0, result.output
    assert "Applied Court filings." in result.output
    assert "Run lilbee rebuild" in result.output
    assert _stored(project)["profile"] == {"name": "Court filings", "values": {"chunk_size": 768}}
    rebuilds.assert_not_called()


def test_apply_with_reindex_rebuilds_when_a_change_needs_it(project, monkeypatch):
    rebuilds = MagicMock(return_value=MagicMock(added=["a.pdf", "b.pdf"]))
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync.rebuild_or_raise", rebuilds)
    _write(_global_dir(), "court-filings", CREDITED)
    payload = _json(project, "apply", "Court filings", "--reindex")
    assert payload["reindex_required"] is True
    assert payload["reindexed"] == 2
    rebuilds.assert_called_once_with()


def test_apply_with_reindex_skips_the_rebuild_when_nothing_needs_it(project, monkeypatch):
    rebuilds = MagicMock()
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync.rebuild_or_raise", rebuilds)
    _write(_global_dir(), "tables", "[values]\ntop_k = 13\n")
    result = _invoke(project, "apply", "tables", "--reindex")
    assert result.exit_code == 0, result.output
    assert "rebuild" not in result.output.lower()
    rebuilds.assert_not_called()


def test_apply_with_reindex_prints_the_rebuild_count(project, monkeypatch):
    rebuilds = MagicMock(return_value=MagicMock(added=["a.pdf"]))
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync.rebuild_or_raise", rebuilds)
    _write(_global_dir(), "court-filings", CREDITED)
    result = _invoke(project, "apply", "Court filings", "--reindex")
    assert "Rebuilt: 1 documents ingested" in result.output


def test_apply_with_reindex_reports_both_when_the_rebuild_fails(project, monkeypatch):
    rebuilds = MagicMock(side_effect=RuntimeError("A sync is already running"))
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync.rebuild_or_raise", rebuilds)
    _write(_global_dir(), "court-filings", CREDITED)
    result = _invoke(project, "apply", "Court filings", "--reindex", json_mode=True)
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["name"] == "Court filings"
    assert payload["reindex_required"] is True
    assert payload["reindexed"] is None
    assert payload["reindex_error"] == "A sync is already running"


def test_apply_with_reindex_exits_nonzero_when_the_rebuild_fails(project, monkeypatch):
    rebuilds = MagicMock(side_effect=RuntimeError("A sync is already running"))
    monkeypatch.setattr("lilbee.cli.commands.ingest_sync.rebuild_or_raise", rebuilds)
    _write(_global_dir(), "court-filings", CREDITED)
    result = _invoke(project, "apply", "Court filings", "--reindex")
    assert result.exit_code == 1
    assert "Applied Court filings." in result.output
    assert "A sync is already running" in result.output


def test_a_refusal_prints_the_core_message_and_exits_1(project):
    result = _invoke(project, "apply", "nope")
    assert result.exit_code == 1
    assert "Error: No profile named 'nope'" in result.output
    refused = _invoke(project, "delete", "Scanned archive", json_mode=True)
    assert refused.exit_code == 1
    assert json.loads(refused.output) == {
        "error": "Scanned archive ships with lilbee and cannot be changed; duplicate it instead"
    }


def test_a_failed_file_write_prints_the_shared_message(project, monkeypatch):
    def fail(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", "/x/mine.toml")

    monkeypatch.setattr("lilbee.app.profiles.delete", fail)
    result = _invoke(project, "delete", "mine", json_mode=True)
    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "error": "Could not save the change: [Errno 13] Permission denied: '/x/mine.toml'"
    }


def test_new_writes_to_the_global_folder_unless_told_project(project):
    wrote = _json(project, "new", "Short notes", "--from", "notes-and-markdown")
    assert Path(wrote["path"]) == _global_dir() / "short-notes.toml"
    assert wrote["folder"] == "global"
    local = _invoke(project, "new", "Local one", "--target", "project")
    assert local.exit_code == 0, local.output
    assert (project / PROFILES_DIRNAME / "local-one.toml").exists()


def test_save_update_and_discard(project):
    (project / "config.toml").write_text("chunk_size = 900\n", encoding="utf-8")
    saved = _invoke(project, "save", "Mine", "--target", "project")
    assert saved.exit_code == 0, saved.output
    assert "It now holds your settings of: chunk_size" in saved.output
    path = project / PROFILES_DIRNAME / "mine.toml"
    assert tomllib.loads(path.read_text(encoding="utf-8"))["values"] == {"chunk_size": 900}
    (project / "config.toml").write_text(
        "top_k = 13\n" + (project / "config.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    updated = _json(project, "update")
    assert (updated["name"], updated["absorbed"]) == ("Mine", ["top_k"])
    (project / "config.toml").write_text(
        "top_k = 14\n" + (project / "config.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    discarded = _invoke(project, "discard")
    assert "Removed your values of: top_k" in discarded.output
    assert "top_k" not in _stored(project)
    assert _json(project, "discard") == {"dropped": []}
    assert "no values of profile settings" in _invoke(project, "discard").output


def test_duplicate_rename_and_delete(project):
    dup = _invoke(project, "duplicate", "Scanned archive", "My scans")
    assert dup.exit_code == 0, dup.output
    assert (_global_dir() / "my-scans.toml").exists()
    renamed = _json(project, "rename", "my scans", "Old scans")
    assert Path(renamed["path"]) == _global_dir() / "old-scans.toml"
    deleted = _invoke(project, "delete", "old scans")
    assert deleted.exit_code == 0, deleted.output
    assert "Deleted Old scans" in deleted.output
    assert not (_global_dir() / "old-scans.toml").exists()


def test_export_then_import_round_trips_a_profile(project, tmp_path):
    _write(_global_dir(), "court-filings", CREDITED)
    out = tmp_path / "out"
    out.mkdir()
    exported = _json(project, "export", "Court filings", str(out))
    assert Path(exported["path"]) == out / "court-filings.toml"
    assert (exported["name"], exported["folder"]) == ("Court filings", "global")
    refused = _invoke(project, "export", "Court filings", str(out))
    assert refused.exit_code == 1
    assert "already exists" in refused.output
    imported = _invoke(project, "import", str(out / "court-filings.toml"), "--target", "project")
    assert imported.exit_code == 0, imported.output
    copy = tomllib.loads(
        (project / PROFILES_DIRNAME / "court-filings.toml").read_text(encoding="utf-8")
    )
    assert copy["profile"]["authors"] == [{"name": "Jane Doe", "github": "janedoe"}]
    shown = _invoke(project, "export", "Court filings", str(out), "--overwrite")
    assert "Exported Court filings" in shown.output


def test_validate_lists_every_problem_and_exits_1(project, tmp_path):
    bad = _write(tmp_path, "bad", "[values]\nchat_model = 'x'\nchunk_size = -1\n")
    result = _invoke(project, "validate", str(bad))
    assert result.exit_code == 1
    assert "bad is not a valid profile:" in result.output
    assert "Profiles cannot set chat_model" in result.output
    good = _write(tmp_path, "good", "[values]\ntop_k = 13\n")
    assert _json(project, "validate", str(good)) == {"name": "good", "valid": True, "problems": []}
    builtin = _invoke(project, "validate", str(good), "--folder", "builtin")
    assert builtin.exit_code == 1
    assert "Sets retrieval setting top_k without evidence" in builtin.output
    assert _invoke(project, "validate", str(good), "--folder", "community").exit_code == 2
    ok = _invoke(project, "validate", str(good))
    assert "good is a valid profile." in ok.output


def test_the_profile_group_is_registered_on_the_cli():
    result = runner.invoke(app, ["profile", "--help"])
    assert result.exit_code == 0
    for command in ("show", "list", "diff", "apply", "new", "save", "update", "discard"):
        assert command in result.output
    for command in ("duplicate", "rename", "delete", "export", "import", "validate"):
        assert command in result.output


def test_sharing_is_export_and_import_with_no_share_command(project):
    result = runner.invoke(app, ["profile", "share", "Default", "--data-dir", str(project)])
    assert result.exit_code == 2
    assert "No such command 'share'" in result.output


def test_list_names_a_profile_a_higher_folder_hides(project):
    _write(_global_dir(), "court-filings", CREDITED)
    _write(project / PROFILES_DIRNAME, "court-filings", CREDITED)
    result = _invoke(project, "list")
    assert "Hidden by the" in result.output
    hidden = [p for p in _json(project, "list")["profiles"] if p["folder"] == "global"]
    assert [(p["name"], p["shadowed_by"]) for p in hidden] == [("Court filings", "project")]


def test_show_says_when_the_applied_file_changed_or_broke(project):
    path = _write(_global_dir(), "court-filings", CREDITED)
    _invoke(project, "apply", "Court filings")
    path.write_text(CREDITED.replace("768", "769"), encoding="utf-8")
    changed = _invoke(project, "show")
    assert "Its file changed since it was applied" in changed.output
    path.write_text("[values]\nchat_model = 'x'\n", encoding="utf-8")
    broken = _invoke(project, "show")
    assert "Its file is broken; the copy recorded on apply stays in use." in broken.output
    assert "Profiles cannot set chat_model" in broken.output


def test_diff_says_when_nothing_changes_and_names_the_values_it_keeps(project):
    _invoke(project, "apply", "Scanned archive")
    assert "No settings change." in _invoke(project, "diff", "scanned-archive").output
    (project / "config.toml").write_text(
        "layout_detection = false\n" + (project / "config.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    kept = _invoke(project, "diff", "research-papers")
    assert "Keeps your values of: layout_detection" in kept.output


def test_duplicate_writes_to_the_project_folder_when_told(project):
    result = _invoke(project, "duplicate", "Scanned archive", "My scans", "--target", "project")
    assert result.exit_code == 0, result.output
    assert (project / PROFILES_DIRNAME / "my-scans.toml").exists()
    assert not _global_dir().exists()
    shown = _json(project, "duplicate", "Scanned archive", "Other", "--target", "project")
    assert (Path(shown["path"]), shown["folder"]) == (
        project / PROFILES_DIRNAME / "other.toml",
        "project",
    )


def test_show_without_a_name_prints_the_active_profiles_credit(project):
    _write(_global_dir(), "court-filings", CREDITED)
    assert _invoke(project, "apply", "Court filings").exit_code == 0
    result = _invoke(project, "show")
    assert result.exit_code == 0, result.output
    assert "Profile: Court filings" in result.output
    assert "Scanned court PDFs." in result.output
    assert "by Jane Doe (@janedoe)" in result.output
    assert "Tested on 4,000 county court filings" in result.output
