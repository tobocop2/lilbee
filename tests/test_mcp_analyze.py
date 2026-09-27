"""The MCP analyze tools: the run and hiding the tip, behind mcp_profiles_enabled."""

import asyncio
import threading
import tomllib
from pathlib import Path

import pytest

from lilbee.app import analyze as analyze_mod
from lilbee.app import services as svc_mod
from lilbee.app.profiles import MCP_PROFILES_DISABLED_HINT
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder
from lilbee.core.project_state import STATE_FILE_NAME, read_state
from lilbee.mcp_server import analyze, analyze_dismiss, build_mcp_server
from tests.conftest import make_mock_services

GERMAN = "Die Bundesregierung hat heute beschlossen, dass die neuen Regeln für alle gelten. " * 8
ANALYZE_TOOLS = ["analyze", "analyze_dismiss"]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "project" / ".lilbee"
    cfg.data_dir = cfg.data_root / "data"
    cfg.documents_dir = cfg.data_root / "documents"
    cfg.data_root.mkdir(parents=True)
    cfg.mcp_profiles_enabled = True
    svc_mod.set_services(make_mock_services())
    monkeypatch.setattr(analyze_mod, "ocr_language_supported", lambda code: code == "eng")
    yield
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def notes(tmp_path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    for i in range(3):
        (folder / f"note{i}.md").write_text(f"# Notiz {i}\n\n{GERMAN}", encoding="utf-8")
    return folder


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _forbid_reading(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    async def _collect(files, *, on_progress, cancel):
        seen.append(dict(files))
        raise AssertionError("a refused request read files")

    monkeypatch.setattr(analyze_mod, "collect_signals", _collect)
    return seen


async def test_analyze_returns_the_report_and_saves_nothing(notes):
    report = await analyze(directory=str(notes))
    assert (report["files_total"], report["files_read"]) == (3, 3)
    assert report["recommendation"]["name"] == "Notes and markdown (project)"
    assert report["saved"] is None
    assert "profile" not in _stored()
    assert read_state(cfg.data_root).analyzed_at is not None


async def test_analyze_without_a_directory_reads_the_corpus():
    cfg.documents_dir.mkdir(parents=True)
    (cfg.documents_dir / "owned.md").write_text(GERMAN, encoding="utf-8")
    report = await analyze()
    assert report["file_types"] == {"md": 1}


async def test_apply_saves_and_switches(notes):
    report = await analyze(directory=str(notes), apply=True)
    assert report["saved"]["applied"] is True
    assert _stored()["profile"]["name"] == "Notes and markdown (project)"


async def test_save_writes_the_named_profile_without_switching(notes):
    report = await analyze(directory=str(notes), save="German notes", target=ProfileFolder.PROJECT)
    saved = report["saved"]
    assert (saved["name"], saved["applied"], saved["folder"]) == ("German notes", False, "project")
    assert (cfg.data_root / PROFILES_DIRNAME / "german-notes.toml").exists()
    assert "profile" not in _stored()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": ProfileFolder.GLOBAL}, "target takes effect only with apply or save"),
        ({"directory": "notes"}, "must be an absolute path on the lilbee server: 'notes'"),
        ({"directory": ""}, "must be an absolute path on the lilbee server: ''"),
    ],
)
async def test_a_refused_request_is_an_error_before_any_file_is_read(monkeypatch, kwargs, message):
    seen = _forbid_reading(monkeypatch)
    result = await analyze(**kwargs)
    assert message in result["error"]
    assert seen == []
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


async def test_a_missing_folder_is_an_error(monkeypatch, tmp_path):
    seen = _forbid_reading(monkeypatch)
    result = await analyze(directory=str(tmp_path / "missing"))
    assert "is not a folder" in result["error"]
    assert seen == []


async def test_a_failed_write_returns_the_shared_message(notes, monkeypatch):
    def _refuse(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(analyze_mod, "save_recommended", _refuse)
    result = await analyze(directory=str(notes), save="X")
    assert result == {"error": "Could not save the change: read-only"}


async def test_an_agent_that_cancels_stops_the_run(monkeypatch, notes):
    seen: list[threading.Event] = []
    started = asyncio.Event()

    async def _collect(files, *, on_progress, cancel):
        seen.append(cancel)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(analyze_mod, "collect_signals", _collect)
    task = asyncio.create_task(analyze(directory=str(notes), apply=True))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen[0].is_set()
    assert "profile" not in _stored()
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


def test_dismiss_hides_the_tip_and_returns_the_state():
    result = analyze_dismiss()
    assert result == {"analyzed": False, "tip_dismissed": True, "tip_shows": False}
    assert read_state(cfg.data_root).tip_dismissed is True


def test_a_failed_dismiss_returns_the_shared_message(monkeypatch):
    def _refuse(root: Path) -> None:
        raise OSError("read-only")

    monkeypatch.setattr("lilbee.mcp_server.hide_tip", _refuse)
    assert analyze_dismiss() == {"error": "Could not save the change: read-only"}


async def test_each_tool_refuses_once_the_setting_is_off(notes):
    cfg.mcp_profiles_enabled = False
    assert await analyze(directory=str(notes), apply=True) == {"error": MCP_PROFILES_DISABLED_HINT}
    assert analyze_dismiss() == {"error": MCP_PROFILES_DISABLED_HINT}
    assert not (cfg.data_root / STATE_FILE_NAME).exists()
    assert "profile" not in _stored()


async def test_the_tools_are_off_the_wire_by_default_and_on_when_enabled():
    cfg.mcp_profiles_enabled = False
    off = [t.name for t in await build_mcp_server().list_tools() if t.name.startswith("analyze")]
    assert off == []
    cfg.mcp_profiles_enabled = True
    tools = {t.name: t for t in await build_mcp_server().list_tools()}
    assert sorted(n for n in tools if n.startswith("analyze")) == ANALYZE_TOOLS
    assert set(tools["analyze"].input_schema["properties"]) == {
        "directory",
        "apply",
        "save",
        "target",
    }
