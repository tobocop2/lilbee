"""The /api/analyze routes: the streamed run, the tip state, and hiding the tip."""

import asyncio
import json
import threading
import tomllib
from pathlib import Path

import pytest
from litestar.testing import TestClient

from lilbee.app import analyze
from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME
from lilbee.core.project_state import STATE_FILE_NAME, read_state
from lilbee.runtime.progress import AnalyzeEvent, EventType
from lilbee.server.handlers import analyze as analyze_handlers
from tests.conftest import make_mock_services

GERMAN = "Die Bundesregierung hat heute beschlossen, dass die neuen Regeln für alle gelten. " * 8
# the drain generator is finalized a few loop ticks after the stream closes
DISCONNECT_TICKS = 200
HAND_MADE = '[profile]\nname = "My notes"\n[values]\nchunk_size = 768\n'


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "project" / ".lilbee"
    cfg.data_dir = cfg.data_root / "data"
    cfg.documents_dir = cfg.data_root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    cfg.data_root.mkdir(parents=True)
    set_services(make_mock_services())
    monkeypatch.setattr(analyze, "ocr_language_supported", lambda code: code == "eng")
    yield tmp_path
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def client():
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


@pytest.fixture
def notes(tmp_path) -> Path:
    folder = tmp_path / "notes"
    folder.mkdir()
    for i in range(3):
        (folder / f"note{i}.md").write_text(f"# Notiz {i}\n\n{GERMAN}", encoding="utf-8")
    return folder


def _events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _forbid_reading(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    async def _collect(files, *, on_progress, cancel):
        seen.append(dict(files))
        raise AssertionError("a refused request read files")

    monkeypatch.setattr(analyze, "collect_signals", _collect)
    return seen


def test_a_run_streams_progress_in_order_then_the_report(client, notes):
    cfg.batch_extraction_size = 1
    assert not notes.is_relative_to(cfg.data_root)
    resp = client.post("/api/analyze", json={"directory": str(notes)})
    assert resp.status_code == 201
    events = _events(resp.text)
    names = [name for name, _ in events]
    assert names == [EventType.ANALYZE] * 3 + ["done"]
    progress = [data for name, data in events if name == EventType.ANALYZE]
    assert [(p["done"], p["total"]) for p in progress] == [(1, 3), (2, 3), (3, 3)]
    assert [p["file"] for p in progress] == ["note0.md", "note1.md", "note2.md"]
    report = events[-1][1]
    assert (report["files_total"], report["files_read"], report["cap"]) == (3, 3, 500)
    assert report["recommendation"]["name"] == "Notes and markdown (project)"
    assert report["saved"] is None
    assert "profile" not in _stored()
    assert read_state(cfg.data_root).analyzed_at is not None


def test_no_body_reads_the_corpus(client):
    cfg.documents_dir.mkdir(parents=True)
    (cfg.documents_dir / "owned.md").write_text(GERMAN, encoding="utf-8")
    events = _events(client.post("/api/analyze").text)
    assert events[-1][0] == "done"
    assert events[-1][1]["file_types"] == {"md": 1}


def test_apply_saves_the_recommendation_and_switches(client, notes):
    resp = client.post("/api/analyze", json={"directory": str(notes), "apply": True})
    saved = _events(resp.text)[-1][1]["saved"]
    assert saved["name"] == "Notes and markdown (project)"
    assert saved["applied"] is True
    assert saved["folder"] == "project"
    assert _stored()["profile"]["name"] == "Notes and markdown (project)"


def test_save_writes_a_named_profile_to_the_target_without_switching(client, notes, tmp_path):
    body = {"directory": str(notes), "save": "German notes", "target": "project"}
    saved = _events(client.post("/api/analyze", json=body).text)[-1][1]["saved"]
    assert (saved["name"], saved["applied"], saved["folder"]) == ("German notes", False, "project")
    assert Path(saved["path"]) == cfg.data_root / PROFILES_DIRNAME / "german-notes.toml"
    assert "profile" not in _stored()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"target": "global"}, "target takes effect only with apply or save"),
        ({"directory": "notes"}, "must be an absolute path on the lilbee server: 'notes'"),
        ({"directory": ""}, "must be an absolute path on the lilbee server: ''"),
    ],
)
def test_a_refused_request_is_a_400_before_any_file_is_read(client, monkeypatch, body, message):
    seen = _forbid_reading(monkeypatch)
    resp = client.post("/api/analyze", json=body)
    assert resp.status_code == 400
    assert message in resp.json()["detail"]
    assert seen == []
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


def test_a_missing_folder_is_a_400(client, monkeypatch, tmp_path):
    seen = _forbid_reading(monkeypatch)
    resp = client.post("/api/analyze", json={"directory": str(tmp_path / "missing")})
    assert resp.status_code == 400
    assert "is not a folder" in resp.json()["detail"]
    assert seen == []


def test_a_refused_save_ends_the_stream_with_an_error(client, notes):
    folder = cfg.data_root / PROFILES_DIRNAME
    folder.mkdir(parents=True)
    (folder / "my-notes.toml").write_text(HAND_MADE, encoding="utf-8")
    body = {"directory": str(notes), "save": "My notes", "target": "project"}
    events = _events(client.post("/api/analyze", json=body).text)
    assert events[-1][0] == "error"
    assert "--save" in events[-1][1]["message"]
    assert (folder / "my-notes.toml").read_text(encoding="utf-8") == HAND_MADE


def test_a_failed_write_ends_the_stream_with_the_shared_message(client, notes, monkeypatch):
    def _refuse(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(analyze, "save_recommended", _refuse)
    events = _events(client.post("/api/analyze", json={"directory": str(notes), "save": "X"}).text)
    assert events[-1] == ("error", {"message": "Could not save the change: read-only"})


async def test_a_client_that_disconnects_cancels_the_run(monkeypatch, notes):
    seen: list[threading.Event] = []
    release = asyncio.Event()

    async def _collect(files, *, on_progress, cancel):
        seen.append(cancel)
        on_progress(EventType.ANALYZE, AnalyzeEvent(done=1, total=3, file="note0.md"))
        await release.wait()
        raise AssertionError("the run continued after the client left")

    monkeypatch.setattr(analyze, "collect_signals", _collect)
    request = analyze.AnalyzeRequest(directory=notes, apply=True)
    stream = analyze_handlers.analyze_stream(request)
    first = await anext(stream)
    assert first.startswith(f"event: {EventType.ANALYZE}")
    await stream.aclose()
    for _ in range(DISCONNECT_TICKS):
        if seen[0].is_set():
            break
        await asyncio.sleep(0.01)
    assert seen[0].is_set()
    assert "profile" not in _stored()
    assert not (cfg.data_root / STATE_FILE_NAME).exists()


def test_state_answers_the_tip_condition_and_dismiss_hides_it(client, notes):
    fresh = {"analyzed": False, "tip_dismissed": False, "tip_shows": True}
    assert client.get("/api/analyze/state").json() == fresh
    resp = client.post("/api/analyze/dismiss")
    assert resp.status_code == 200
    hidden = {"analyzed": False, "tip_dismissed": True, "tip_shows": False}
    assert resp.json() == hidden
    assert client.get("/api/analyze/state").json() == hidden
    assert read_state(cfg.data_root).tip_dismissed is True


def test_state_reports_a_completed_run(client, notes):
    client.post("/api/analyze", json={"directory": str(notes)})
    state = client.get("/api/analyze/state").json()
    assert state == {"analyzed": True, "tip_dismissed": False, "tip_shows": False}


def test_a_failed_dismiss_is_a_503(client, monkeypatch):
    def _refuse(root: Path) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(analyze_handlers, "hide_tip", _refuse)
    resp = client.post("/api/analyze/dismiss")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Could not save the change: read-only"
