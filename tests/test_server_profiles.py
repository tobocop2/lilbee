"""The /api/profiles routes: every profile operation over HTTP, and URL-name path safety."""

import asyncio
import tomllib
from pathlib import Path
from urllib.parse import quote

import pytest
from litestar.testing import TestClient

from lilbee.app import profiles
from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME
from lilbee.core.system import default_data_dir
from tests.conftest import make_mock_services

CREDITED = (
    '[profile]\nname = "Court filings"\ndescription = "Scanned court PDFs."\n'
    'authors = [{ name = "Jane Doe", github = "janedoe" }, { name = "Sam Roe" }]\n'
    'tested_on = "4,000 county court filings"\n[values]\nchunk_size = 768\n'
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "project"
    cfg.data_dir = cfg.data_root / "data"
    cfg.documents_dir = cfg.data_root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    set_services(make_mock_services())
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


def _global_dir() -> Path:
    return default_data_dir() / PROFILES_DIRNAME


def _project_dir() -> Path:
    return cfg.data_root / PROFILES_DIRNAME


def _write(folder: Path, stem: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _url(name: str, suffix: str = "") -> str:
    return f"/api/profiles/{quote(name, safe='')}{suffix}"


def test_list_names_every_profile_with_its_credit(client):
    _write(_global_dir(), "court-filings", CREDITED)
    _write(_global_dir(), "broken", "[values]\nchat_model = 'x'\n")
    resp = client.get("/api/profiles")
    assert resp.status_code == 200
    by_name = {p["name"]: p for p in resp.json()["profiles"]}
    court = by_name["Court filings"]
    assert court["folder"] == "global"
    assert court["credit"] == "by Jane Doe (@janedoe), Sam Roe"
    assert court["authors"] == [
        {"name": "Jane Doe", "github": "janedoe"},
        {"name": "Sam Roe", "github": None},
    ]
    assert court["tested_on"] == "4,000 county court filings"
    assert court["values"] == {"chunk_size": 768}
    assert by_name["broken"]["valid"] is False
    assert by_name["broken"]["error"] == "Profiles cannot set chat_model"
    assert {"Default", "Scanned archive"} <= set(by_name)


def test_active_is_default_until_an_apply_then_carries_the_applied_profile(client):
    resp = client.get("/api/profiles/active")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Default"
    assert resp.json()["status"] == "current"
    assert resp.json()["profile"]["folder"] == "builtin"
    _write(_global_dir(), "court-filings", CREDITED)
    assert client.post(_url("court filings", "/apply")).status_code == 200
    active = client.get("/api/profiles/active").json()
    assert (active["name"], active["values"]) == ("Court filings", {"chunk_size": 768})
    assert active["profile"]["credit"] == "by Jane Doe (@janedoe), Sam Roe"
    assert active["profile"]["tested_on"] == "4,000 county court filings"


def test_show_resolves_a_name_loosely_and_404s_an_unknown_one(client):
    _write(_global_dir(), "court-filings", CREDITED)
    resp = client.get(_url("COURT-filings"))
    assert resp.status_code == 200
    assert resp.json()["name"] == "Court filings"
    assert resp.json()["credit"] == "by Jane Doe (@janedoe), Sam Roe"
    missing = client.get(_url("nope"))
    assert missing.status_code == 404
    assert missing.json()["detail"] == "No profile named 'nope'"


def test_diff_lists_changes_with_their_effect_and_apply_records_the_profile(client):
    diff = client.get(_url("scanned-archive", "/diff"))
    assert diff.status_code == 200
    rows = {row["key"]: row for row in diff.json()["changes"]}
    assert rows["layout_detection"]["new"] is True
    assert rows["layout_detection"]["current_source"] == "built_in"
    assert rows["ocr_strategy"]["effect"] == "new_files_only"
    assert diff.json()["untouched_count"] > 0
    applied = client.post(_url("Scanned archive", "/apply"))
    assert applied.status_code == 200
    assert applied.json()["name"] == "Scanned archive"
    assert "ocr_strategy" in applied.json()["new_files_only"]
    assert _stored()["profile"]["name"] == "Scanned archive"
    assert cfg.layout_detection is True


def test_apply_reports_a_reindex_when_a_reindex_setting_changes(client):
    _write(_global_dir(), "court-filings", CREDITED)
    applied = client.post(_url("Court filings", "/apply")).json()
    assert applied["reindex_required"] is True
    assert [(r["key"], r["effect"]) for r in applied["changes"]] == [("chunk_size", "reindex")]


def test_save_as_writes_the_profile_to_the_target_folder_and_switches_to_it(client):
    cfg.data_root.mkdir(parents=True)
    (cfg.data_root / "config.toml").write_text("chunk_size = 900\n", encoding="utf-8")
    resp = client.post("/api/profiles", json={"name": "Mine", "target": "project"})
    assert resp.status_code == 200
    body = resp.json()
    assert (body["name"], body["folder"], body["absorbed"]) == ("Mine", "project", ["chunk_size"])
    assert Path(body["path"]) == _project_dir() / "mine.toml"
    assert _stored() == {"profile": {"name": "Mine", "values": {"chunk_size": 900}}}
    default_target = client.post("/api/profiles", json={"name": "Other"}).json()
    assert Path(default_target["path"]) == _global_dir() / "other.toml"


def test_update_writes_into_the_named_active_profile_and_refuses_another_name(client):
    path = _write(_global_dir(), "mine", '[profile]\nname = "Mine"\n[values]\n')
    client.post(_url("mine", "/apply"))
    (cfg.data_root / "config.toml").write_text(
        "chunk_size = 700\n" + (cfg.data_root / "config.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    refused = client.put(_url("Scanned archive"))
    assert refused.status_code == 400
    assert refused.json()["detail"] == "Scanned archive is not this project's profile; Mine is"
    resp = client.put(_url("MINE"))
    assert resp.status_code == 200
    assert resp.json()["absorbed"] == ["chunk_size"]
    assert tomllib.loads(path.read_text(encoding="utf-8"))["values"] == {"chunk_size": 700}


def test_discard_drops_your_profile_settings(client):
    cfg.data_root.mkdir(parents=True)
    (cfg.data_root / "config.toml").write_text("chunk_size = 700\n", encoding="utf-8")
    resp = client.post("/api/profiles/discard")
    assert resp.status_code == 200
    assert resp.json() == {"dropped": ["chunk_size"]}
    assert _stored() == {}


def test_new_writes_a_template_from_a_profile(client):
    resp = client.post(
        "/api/profiles/new", json={"name": "Short notes", "from_profile": "notes-and-markdown"}
    )
    assert resp.status_code == 200
    written = tomllib.loads(Path(resp.json()["path"]).read_text(encoding="utf-8"))
    assert written["values"]["chunk_size"] == 384
    assert Path(resp.json()["path"]).parent == _global_dir()


def test_duplicate_rename_and_delete(client):
    dup = client.post(_url("Scanned archive", "/duplicate"), json={"new_name": "My scans"})
    assert dup.status_code == 200
    assert Path(dup.json()["path"]) == _global_dir() / "my-scans.toml"
    renamed = client.patch(_url("my scans"), json={"new_name": "Old scans"})
    assert renamed.status_code == 200
    assert Path(renamed.json()["path"]) == _global_dir() / "old-scans.toml"
    assert not (_global_dir() / "my-scans.toml").exists()
    refused = client.delete(_url("Scanned archive"))
    assert refused.status_code == 400
    assert refused.json()["detail"] == (
        "Scanned archive ships with lilbee and cannot be changed; duplicate it instead"
    )
    deleted = client.delete(_url("old scans"))
    assert deleted.status_code == 200
    assert deleted.json()["name"] == "Old scans"
    assert not (_global_dir() / "old-scans.toml").exists()


def test_export_downloads_a_clean_file_named_by_the_slug(client):
    _write(_global_dir(), "Court Filings File", CREDITED)
    resp = client.get(_url("court filings", "/export"))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/toml")
    assert 'filename="court-filings.toml"' in resp.headers["content-disposition"]
    exported = tomllib.loads(resp.text)
    assert exported["profile"]["authors"][0] == {"name": "Jane Doe", "github": "janedoe"}
    assert exported["values"] == {"chunk_size": 768}


def test_import_copies_uploaded_text_into_the_folder_whatever_the_file_name(client, tmp_path):
    resp = client.post(
        "/api/profiles/import",
        json={"content": "[values]\nchunk_size = 640\n", "filename": "../../My Upload.toml"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "my-upload"
    assert Path(resp.json()["path"]) == _global_dir() / "my-upload.toml"
    assert not (tmp_path / "My Upload.toml").exists()
    again = client.post(
        "/api/profiles/import",
        json={"content": "[values]\nchunk_size = 641\n", "filename": "my-upload.toml"},
    )
    assert again.status_code == 400
    assert again.json()["detail"].startswith("A profile named my-upload already exists")
    replaced = client.post(
        "/api/profiles/import",
        json={
            "content": "[values]\nchunk_size = 641\n",
            "filename": "my-upload.toml",
            "overwrite": True,
        },
    )
    assert replaced.status_code == 200


def test_validate_reports_every_problem_without_writing(client):
    resp = client.post(
        "/api/profiles/validate",
        json={"content": "[values]\nchat_model = 'x'\nchunk_size = -1\n", "filename": "bad.toml"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert (body["name"], body["valid"]) == ("bad", False)
    assert body["problems"][0] == "Profiles cannot set chat_model"
    assert body["problems"][1].startswith("Bad value for chunk_size")
    ok = client.post(
        "/api/profiles/validate", json={"content": "[values]\n", "filename": "fine.toml"}
    ).json()
    assert ok == {"name": "fine", "valid": True, "problems": []}
    assert not _global_dir().exists()


def test_validate_applies_the_community_evidence_rule_when_asked(client):
    body = {"content": "[values]\ntop_k = 13\n", "filename": "c.toml"}
    assert client.post("/api/profiles/validate", json=body).json()["valid"] is True
    community = client.post("/api/profiles/validate", json={**body, "folder": "community"})
    assert community.json()["problems"] == ["Sets retrieval setting top_k without evidence"]


@pytest.mark.parametrize(
    ("name", "detail"),
    [
        ("../outside", "Not Found"),
        ("..%2Foutside", "Not Found"),
        ("%2E%2E%2Foutside", "Not Found"),
        ("a/../../outside", "Not Found"),
        ("outside file", "No profile named 'outside file'"),
        ("outside&x", "No profile named 'outside&x'"),
        ("café", "No profile named 'café'"),
    ],
)
def test_a_url_name_never_reaches_a_file_outside_the_profile_folders(
    client, tmp_path, name, detail
):
    outside = _write(tmp_path, "outside", "[values]\nchunk_size = 900\n")
    in_root = _write(cfg.data_root, "outside", "[values]\nchunk_size = 900\n")
    for method, suffix in (("get", ""), ("delete", ""), ("post", "/apply"), ("get", "/export")):
        resp = getattr(client, method)(_url(name, suffix))
        assert (resp.status_code, resp.json()["detail"]) == (404, detail), (method, suffix)
    assert outside.exists() and in_root.exists()
    assert "profile" not in _stored()


@pytest.mark.parametrize("name", ["../outside", "../../outside", "..", "/tmp/outside"])
def test_a_name_with_path_parts_is_only_ever_looked_up(tmp_path, name):
    from litestar.exceptions import NotFoundException

    from lilbee.server.handlers import profiles as handlers

    outside = _write(cfg.data_root, "outside", "[values]\n")
    for call in (handlers.delete_profile, handlers.apply_profile, handlers.export_profile):
        with pytest.raises(NotFoundException, match=f"No profile named {name!r}"):
            asyncio.run(call(name))
    assert outside.exists()


def test_an_encoded_name_reaches_the_lookup_decoded(client):
    _write(_global_dir(), "stem", '[profile]\nname = "A_B (x) y"\n[values]\n')
    resp = client.get(_url("A_B (x) y"))
    assert resp.status_code == 200
    assert resp.json()["name"] == "A_B (x) y"
    assert client.get(_url("a-b-x-y")).json()["name"] == "A_B (x) y"


@pytest.mark.parametrize("new_name", ["../escape", "a/b", "..", "x" * 41, ""])
def test_a_bad_new_name_is_refused_and_writes_nothing(client, tmp_path, new_name):
    resp = client.post(_url("Scanned archive", "/duplicate"), json={"new_name": new_name})
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("Bad name")
    assert not _global_dir().exists()
    assert not (tmp_path / "escape.toml").exists()


def test_a_failed_file_write_is_a_503_with_the_shared_message(client, monkeypatch):
    def fail(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", "/x/mine.toml")

    monkeypatch.setattr(profiles, "delete", fail)
    resp = client.delete(_url("mine"))
    assert resp.status_code == 503
    assert resp.json()["detail"] == (
        "Could not save the change: [Errno 13] Permission denied: '/x/mine.toml'"
    )


def test_a_profile_operation_runs_off_the_event_loop(client, monkeypatch):
    seen: list[bool] = []
    real = profiles.list_profiles

    def record(store):
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return real(store)

    monkeypatch.setattr(profiles, "list_profiles", record)
    assert client.get("/api/profiles").status_code == 200
    assert seen == [False]
