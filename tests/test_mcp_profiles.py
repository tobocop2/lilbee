"""The MCP profile tools: list, show, apply and manage, behind mcp_profiles_enabled."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from lilbee.app import services as svc_mod
from lilbee.app.profiles import MCP_PROFILES_DISABLED_HINT
from lilbee.core.config import Config, cfg
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder
from lilbee.core.system import default_data_dir
from lilbee.mcp_server import (
    ProfileAction,
    build_mcp_server,
    profile_apply,
    profile_list,
    profile_manage,
    profile_show,
)
from tests.conftest import make_mock_services

CREDITED = (
    '[profile]\nname = "Court filings"\n'
    'authors = [{ name = "Jane Doe", github = "janedoe" }]\n'
    'tested_on = "4,000 county court filings"\n[values]\nchunk_size = 768\n'
)
PROFILE_TOOLS = ["profile_apply", "profile_list", "profile_manage", "profile_show"]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path / "project"
    cfg.data_dir = cfg.data_root / "data"
    cfg.mcp_profiles_enabled = True
    svc_mod.set_services(make_mock_services())
    yield
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _global_dir() -> Path:
    return default_data_dir() / PROFILES_DIRNAME


def _write(folder: Path, stem: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_list_returns_the_active_profile_and_every_profile_with_credit():
    _write(_global_dir(), "court-filings", CREDITED)
    result = profile_list()
    assert result["active"]["name"] == "Default"
    court = next(p for p in result["profiles"] if p["name"] == "Court filings")
    assert court["credit"] == "by Jane Doe (@janedoe)"
    assert court["tested_on"] == "4,000 county court filings"


def test_show_returns_the_profile_and_its_diff():
    _write(_global_dir(), "court-filings", CREDITED)
    result = profile_show("court filings")
    assert result["profile"]["credit"] == "by Jane Doe (@janedoe)"
    assert [row["key"] for row in result["diff"]["changes"]] == ["chunk_size"]
    assert result["diff"]["changes"][0]["effect"] == "reindex"


def test_show_a_broken_profile_has_no_diff_and_names_the_reason():
    _write(_global_dir(), "broken", "[values]\nchat_model = 'x'\n")
    result = profile_show("broken")
    assert result["diff"] is None
    assert result["profile"]["error"] == "Profiles cannot set chat_model"


def test_apply_records_the_profile_and_reports_the_reindex():
    _write(_global_dir(), "court-filings", CREDITED)
    result = profile_apply("Court filings")
    assert result["reindex_required"] is True
    assert _stored()["profile"]["name"] == "Court filings"
    assert profile_list()["active"]["profile"]["credit"] == "by Jane Doe (@janedoe)"


def test_a_refusal_returns_the_core_message():
    assert profile_apply("nope") == {"error": "No profile named 'nope'"}
    assert profile_manage(ProfileAction.DELETE, name="Scanned archive") == {
        "error": "Scanned archive ships with lilbee and cannot be changed; duplicate it instead"
    }


def test_a_failed_file_write_returns_the_shared_message(monkeypatch):
    def fail(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", "/x/mine.toml")

    monkeypatch.setattr("lilbee.app.profiles.delete", fail)
    assert profile_manage(ProfileAction.DELETE, name="mine") == {
        "error": "Could not save the change: [Errno 13] Permission denied: '/x/mine.toml'"
    }


def test_manage_new_save_update_and_discard():
    wrote = profile_manage(ProfileAction.NEW, name="Short notes", from_profile="notes-and-markdown")
    assert Path(wrote["path"]) == _global_dir() / "short-notes.toml"
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    (cfg.data_root / "config.toml").write_text("chunk_size = 900\n", encoding="utf-8")
    saved = profile_manage(ProfileAction.SAVE, name="Mine", folder=ProfileFolder.PROJECT)
    assert (saved["folder"], saved["absorbed"]) == ("project", ["chunk_size"])
    config = cfg.data_root / "config.toml"
    config.write_text("top_k = 13\n" + config.read_text(encoding="utf-8"), encoding="utf-8")
    assert profile_manage(ProfileAction.UPDATE)["absorbed"] == ["top_k"]
    config.write_text("top_k = 14\n" + config.read_text(encoding="utf-8"), encoding="utf-8")
    assert profile_manage(ProfileAction.DISCARD) == {"dropped": ["top_k"]}


def test_manage_duplicate_rename_and_delete():
    dup = profile_manage(ProfileAction.DUPLICATE, name="Scanned archive", new_name="My scans")
    assert Path(dup["path"]) == _global_dir() / "my-scans.toml"
    renamed = profile_manage(ProfileAction.RENAME, name="my scans", new_name="Old scans")
    assert Path(renamed["path"]) == _global_dir() / "old-scans.toml"
    deleted = profile_manage(ProfileAction.DELETE, name="old scans")
    assert deleted["name"] == "Old scans"
    assert not (_global_dir() / "old-scans.toml").exists()


def test_manage_export_import_and_validate_use_the_path(tmp_path):
    _write(_global_dir(), "court-filings", CREDITED)
    out = tmp_path / "out"
    out.mkdir()
    exported = profile_manage(ProfileAction.EXPORT, name="Court filings", path=str(out))
    assert Path(exported["path"]) == out / "court-filings.toml"
    imported = profile_manage(
        ProfileAction.IMPORT, path=exported["path"], folder=ProfileFolder.PROJECT
    )
    assert Path(imported["path"]) == cfg.data_root / PROFILES_DIRNAME / "court-filings.toml"
    checked = profile_manage(
        ProfileAction.VALIDATE, path=exported["path"], folder=ProfileFolder.COMMUNITY
    )
    assert checked == {"name": "Court filings", "valid": True, "problems": []}


@pytest.mark.parametrize(
    "action", [ProfileAction.EXPORT, ProfileAction.IMPORT, ProfileAction.VALIDATE]
)
def test_a_file_action_without_a_path_is_refused(action):
    result = profile_manage(action, name="Scanned archive")
    assert result == {"error": "path is required to export, import or validate a profile"}


def test_every_action_has_a_handler():
    from lilbee.mcp_server import _MANAGE_ACTIONS

    assert set(_MANAGE_ACTIONS) == set(ProfileAction)


_CALLS = {
    "profile_list": lambda: profile_list(),
    "profile_show": lambda: profile_show("Scanned archive"),
    "profile_apply": lambda: profile_apply("Scanned archive"),
    "profile_manage": lambda: profile_manage(ProfileAction.DISCARD),
}


@pytest.mark.parametrize("tool", sorted(_CALLS))
def test_each_tool_refuses_once_the_setting_is_off(tool):
    cfg.mcp_profiles_enabled = False
    assert _CALLS[tool]() == {"error": MCP_PROFILES_DISABLED_HINT}
    assert "profile" not in _stored()


async def test_the_tools_are_off_the_wire_by_default_and_on_when_enabled():
    assert Config.model_fields["mcp_profiles_enabled"].default is False
    cfg.mcp_profiles_enabled = False
    off = [t.name for t in await build_mcp_server().list_tools() if t.name.startswith("profile")]
    assert off == []
    cfg.mcp_profiles_enabled = True
    tools = {t.name: t for t in await build_mcp_server().list_tools()}
    assert sorted(n for n in tools if n.startswith("profile")) == PROFILE_TOOLS
    actions = tools["profile_manage"].input_schema["$defs"]["ProfileAction"]["enum"]
    assert sorted(actions) == sorted(a.value for a in ProfileAction)
