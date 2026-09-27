"""Profile use cases: list, show, the active profile, diff and apply."""

import tomllib
from pathlib import Path

import pytest

from lilbee.app import profiles
from lilbee.app.profiles import ProfileEffect, ProfileStatus
from lilbee.app.services import get_services, set_services
from lilbee.app.settings import apply_profile_layer, apply_settings_update
from lilbee.config_meta import PUBLIC_CONFIG_FIELDS, WRITABLE_CONFIG_FIELDS
from lilbee.core import profile_files, settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import OcrPageStrategy, SettingSource
from lilbee.core.config.resolve import PROFILE_FIELDS, ROOT_DERIVED_FIELDS
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder, ProfileStore
from lilbee.providers.roles import MODEL_ROLE_FIELDS
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def _services():
    set_services(make_mock_services())
    yield
    set_services(None)


@pytest.fixture
def store() -> ProfileStore:
    return get_services().profile_store


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_config(text: str) -> None:
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    (cfg.data_root / "config.toml").write_text(text, encoding="utf-8")
    settings.overlay_persisted_settings(cfg.data_root)


def _global_profile(stem: str, text: str) -> Path:
    folder = profile_files.default_data_dir() / PROFILES_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_services_carry_a_profile_store_that_sees_the_builtins(store):
    assert isinstance(store, ProfileStore)
    names = [e.name for e in profiles.list_profiles(store).entries]
    assert "Scanned archive" in names
    entry = profiles.show(store, "scanned-archive")
    assert entry.folder is ProfileFolder.BUILTIN
    with pytest.raises(ValueError, match="No profile named 'nope'"):
        profiles.show(store, "nope")


def test_apply_copies_values_into_profile_table(store):
    _write_config("top_k = 7\n")
    result = profiles.apply(store, "scanned archive")
    assert _stored() == {
        "top_k": 7,
        "profile": {
            "name": "Scanned archive",
            "values": {"ocr_strategy": "scanned_pages", "layout_detection": True},
        },
    }
    assert cfg.ocr_strategy is OcrPageStrategy.SCANNED_PAGES
    assert cfg.layout_detection is True
    assert result.name == "Scanned archive"
    assert result.reindex_required is True
    assert result.new_files_only == ("ocr_strategy",)
    assert {row.key for row in result.changes} == {"ocr_strategy", "layout_detection"}


def test_apply_default_records_name_only(store):
    profiles.apply(store, "Notes and markdown")
    assert cfg.chunk_size == 384
    profiles.apply(store, "default")
    assert _stored() == {"profile": {"name": "Default"}}
    assert (cfg.chunk_size, cfg.chunk_overlap, cfg.enable_ocr) == (512, 100, None)


def test_apply_keeps_user_and_env_values_and_lists_them_as_kept(store, monkeypatch):
    monkeypatch.setenv("LILBEE_ENABLE_OCR", "true")
    _write_config("chunk_size = 900\ntop_k = 7\n")
    planned = profiles.diff(store, "Notes and markdown")
    assert planned.kept == ("enable_ocr", "chunk_size")
    assert [row.key for row in planned.changes] == ["chunk_overlap"]
    profiles.apply(store, "Notes and markdown")
    assert (cfg.enable_ocr, cfg.chunk_size, cfg.chunk_overlap) == (True, 900, 64)
    stored = _stored()
    assert (stored["chunk_size"], stored["top_k"]) == (900, 7)
    assert stored["profile"]["values"]["chunk_size"] == 384


def test_apply_removes_keys_the_new_profile_does_not_set(store):
    profiles.apply(store, "Notes and markdown")
    profiles.apply(store, "Research papers")
    assert _stored()["profile"] == {
        "name": "Research papers",
        "values": {"layout_detection": True, "table_extraction": True},
    }
    assert (cfg.chunk_size, cfg.chunk_overlap, cfg.enable_ocr) == (512, 100, None)
    assert (cfg.layout_detection, cfg.table_extraction) == (True, True)


def test_apply_rejects_overlap_against_user_chunk_size(store):
    _global_profile("wide", "[values]\nchunk_overlap = 200\n")
    _write_config("chunk_size = 128\n")
    with pytest.raises(ValueError, match=r"chunk_overlap \(200\) must be < chunk_size \(128\)"):
        profiles.apply(store, "wide")
    assert _stored() == {"chunk_size": 128}
    assert (cfg.chunk_size, cfg.chunk_overlap) == (128, 100)


def test_apply_profile_layer_refuses_keys_and_values_a_profile_cannot_hold():
    with pytest.raises(ValueError, match="Profiles cannot set chat_model"):
        apply_profile_layer("x", {"chat_model": "a/b/c.gguf"})
    with pytest.raises(ValueError, match=r"Cannot apply 'chunk_size': its profile value 10"):
        apply_profile_layer("x", {"chunk_size": 10})
    assert _stored() == {}


def test_diff_marks_ocr_keys_new_files_only_and_chunk_size_reindex(store):
    _global_profile("fast", '[profile]\nname = "Fast"\n[values]\ntop_k = 4\nchunk_size = 384\n')
    planned = profiles.diff(store, "notes-and-markdown")
    rows = {row.key: row for row in planned.changes}
    assert rows["enable_ocr"].effect is ProfileEffect.NEW_FILES_ONLY
    assert (rows["enable_ocr"].current, rows["enable_ocr"].new) == (None, False)
    assert rows["enable_ocr"].current_source is SettingSource.AUTO
    assert rows["chunk_size"].effect is ProfileEffect.REINDEX
    assert (rows["chunk_size"].current, rows["chunk_size"].new) == (512, 384)
    assert rows["chunk_size"].current_source is SettingSource.BUILT_IN
    assert rows["chunk_overlap"].effect is ProfileEffect.REINDEX
    assert planned.untouched_count == len(PUBLIC_CONFIG_FIELDS - set(PROFILE_FIELDS)) > 0
    fast = profiles.diff(store, "fast")
    assert {row.key: row.effect for row in fast.changes} == {
        "top_k": ProfileEffect.NOW,
        "chunk_size": ProfileEffect.REINDEX,
    }
    result = profiles.apply(store, "fast")
    assert (result.reindex_required, result.new_files_only) == (True, ())


def test_diff_back_to_builtin_after_a_profile_names_the_profile_as_source(store):
    profiles.apply(store, "Notes and markdown")
    rows = {row.key: row for row in profiles.diff(store, "Default").changes}
    assert (rows["chunk_size"].current, rows["chunk_size"].new) == (384, 512)
    assert rows["chunk_size"].current_source is SettingSource.PROFILE


def test_cfg_matches_fresh_config_after_apply(store, monkeypatch):
    monkeypatch.setenv("LILBEE_CHUNK_OVERLAP", "32")
    _write_config("top_k = 7\n")
    profiles.apply(store, "Notes and markdown")
    fresh = Config()
    keys = sorted((set(WRITABLE_CONFIG_FIELDS) | MODEL_ROLE_FIELDS) - ROOT_DERIVED_FIELDS)
    assert len(keys) > 100
    diverged = {k: (getattr(cfg, k), getattr(fresh, k)) for k in keys}
    assert {k: pair for k, pair in diverged.items() if pair[0] != pair[1]} == {}
    assert (cfg.chunk_size, cfg.chunk_overlap, cfg.top_k) == (384, 32, 7)


def test_changed_builtin_sets_banner_and_default_never_does(store):
    assert profiles.active(store).name == "Default"
    profiles.apply(store, "Scanned archive")
    assert profiles.active(store).status is ProfileStatus.CURRENT
    settings.write_profile_table(cfg.data_root, "Scanned archive", {"layout_detection": True})
    current = profiles.active(store)
    assert (current.name, current.status) == ("Scanned archive", ProfileStatus.CHANGED)
    assert dict(current.values) == {"layout_detection": True}
    settings.write_profile_table(cfg.data_root, "Default", {"chunk_size": 900})
    default = profiles.active(store)
    assert (default.name, default.status) == ("Default", ProfileStatus.CURRENT)


def test_deleted_profile_keeps_recorded_copy(store):
    path = _global_profile("court", '[profile]\nname = "Court"\n[values]\nchunk_size = 768\n')
    profiles.apply(store, "court")
    path.unlink()
    current = profiles.active(store)
    assert (current.name, current.status, current.error) == ("Court", ProfileStatus.MISSING, None)
    assert dict(current.values) == {"chunk_size": 768}
    settings.overlay_persisted_settings(cfg.data_root)
    assert cfg.chunk_size == 768


def test_applied_profile_whose_file_broke_reports_broken_with_the_reason(store):
    path = _global_profile("court", '[profile]\nname = "Court"\n[values]\nchunk_size = 768\n')
    profiles.apply(store, "court")
    path.write_text('[profile]\nname = "Court"\n[values]\nchat_model = "x"\n', encoding="utf-8")
    current = profiles.active(store)
    assert (current.name, current.status) == ("Court", ProfileStatus.BROKEN)
    assert current.error == "Profiles cannot set chat_model"
    assert dict(current.values) == {"chunk_size": 768}


def test_broken_or_missing_profile_refuses_apply(store):
    _global_profile("bad", '[profile]\nname = "Bad"\n[values]\nchat_model = "a/b/c.gguf"\n')
    with pytest.raises(ValueError, match="No profile named 'nope'"):
        profiles.apply(store, "nope")
    with pytest.raises(ValueError, match="'Bad' cannot be used: Profiles cannot set chat_model"):
        profiles.apply(store, "bad")
    with pytest.raises(ValueError, match="'Bad' cannot be used"):
        profiles.diff(store, "bad")
    assert _stored() == {}


def test_profile_table_not_settable_through_settings_update(store):
    profiles.apply(store, "Research papers")
    with pytest.raises(ValueError, match="Unknown or read-only setting: profile"):
        apply_settings_update({"profile": {"name": "Other"}})
    apply_settings_update({"top_k": 5})
    assert _stored()["profile"]["name"] == "Research papers"
    assert _stored()["top_k"] == 5


def test_write_profile_table_replaces_the_table_and_keeps_user_keys():
    _write_config('top_k = 7\n[profile]\nname = "Old"\n[profile.values]\nchunk_size = 900\n')
    settings.write_profile_table(cfg.data_root, "New", {"layout_detection": True})
    assert _stored() == {
        "top_k": 7,
        "profile": {"name": "New", "values": {"layout_detection": True}},
    }


def test_diff_skips_an_invalid_user_value_it_does_not_change(store):
    _write_config("ocr_scan_confidence = 5.0\n")
    planned = profiles.diff(store, "Notes and markdown")
    assert {row.key for row in planned.changes} == {"enable_ocr", "chunk_size", "chunk_overlap"}
    assert planned.kept == ()
