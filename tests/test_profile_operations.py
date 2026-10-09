"""Profile file operations: new, save as, update, discard, duplicate, rename, delete, export,
import and validate."""

import os
import tomllib
from pathlib import Path

import pytest

from lilbee.app import profiles
from lilbee.app.profiles import ProfileStatus
from lilbee.app.services import get_services, set_services
from lilbee.app.settings import apply_profile_layer, list_settings
from lilbee.core import profile_files, settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import SettingSource
from lilbee.core.config.resolve import PROFILE_FIELDS, read_layers, resolve
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder, ProfileStore
from lilbee.core.system import default_data_dir
from tests._private_mode import file_mode, posix_only
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def _services():
    set_services(make_mock_services())
    yield
    set_services(None)


@pytest.fixture
def store() -> ProfileStore:
    return get_services().profile_store


def _global_dir() -> Path:
    return default_data_dir() / PROFILES_DIRNAME


def _project_dir() -> Path:
    return cfg.data_root / PROFILES_DIRNAME


def _stored() -> dict:
    path = cfg.data_root / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_config(text: str) -> None:
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    (cfg.data_root / "config.toml").write_text(text, encoding="utf-8")
    settings.overlay_persisted_settings(cfg.data_root)


def _prepend_config(text: str) -> None:
    """Add top-level keys above the ``[profile]`` table an apply wrote."""
    _write_config(text + (cfg.data_root / "config.toml").read_text(encoding="utf-8"))


def _write(folder: Path, stem: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.toml")) if folder.is_dir() else []


def _source(key: str) -> SettingSource:
    return resolve(key, read_layers(cfg.data_root)).source


def test_new_writes_every_profile_setting_commented_out_with_default_and_help(store):
    location = profiles.new(store, "Court filings")
    assert location.path == _global_dir() / "court-filings.toml"
    assert (location.name, location.folder) == ("Court filings", ProfileFolder.GLOBAL)
    text = location.path.read_text(encoding="utf-8")
    assert _read(location.path) == {"profile": {"name": "Court filings", "format": 1}, "values": {}}
    help_texts = {info.key: info.help_text for info in list_settings()}
    assert len(PROFILE_FIELDS) > 40
    for key in PROFILE_FIELDS:
        assert f"# {help_texts[key].splitlines()[0]}" in text
    assert "# chunk_size = 512\n" in text
    assert '# ocr_strategy = "auto"\n' in text
    assert "# enable_ocr has no default: leaving it out lets lilbee decide\n" in text
    entry = store.scan().find("court filings")
    assert entry is not None and entry.file is not None


def test_new_from_a_profile_sets_its_values_and_comments_the_rest(store):
    location = profiles.new(
        store, "Short notes", ProfileFolder.PROJECT, from_name="notes-and-markdown"
    )
    assert location.path == _project_dir() / "short-notes.toml"
    assert _read(location.path)["values"] == {
        "enable_ocr": False,
        "chunk_size": 384,
        "chunk_overlap": 64,
    }
    text = location.path.read_text(encoding="utf-8")
    assert "\nchunk_size = 384\n" in text
    assert "# top_k = 12\n" in text


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("court_filings", "A profile named court_filings already exists"),
        ("Scanned Archive", "Reserved name: Scanned Archive is a built-in profile"),
        ("a/b", "Bad name 'a/b'"),
    ],
)
def test_new_refuses_a_taken_reserved_or_bad_name_and_writes_nothing(store, name, error):
    _write(_global_dir(), "court-filings", "[values]\n")
    with pytest.raises(ValueError, match=error):
        profiles.new(store, name)
    assert _files(_global_dir()) == ["court-filings.toml"]


def test_a_name_held_by_a_file_with_another_stem_is_taken(store):
    _write(_global_dir(), "old", '[profile]\nname = "Court filings"\n[values]\n')
    with pytest.raises(ValueError, match=r"already exists: .*old\.toml"):
        profiles.new(store, "court filings")


def test_a_file_whose_name_differs_only_in_case_is_taken(store):
    _write(_global_dir(), "Mine", '[profile]\nname = "Other"\n[values]\nchunk_size = 900\n')
    with pytest.raises(ValueError, match=r"already exists: .*Mine\.toml"):
        profiles.new(store, "mine")
    assert _read(_global_dir() / "Mine.toml")["profile"]["name"] == "Other"


def test_writes_go_only_to_the_project_or_global_folder(store, monkeypatch, tmp_path):
    monkeypatch.setattr(profile_files, "PACKAGE_PROFILES_DIR", tmp_path / "package")
    with pytest.raises(ValueError, match="saved to this project or to all projects"):
        profiles.new(store, "X", ProfileFolder.BUILTIN)
    monkeypatch.setattr(cfg, "data_root", default_data_dir())
    with pytest.raises(ValueError, match="there is no project folder"):
        profiles.save_as("X", ProfileFolder.PROJECT)
    assert _files(_global_dir()) == []
    assert not (tmp_path / "package").exists()


def test_save_as_moves_your_profile_settings_into_the_new_profile(store, monkeypatch):
    monkeypatch.setenv("LILBEE_TABLE_EXTRACTION", "true")
    profiles.apply(store, "Notes and markdown")
    _prepend_config("chunk_size = 900\ntop_k = 7\nauto_sync = false\n")
    result = profiles.save_as("Mine", ProfileFolder.PROJECT)
    assert result.absorbed == ("chunk_size", "top_k")
    assert result.location.path == _project_dir() / "mine.toml"
    saved = {"enable_ocr": False, "chunk_size": 900, "chunk_overlap": 64, "top_k": 7}
    assert _read(result.location.path)["values"] == saved
    assert _stored() == {"auto_sync": False, "profile": {"name": "Mine", "values": saved}}
    assert (cfg.chunk_size, cfg.chunk_overlap, cfg.top_k, cfg.auto_sync) == (900, 64, 7, False)
    assert cfg.table_extraction is True
    assert (_source("chunk_size"), _source("chunk_overlap")) == (
        SettingSource.PROFILE,
        SettingSource.PROFILE,
    )
    assert _source("table_extraction") is SettingSource.ENV
    assert profiles.active(store).status is ProfileStatus.CURRENT


def test_save_as_warns_when_it_leaves_ocr_off_with_a_vision_model(store):
    profiles.apply(store, "Notes and markdown")
    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    result = profiles.save_as("Mine", ProfileFolder.PROJECT)
    assert len(result.warnings) == 1
    assert "enable_ocr" in result.warnings[0]
    assert cfg.enable_ocr is False


def test_save_as_on_default_saves_only_your_settings(store):
    _write_config("layout_detection = true\n")
    result = profiles.save_as("Layout")
    assert _read(result.location.path)["values"] == {"layout_detection": True}
    assert _stored() == {"profile": {"name": "Layout", "values": {"layout_detection": True}}}
    assert cfg.layout_detection is True


def test_refused_save_as_writes_nothing(store):
    _write(_global_dir(), "mine", "[values]\n")
    _write_config("chunk_size = 900\n")
    with pytest.raises(ValueError, match="A profile named Mine already exists"):
        profiles.save_as("Mine")
    assert _stored() == {"chunk_size": 900}
    assert _files(_global_dir()) == ["mine.toml"]


def test_update_writes_your_settings_into_the_active_file_and_keeps_its_metadata(store):
    path = _write(
        _global_dir(),
        "mine",
        '[profile]\nname = "Mine"\ndescription = "Mine."\n'
        'authors = [{ name = "Jo" }]\n[values]\nchunk_overlap = 50\nlayout_detection = true\n',
    )
    profiles.apply(store, "mine")
    _prepend_config("chunk_size = 700\n")
    result = profiles.update(store)
    assert result.absorbed == ("chunk_size",)
    assert result.location.path == path
    written = _read(path)
    assert written["profile"] == {
        "name": "Mine",
        "description": "Mine.",
        "authors": [{"name": "Jo"}],
        "format": 1,
    }
    values = {"chunk_overlap": 50, "layout_detection": True, "chunk_size": 700}
    assert written["values"] == values
    assert _stored() == {"profile": {"name": "Mine", "values": values}}
    assert _source("chunk_size") is SettingSource.PROFILE
    assert profiles.active(store).status is ProfileStatus.CURRENT


def test_update_warns_when_it_leaves_ocr_off_with_a_vision_model(store):
    _write(_global_dir(), "mine", "[values]\nenable_ocr = false\n")
    profiles.apply(store, "mine")
    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    result = profiles.update(store)
    assert len(result.warnings) == 1
    assert "enable_ocr" in result.warnings[0]
    assert cfg.enable_ocr is False


def test_update_renames_a_hand_named_file_to_the_slug(store):
    old = _write(_global_dir(), "Mine File", '[profile]\nname = "Mine"\n[values]\n')
    profiles.apply(store, "mine")
    result = profiles.update(store)
    assert result.location.path == _global_dir() / "mine.toml"
    assert not old.exists()
    assert _files(_global_dir()) == ["mine.toml"]


def test_update_refuses_a_builtin_and_offers_save_as(store):
    _write_config("chunk_size = 700\n")
    with pytest.raises(ValueError, match=r"Default ships with lilbee.*save your settings as a new"):
        profiles.update(store)
    profiles.apply(store, "Scanned archive")
    with pytest.raises(ValueError, match="Scanned archive ships with lilbee"):
        profiles.update(store)
    assert _stored()["chunk_size"] == 700


def test_update_refuses_a_missing_or_broken_file(store):
    path = _write(_global_dir(), "mine", "[values]\nchunk_size = 900\n")
    profiles.apply(store, "mine")
    path.write_text("[values]\nchat_model = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be used: Profiles cannot set chat_model"):
        profiles.update(store)
    path.unlink()
    with pytest.raises(ValueError, match="No profile named 'mine'"):
        profiles.update(store)


def test_discard_drops_your_profile_settings_only(store, monkeypatch):
    monkeypatch.setenv("LILBEE_TOP_K", "3")
    profiles.apply(store, "Notes and markdown")
    _prepend_config("chunk_size = 900\nauto_sync = false\n")
    result = profiles.discard()
    assert (result.dropped, result.reindex_required) == (("chunk_size",), True)
    assert "chunk_size" not in _stored()
    assert _stored()["auto_sync"] is False
    assert (cfg.chunk_size, cfg.top_k) == (384, 3)
    assert profiles.discard() == profiles.DiscardResult((), reindex_required=False)


def test_discard_of_settings_that_need_no_rebuild_reports_no_reindex(store):
    profiles.apply(store, "Notes and markdown")
    _prepend_config("top_k = 7\nmax_chunks_per_file = 9\n")
    result = profiles.discard()
    assert (result.dropped, result.reindex_required) == (("max_chunks_per_file", "top_k"), False)


def test_discard_warns_when_it_leaves_ocr_off_with_a_vision_model(store):
    _write(_global_dir(), "mine", "[values]\nenable_ocr = false\n")
    profiles.apply(store, "mine")
    _prepend_config("enable_ocr = true\n")
    cfg.vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    result = profiles.discard()
    assert result.dropped == ("enable_ocr",)
    assert len(result.warnings) == 1
    assert "enable_ocr" in result.warnings[0]
    assert cfg.enable_ocr is False


def test_discard_of_a_setting_with_no_conflict_reports_no_warnings(store):
    profiles.apply(store, "Notes and markdown")
    _prepend_config("top_k = 7\n")
    assert profiles.discard().warnings == ()


def test_duplicate_copies_a_builtin_with_its_metadata_under_a_new_name(store):
    location = profiles.duplicate(store, "Scanned archive", "My scans", ProfileFolder.PROJECT)
    assert location.path == _project_dir() / "my-scans.toml"
    written = _read(location.path)
    builtin = profiles.show(store, "Scanned archive").file
    assert builtin is not None
    assert written["profile"]["name"] == "My scans"
    assert written["profile"]["description"] == builtin.description
    assert written["values"] == dict(builtin.values)


def test_rename_moves_the_file_and_the_active_project_follows(store):
    old = _write(_global_dir(), "mine", "[values]\nchunk_size = 900\n")
    profiles.apply(store, "mine")
    location = profiles.rename(store, "mine", "Court filings")
    assert location.path == _global_dir() / "court-filings.toml"
    assert not old.exists()
    assert _read(location.path)["profile"]["name"] == "Court filings"
    assert _stored()["profile"] == {"name": "Court filings", "values": {"chunk_size": 900}}
    assert profiles.active(store).status is ProfileStatus.CURRENT


def test_rename_of_another_profile_leaves_the_project_record(store):
    _write(_global_dir(), "mine", "[values]\nchunk_size = 900\n")
    _write(_global_dir(), "other", "[values]\n")
    profiles.apply(store, "mine")
    profiles.rename(store, "other", "Renamed")
    assert _stored()["profile"]["name"] == "mine"
    assert _files(_global_dir()) == ["mine.toml", "renamed.toml"]


def test_rename_that_changes_only_case_rewrites_in_place(store):
    _write(_global_dir(), "mine", '[profile]\nname = "mine"\n[values]\n')
    location = profiles.rename(store, "mine", "Mine")
    assert location.path == _global_dir() / "mine.toml"
    assert _read(location.path)["profile"]["name"] == "Mine"


def test_rename_refuses_a_builtin_or_a_taken_name_and_changes_nothing(store):
    with pytest.raises(ValueError, match=r"Research papers ships with lilbee.*duplicate it"):
        profiles.rename(store, "research papers", "Papers")
    _write(_global_dir(), "mine", "[values]\n")
    _write(_global_dir(), "other", "[values]\n")
    with pytest.raises(ValueError, match="A profile named Other already exists"):
        profiles.rename(store, "mine", "Other")
    assert _files(_global_dir()) == ["mine.toml", "other.toml"]


def test_delete_removes_the_file_and_the_project_keeps_its_copy(store):
    _write(_global_dir(), "mine", "[values]\nchunk_size = 900\n")
    profiles.apply(store, "mine")
    location = profiles.delete(store, "mine")
    assert location.path == _global_dir() / "mine.toml"
    assert not location.path.exists()
    assert _stored()["profile"] == {"name": "mine", "values": {"chunk_size": 900}}
    assert cfg.chunk_size == 900
    assert profiles.active(store).status is ProfileStatus.MISSING


def test_delete_removes_a_broken_file_and_refuses_a_builtin(store):
    _write(_global_dir(), "broken", "[values]\nchat_model = 'x'\n")
    profiles.delete(store, "broken")
    assert _files(_global_dir()) == []
    with pytest.raises(ValueError, match="Default ships with lilbee"):
        profiles.delete(store, "default")


def test_export_writes_a_clean_validated_file(store, tmp_path):
    _write(
        _global_dir(),
        "mine",
        '# a comment\n[profile]\nname = "Mine"\ntested_on = "notes"\n[values]\nchunk_size = 900\n',
    )
    out = tmp_path / "out"
    out.mkdir()
    location = profiles.export(store, "mine", out)
    assert (location.name, location.folder, location.path) == (
        "Mine",
        ProfileFolder.GLOBAL,
        out / "mine.toml",
    )
    assert "# a comment" not in location.path.read_text(encoding="utf-8")
    assert _read(location.path) == {
        "profile": {"name": "Mine", "tested_on": "notes", "format": 1},
        "values": {"chunk_size": 900},
    }
    with pytest.raises(ValueError, match="already exists"):
        profiles.export(store, "mine", location.path)
    target = profiles.export(store, "Scanned archive", location.path, overwrite=True)
    assert target.name == "Scanned archive"
    assert _read(target.path)["profile"]["name"] == "Scanned archive"


def test_export_refuses_a_broken_profile(store, tmp_path):
    _write(_global_dir(), "broken", "[values]\nchat_model = 'x'\n")
    with pytest.raises(ValueError, match="cannot be used: Profiles cannot set chat_model"):
        profiles.export(store, "broken", tmp_path / "x.toml")
    assert not (tmp_path / "x.toml").exists()


def test_import_copies_the_file_verbatim_and_refuses_a_clash_unless_overwrite(store, tmp_path):
    text = '# kept\n[profile]\nname = "Court filings"\n[values]\nchunk_size = 900\n'
    source = _write(tmp_path / "in", "anything", text)
    location = profiles.import_profile(store, source, ProfileFolder.PROJECT)
    assert location.path == _project_dir() / "court-filings.toml"
    assert location.path.read_text(encoding="utf-8") == text
    changed = _write(tmp_path / "in", "again", text.replace("900", "800"))
    with pytest.raises(ValueError, match="A profile named Court filings already exists"):
        profiles.import_profile(store, changed, ProfileFolder.PROJECT)
    assert _read(location.path)["values"] == {"chunk_size": 900}
    profiles.import_profile(store, changed, ProfileFolder.PROJECT, overwrite=True)
    assert _read(location.path)["values"] == {"chunk_size": 800}


def test_import_with_overwrite_replaces_a_file_with_another_stem(store, tmp_path):
    old = _write(_global_dir(), "hand named", '[profile]\nname = "Court filings"\n[values]\n')
    source = _write(tmp_path / "in", "x", '[profile]\nname = "court filings"\n[values]\n')
    location = profiles.import_profile(store, source, overwrite=True)
    assert location.path == _global_dir() / "court-filings.toml"
    assert not old.exists()


def test_import_refuses_an_invalid_or_unreadable_file_and_writes_nothing(store, tmp_path):
    bad = _write(tmp_path / "in", "bad", "[values]\nchunk_size = 10\n")
    with pytest.raises(ValueError, match="Bad value for chunk_size"):
        profiles.import_profile(store, bad)
    raw = b"# caf\xe9\n[values]\n"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    latin = tmp_path / "in" / "latin.toml"
    latin.write_bytes(raw)
    with pytest.raises(ValueError, match="Not UTF-8 text"):
        profiles.import_profile(store, latin)
    with pytest.raises(ValueError, match="Cannot read the file"):
        profiles.import_profile(store, tmp_path / "in" / "missing.toml")
    assert _files(_global_dir()) == []


def test_validate_reports_every_problem_in_a_file(tmp_path):
    path = _write(
        tmp_path,
        "x",
        '[profile]\nname = "Default"\ndescription = 3\n[values]\n'
        "chunk_size = 100\nchunk_overlap = 200\ntop_k = 0\nchat_model = 'x'\nnope = 1\n"
        "[extras]\n",
    )
    report = profiles.validate(path)
    assert report.name == "Default"
    assert not report.valid
    assert report.problems == (
        "Unknown table: extras",
        "description must be text",
        "Bad value for top_k: Input should be greater than or equal to 1",
        "Profiles cannot set chat_model",
        "Unknown setting: nope",
        "chunk_overlap (200) must be < chunk_size (100)",
        "Reserved name: Default is a built-in profile",
    )


def test_validate_passes_a_good_file_and_reports_an_unreadable_one(tmp_path):
    good = profiles.validate(_write(tmp_path, "good", "[values]\nchunk_size = 900\n"))
    assert (good.valid, good.problems, good.name) == (True, (), "good")
    broken = profiles.validate(_write(tmp_path, "broken", "[values\n"))
    assert len(broken.problems) == 1
    assert broken.problems[0].startswith("Not valid TOML")


def test_validate_applies_the_evidence_rule_for_builtin_profiles(tmp_path):
    path = _write(tmp_path, "x", "[values]\ntop_k = 20\nmmr_lambda = 0.9\n")
    assert profiles.validate(path).valid
    report = profiles.validate(path, ProfileFolder.BUILTIN)
    assert report.problems == (
        "Sets retrieval setting top_k without evidence",
        "Sets retrieval setting mmr_lambda without evidence",
    )


def test_apply_profile_layer_refuses_to_take_over_a_key_the_profile_does_not_hold():
    _write_config("chunk_size = 900\n")
    with pytest.raises(ValueError, match="does not hold chunk_size"):
        apply_profile_layer("x", {"top_k": 7}, absorb=("chunk_size",))
    assert _stored() == {"chunk_size": 900}


def test_write_profile_table_drops_the_named_keys_in_the_same_write():
    _write_config("chunk_size = 900\ntop_k = 7\n")
    settings.write_profile_table(cfg.data_root, "New", {"chunk_size": 900}, drop=("chunk_size",))
    assert _stored() == {"top_k": 7, "profile": {"name": "New", "values": {"chunk_size": 900}}}


def test_apply_profile_layer_validates_a_taken_over_key_at_the_profile_value():
    _write_config("chunk_overlap = 50\n")
    with pytest.raises(ValueError, match=r"chunk_overlap \(600\) must be < chunk_size \(512\)"):
        apply_profile_layer("x", {"chunk_overlap": 600}, absorb=("chunk_overlap",))
    assert _stored() == {"chunk_overlap": 50}


def _refuse(*_args, **_kwargs):
    raise OSError("disk full")


def test_update_of_a_file_named_in_another_case_keeps_the_file(store):
    _write(_global_dir(), "Mine", '[profile]\nname = "Mine"\n[values]\nchunk_size = 900\n')
    profiles.apply(store, "Mine")
    _prepend_config("top_k = 7\n")
    result = profiles.update(store)
    assert _read(result.location.path)["values"] == {"chunk_size": 900, "top_k": 7}
    assert [name.casefold() for name in _files(_global_dir())] == ["mine.toml"]
    assert profiles.active(store).status is ProfileStatus.CURRENT


def test_rename_to_another_case_of_the_file_name_keeps_the_file(store):
    _write(_global_dir(), "Mine", '[profile]\nname = "Mine"\n[values]\nchunk_size = 900\n')
    location = profiles.rename(store, "Mine", "mine")
    assert _read(location.path) == {
        "profile": {"name": "mine", "format": 1},
        "values": {"chunk_size": 900},
    }
    assert [name.casefold() for name in _files(_global_dir())] == ["mine.toml"]


def test_import_with_overwrite_over_a_file_named_in_another_case_keeps_the_file(store, tmp_path):
    _write(_global_dir(), "Mine", '[profile]\nname = "Mine"\n[values]\n')
    text = '[profile]\nname = "Mine"\n[values]\nchunk_size = 777\n'
    source = _write(tmp_path / "in", "incoming", text)
    location = profiles.import_profile(store, source, overwrite=True)
    assert location.path.read_text(encoding="utf-8") == text
    assert [name.casefold() for name in _files(_global_dir())] == ["mine.toml"]


def test_a_write_that_replaces_another_spelling_of_its_own_path_keeps_the_file(tmp_path):
    (tmp_path / "sub").mkdir()
    path = _write(tmp_path, "mine", "[values]\n")
    text = "[values]\nchunk_size = 900\n"
    profile = profile_files.parse_text(text, "mine", ProfileFolder.GLOBAL)
    replacing = tmp_path / "sub" / ".." / "mine.toml"
    assert replacing != path
    written = profile_files.PlannedWrite(
        path, ProfileFolder.GLOBAL, profile, text, replacing
    ).write()
    assert written.read_text(encoding="utf-8") == text


def test_save_as_whose_file_write_fails_leaves_config_toml_unchanged(store, monkeypatch):
    profiles.apply(store, "Notes and markdown")
    _prepend_config("chunk_size = 900\n")
    before = _stored()
    monkeypatch.setattr(profile_files.PlannedWrite, "write", _refuse)
    with pytest.raises(OSError, match="disk full"):
        profiles.save_as("Mine")
    assert _stored() == before
    assert _files(_global_dir()) == []
    assert cfg.chunk_size == 900


def test_update_whose_file_write_fails_leaves_config_toml_unchanged(store, monkeypatch):
    path = _write(_global_dir(), "mine", "[values]\nchunk_overlap = 50\n")
    profiles.apply(store, "mine")
    _prepend_config("chunk_size = 700\n")
    before = _stored()
    monkeypatch.setattr(profile_files.PlannedWrite, "write", _refuse)
    with pytest.raises(OSError, match="disk full"):
        profiles.update(store)
    assert _stored() == before
    assert path.read_text(encoding="utf-8") == "[values]\nchunk_overlap = 50\n"


def test_save_as_whose_config_write_fails_keeps_the_new_file_and_says_so(store, monkeypatch):
    profiles.apply(store, "Notes and markdown")
    _prepend_config("chunk_size = 900\n")
    before = _stored()
    monkeypatch.setattr(settings, "save", _refuse)
    with pytest.raises(
        ValueError,
        match=r"Saved the profile to .*mine\.toml, but switching this project to it failed: "
        "disk full",
    ):
        profiles.save_as("Mine")
    assert _stored() == before
    assert _read(_global_dir() / "mine.toml")["values"]["chunk_size"] == 900


def test_update_whose_config_write_fails_keeps_the_new_file_and_says_so(store, monkeypatch):
    path = _write(_global_dir(), "mine", "[values]\nchunk_overlap = 50\n")
    profiles.apply(store, "mine")
    _prepend_config("chunk_size = 700\n")
    before = _stored()
    monkeypatch.setattr(settings, "save", _refuse)
    with pytest.raises(ValueError, match="switching this project to it failed: disk full"):
        profiles.update(store)
    assert _stored() == before
    assert _read(path)["values"] == {"chunk_overlap": 50, "chunk_size": 700}


@posix_only
def test_profile_files_and_exports_are_written_with_the_umask(store, tmp_path):
    previous = os.umask(0o022)
    try:
        written = [
            profiles.new(store, "Fresh").path,
            profiles.save_as("Saved").location.path,
            profiles.export(store, "Scanned archive", tmp_path).path,
        ]
    finally:
        os.umask(previous)
    assert [file_mode(path) for path in written] == [0o644, 0o644, 0o644]


def test_import_refuses_a_file_over_the_size_cap_and_writes_nothing(store, tmp_path):
    source = tmp_path / "big.toml"
    source.write_text("[values]\n" + "#" * profile_files.MAX_PROFILE_BYTES + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"too large for a profile file"):
        profiles.import_profile(store, source)
    assert _files(_global_dir()) == []


def test_validate_and_scan_report_a_file_over_the_size_cap(store, tmp_path):
    big = "[values]\n" + "#" * profile_files.MAX_PROFILE_BYTES + "\n"
    path = _write(_global_dir(), "big", big)
    reason = "The file is over 256 KB, too large for a profile file"
    assert profiles.validate(path).problems == (reason,)
    entry = store.scan().find("big")
    assert entry is not None and entry.error == reason


@posix_only
def test_import_refuses_an_endless_source(store):
    with pytest.raises(ValueError, match=r"too large for a profile file"):
        profiles.import_profile(store, Path("/dev/zero"))
    assert _files(_global_dir()) == []


def test_a_nameless_import_shows_the_name_the_scan_shows(store, tmp_path):
    source = _write(tmp_path / "in", "My Profile", "[values]\nchunk_size = 700\n")
    location = profiles.import_profile(store, source)
    entry = store.scan().find(location.name)
    assert entry is not None
    assert (entry.name, entry.path) == (location.name, location.path)


def test_update_refuses_a_file_changed_since_it_was_applied(store):
    path = _write(_global_dir(), "mine", "[values]\nchunk_size = 900\n")
    profiles.apply(store, "mine")
    edited = "[values]\nchunk_size = 900\ntop_k = 9\n"
    path.write_text(edited, encoding="utf-8")
    _prepend_config("chunk_overlap = 50\n")
    with pytest.raises(ValueError, match="mine changed on disk since it was applied"):
        profiles.update(store)
    assert path.read_text(encoding="utf-8") == edited
    assert _stored()["chunk_overlap"] == 50


def test_update_with_a_name_needs_the_active_profile(store):
    path = _write(_global_dir(), "mine", '[profile]\nname = "Mine"\n[values]\n')
    profiles.apply(store, "mine")
    _prepend_config("chunk_size = 700\n")
    with pytest.raises(ValueError, match=r"^Other is not this project's profile; Mine is$"):
        profiles.update(store, "Other")
    assert _read(path)["values"] == {}
    assert profiles.update(store, "MINE").absorbed == ("chunk_size",)
    assert _read(path)["values"] == {"chunk_size": 700}


def test_import_text_names_the_file_by_its_slug_whatever_the_upload_name(store, tmp_path):
    location = profiles.import_text(store, "[values]\n", "../../Up Load.toml")
    assert (location.name, location.path) == ("up-load", _global_dir() / "up-load.toml")
    assert not (tmp_path / "Up Load.toml").exists()


def test_a_missing_name_raises_the_not_found_error(store):
    with pytest.raises(profiles.ProfileNotFoundError, match=r"^No profile named 'nope'$"):
        profiles.show(store, "nope")


def test_credit_line_names_each_author_and_their_github():
    authors = (
        profile_files.ProfileAuthor("Jane Doe", "janedoe"),
        profile_files.ProfileAuthor("Sam Roe", None),
    )
    assert profiles.credit_line(authors) == "by Jane Doe (@janedoe), Sam Roe"
    assert profiles.credit_line(()) is None


def test_tested_on_line_fills_the_template_or_is_absent():
    assert profiles.tested_on_line("4,000 filings", "Tested on: {text}") == (
        "Tested on: 4,000 filings"
    )
    assert profiles.tested_on_line(None, "Tested on: {text}") is None
    assert profiles.tested_on_line("", "Tested on: {text}") is None


@pytest.mark.parametrize(
    ("key", "effect"),
    [
        ("chunk_size", profiles.ProfileEffect.REINDEX),
        ("max_chunks_per_file", profiles.ProfileEffect.NEW_FILES_ONLY),
        ("top_k", profiles.ProfileEffect.NOW),
    ],
)
def test_effect_of_names_when_a_profile_setting_takes_effect(key, effect):
    assert profiles.effect_of(key) is effect


def test_your_changes_lists_your_profile_settings_against_the_profile(store, monkeypatch):
    monkeypatch.setenv("LILBEE_CHUNK_OVERLAP", "40")
    profiles.apply(store, "Notes and markdown")
    _prepend_config("chunk_size = 900\ntop_k = 7\nchunk_overlap = 50\nauto_sync = false\n")
    assert profiles.your_changes() == (
        profiles.ChangeRow(
            "chunk_size", 900, 384, SettingSource.PROFILE, profiles.ProfileEffect.REINDEX
        ),
        profiles.ChangeRow("top_k", 7, 12, SettingSource.BUILT_IN, profiles.ProfileEffect.NOW),
    )
    profiles.discard()
    assert profiles.your_changes() == ()


def test_your_changes_under_default_compare_with_the_built_in_value():
    _write_config("chunk_size = 900\n")
    assert profiles.your_changes() == (
        profiles.ChangeRow(
            "chunk_size", 900, 512, SettingSource.BUILT_IN, profiles.ProfileEffect.REINDEX
        ),
    )


def test_active_carries_your_changes_under_default_and_an_applied_profile(store):
    _write_config("chunk_size = 900\n")
    default = profiles.active(store)
    assert default.name == "Default"
    assert [(row.key, row.profile_source) for row in default.changes] == [
        ("chunk_size", SettingSource.BUILT_IN)
    ]
    profiles.apply(store, "Notes and markdown")
    applied = profiles.active(store)
    assert applied.changes == profiles.your_changes()
    assert [(row.key, row.profile_value) for row in applied.changes] == [("chunk_size", 384)]
    profiles.discard()
    assert profiles.active(store).changes == ()
