"""The suite never resolves the developer's real global config or data dirs."""

import os
import tempfile
from itertools import pairwise
from pathlib import Path

import pytest

from lilbee.cli import apply_overrides
from lilbee.core import profile_files
from lilbee.core.config import Config, cfg
from lilbee.core.config import resolve as resolve_mod
from lilbee.core.profile_files import profile_folders
from lilbee.core.system import canonical_models_dir, default_data_dir
from tests.conftest import (
    REAL_GLOBAL_ROOT,
    REAL_PACKAGE_PROFILES_DIR,
    folder_state,
    profile_folders_under_real_root,
)

# Read while pytest collects this module, before any per-test fixture runs.
_ROOT_AT_COLLECTION = default_data_dir()
_HOME_AT_COLLECTION = Path(os.environ["HOME"])


@pytest.fixture
def config_reads(monkeypatch) -> list[Path]:
    """Every config.toml path the resolver reads during the test."""
    reads: list[Path] = []
    real_read = resolve_mod._read_toml

    def _spy(path: Path) -> dict:
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(resolve_mod, "_read_toml", _spy)
    return reads


def test_global_root_and_fresh_config_never_read_the_real_global_config(
    tmp_path, monkeypatch, config_reads
):
    real_root = REAL_GLOBAL_ROOT
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LILBEE_DATA")
    apply_overrides(use_global=True)
    fresh = Config()
    assert len(config_reads) >= 2
    assert [p for p in config_reads if p.is_relative_to(real_root)] == []
    assert not cfg.data_root.is_relative_to(real_root)
    assert not fresh.data_root.is_relative_to(real_root)
    assert not canonical_models_dir().is_relative_to(real_root)


def test_profile_folders_never_resolve_under_the_real_global_root():
    assert default_data_dir() != REAL_GLOBAL_ROOT
    assert len(profile_folders(cfg.data_root)) == 4
    assert profile_folders_under_real_root(cfg.data_root) == []


def test_a_module_read_at_collection_never_sees_the_real_global_root():
    assert not _ROOT_AT_COLLECTION.is_relative_to(REAL_GLOBAL_ROOT)


def test_the_real_global_root_is_not_a_session_scratch_dir():
    assert not REAL_GLOBAL_ROOT.is_relative_to(tempfile.gettempdir())


def test_the_session_scratch_dir_outlives_collection():
    assert _HOME_AT_COLLECTION.parent.is_dir()


def test_the_package_profiles_folder_is_a_session_copy_of_the_real_one():
    copy = profile_files.PACKAGE_PROFILES_DIR
    assert not copy.is_relative_to(REAL_PACKAGE_PROFILES_DIR)
    real = folder_state(REAL_PACKAGE_PROFILES_DIR)
    assert "builtin/default.toml" in real
    assert folder_state(copy) == real


def test_the_folder_guard_sees_a_file_changed_added_or_removed(tmp_path):
    (tmp_path / "builtin").mkdir()
    kept = tmp_path / "builtin" / "a.toml"
    kept.write_text("x", encoding="utf-8")
    states = [folder_state(tmp_path)]
    kept.write_text("y", encoding="utf-8")
    states.append(folder_state(tmp_path))
    (tmp_path / "builtin" / "b.toml").write_text("y", encoding="utf-8")
    states.append(folder_state(tmp_path))
    kept.unlink()
    states.append(folder_state(tmp_path))
    assert all(before != after for before, after in pairwise(states))
