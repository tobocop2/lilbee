"""The suite never resolves the developer's real global config or data dirs."""

import sys
from pathlib import Path

import pytest

from lilbee.cli import apply_overrides
from lilbee.core.config import Config, cfg
from lilbee.core.config import resolve as resolve_mod
from lilbee.core.profile_files import profile_folders
from lilbee.core.system import canonical_models_dir
from lilbee.core.system import default_data_dir as platform_default_data_dir


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
    real_root = platform_default_data_dir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LILBEE_DATA")
    apply_overrides(use_global=True)
    fresh = Config()
    assert len(config_reads) >= 2
    assert [p for p in config_reads if p.is_relative_to(real_root)] == []
    assert not cfg.data_root.is_relative_to(real_root)
    assert not fresh.data_root.is_relative_to(real_root)
    assert not canonical_models_dir().is_relative_to(real_root)


def test_no_loaded_lilbee_module_resolves_the_real_global_root():
    holders = {
        name: vars(module)["default_data_dir"]
        for name, module in list(sys.modules.items())
        if name.startswith("lilbee") and "default_data_dir" in vars(module)
    }
    assert "lilbee.core.profile_files" in holders
    assert "lilbee.core.system" in holders
    assert [name for name, fn in holders.items() if fn is platform_default_data_dir] == []


def test_profile_folders_never_resolve_under_the_real_global_root():
    real_root = platform_default_data_dir()
    folders = [path for _, path in profile_folders(cfg.data_root)]
    assert len(folders) == 4
    assert [path for path in folders if path.is_relative_to(real_root)] == []
