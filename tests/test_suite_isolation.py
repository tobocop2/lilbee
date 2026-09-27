"""The suite never resolves the developer's real global config or data dirs."""

from pathlib import Path

import pytest

from lilbee.cli import apply_overrides
from lilbee.core.config import Config, cfg
from lilbee.core.config import resolve as resolve_mod
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
