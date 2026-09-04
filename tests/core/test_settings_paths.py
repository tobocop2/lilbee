"""Path-typed settings must round-trip through config.toml."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core import settings


@pytest.mark.parametrize("key", ["documents_dir", "vault_base"])
def test_a_path_value_is_written_as_its_string_and_read_back(tmp_path: Path, key: str) -> None:
    """tomli_w refuses Path objects; save() must hand it the string form."""
    target = tmp_path / "vault" / "lilbee"
    settings.save(tmp_path, {key: target})
    assert settings.load(tmp_path) == {key: str(target)}


def test_update_values_accepts_a_path(tmp_path: Path) -> None:
    settings.save(tmp_path, {"chunk_size": 512})
    settings.update_values(tmp_path, {"documents_dir": tmp_path / "docs"})
    assert settings.load(tmp_path) == {"chunk_size": 512, "documents_dir": str(tmp_path / "docs")}
