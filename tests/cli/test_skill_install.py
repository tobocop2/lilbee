"""Tests for the shared bundled-skill installer."""

from __future__ import annotations

import pytest

from lilbee.cli.launchers import skill_install
from lilbee.cli.launchers.skill_install import install_bundled_skill


def test_installs_into_empty_dest(tmp_path):
    dest = tmp_path / "skills" / "lilbee-mcp"
    result = install_bundled_skill(dest)
    assert result == dest
    assert dest.is_dir()
    assert (dest / "SKILL.md").exists()


def test_skips_when_present(tmp_path):
    dest = tmp_path / "skills" / "lilbee-mcp"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("user edit")
    assert install_bundled_skill(dest) is None
    assert (dest / "SKILL.md").read_text() == "user edit"


def test_atomic_on_failure(tmp_path, monkeypatch):
    """A failed copy must not leave a half-written skill dir or staging litter."""
    dest = tmp_path / "skills" / "lilbee-mcp"

    def _boom(*_a, **_k):
        raise OSError("rename failed")

    monkeypatch.setattr(skill_install.os, "replace", _boom)
    with pytest.raises(OSError):
        install_bundled_skill(dest)
    assert not dest.exists()
    assert not list((tmp_path / "skills").glob(".lilbee-mcp-*"))
