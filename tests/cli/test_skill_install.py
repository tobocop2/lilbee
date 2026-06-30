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


def test_race_collision_dest_created_concurrently(tmp_path, monkeypatch):
    """If os.replace raises OSError and dest now exists, treat it as already installed.

    On Windows this can happen when two processes race to install the same skill
    (finding #7: PermissionError from a concurrent writer).
    """
    dest = tmp_path / "skills" / "lilbee-mcp"

    def _boom_then_dest_exists(*_a, **_k):
        # Simulate the concurrent writer: dest appears while we were staging.
        dest.mkdir(parents=True, exist_ok=True)
        raise PermissionError("file in use")

    monkeypatch.setattr(skill_install.os, "replace", _boom_then_dest_exists)
    result = install_bundled_skill(dest)
    # Concurrent install: returns None (already exists), no exception.
    assert result is None
    assert dest.exists()
