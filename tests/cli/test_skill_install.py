"""Tests for the shared bundled-skill installer."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.cli.launchers import skill_install
from lilbee.cli.launchers.skill_install import install_bundled_skill


def test_installs_into_empty_dest(tmp_path):
    dest = tmp_path / "skills" / "lilbee-mcp"
    result = install_bundled_skill(dest)
    assert result == dest
    assert dest.is_dir()
    assert (dest / "SKILL.md").exists()


def _bundle(tmp_path, monkeypatch, text):
    packaged = tmp_path / "packaged"
    packaged.mkdir(exist_ok=True)
    (packaged / "SKILL.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", packaged)


def test_keeps_a_copy_with_no_install_record_and_says_so(tmp_path, caplog):
    dest = tmp_path / "skills" / "lilbee-mcp"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("user edit")
    with caplog.at_level("WARNING", logger=skill_install.__name__):
        assert install_bundled_skill(dest) is None
    assert (dest / "SKILL.md").read_text() == "user edit"
    assert "has local changes" in caplog.text


def test_refreshes_an_unedited_copy_when_the_bundled_skill_changes(tmp_path, monkeypatch):
    dest = tmp_path / "skills" / "lilbee-mcp"
    _bundle(tmp_path, monkeypatch, "old text")
    assert install_bundled_skill(dest) == dest
    _bundle(tmp_path, monkeypatch, "new text")
    assert install_bundled_skill(dest) == dest
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "new text"
    assert install_bundled_skill(dest) is None  # now current, nothing to do
    assert not list((tmp_path / "skills").glob(".lilbee-mcp-*"))


def test_keeps_a_user_edit_when_the_bundled_skill_changes(tmp_path, monkeypatch, caplog):
    dest = tmp_path / "skills" / "lilbee-mcp"
    _bundle(tmp_path, monkeypatch, "old text")
    install_bundled_skill(dest)
    (dest / "SKILL.md").write_text("my edit", encoding="utf-8")
    _bundle(tmp_path, monkeypatch, "new text")
    with caplog.at_level("WARNING", logger=skill_install.__name__):
        assert install_bundled_skill(dest) is None
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "my edit"
    assert "has local changes" in caplog.text


def test_a_failed_refresh_restores_the_old_copy(tmp_path, monkeypatch):
    dest = tmp_path / "skills" / "lilbee-mcp"
    _bundle(tmp_path, monkeypatch, "old text")
    install_bundled_skill(dest)
    _bundle(tmp_path, monkeypatch, "new text")
    real_replace = skill_install.os.replace

    def _fail_install_step(src, dst):
        if Path(dst) == dest and "-old-" not in Path(src).name:
            raise OSError("rename failed")
        real_replace(src, dst)

    monkeypatch.setattr(skill_install.os, "replace", _fail_install_step)
    with pytest.raises(OSError, match="rename failed"):
        install_bundled_skill(dest)
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "old text"
    assert not list((tmp_path / "skills").glob(".lilbee-mcp-*"))


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


def test_bundled_skill_dir_prefers_installed_copy(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / "SKILL.md").write_text("packaged", encoding="utf-8")
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", packaged)
    assert skill_install.bundled_skill_dir() == packaged


def test_bundled_skill_dir_uses_checkout_copy_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", tmp_path / "absent")
    result = skill_install.bundled_skill_dir()
    assert result == skill_install._CHECKOUT_SKILL
    assert (result / "SKILL.md").is_file()


def test_bundled_skill_dir_names_both_locations_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", tmp_path / "no-installed")
    monkeypatch.setattr(skill_install, "_CHECKOUT_SKILL", tmp_path / "no-checkout")
    with pytest.raises(FileNotFoundError, match=r"no-installed.*no-checkout"):
        skill_install.bundled_skill_dir()


def test_install_copies_the_packaged_skill_files(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / "SKILL.md").write_text("packaged text", encoding="utf-8")
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", packaged)
    dest = tmp_path / "skills" / "lilbee-mcp"
    assert install_bundled_skill(dest) == dest
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "packaged text"


def test_bundled_pi_extension_prefers_installed_copy(tmp_path, monkeypatch):
    extension = tmp_path / "agent_plugin" / "extensions" / "lilbee.js"
    extension.parent.mkdir(parents=True)
    extension.write_text("export default () => {}", encoding="utf-8")
    monkeypatch.setattr(skill_install, "_PACKAGED_AGENT_PLUGIN", tmp_path / "agent_plugin")
    assert skill_install.bundled_pi_extension() == extension


def test_bundled_pi_extension_uses_checkout_copy_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_install, "_PACKAGED_AGENT_PLUGIN", tmp_path / "absent")
    result = skill_install.bundled_pi_extension()
    assert result == skill_install._CHECKOUT_AGENT_PLUGIN / "extensions" / "lilbee.js"
    assert "registerMcpServer" in result.read_text(encoding="utf-8")


def test_bundled_pi_extension_names_both_locations_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(skill_install, "_PACKAGED_AGENT_PLUGIN", tmp_path / "no-installed")
    monkeypatch.setattr(skill_install, "_CHECKOUT_AGENT_PLUGIN", tmp_path / "no-checkout")
    with pytest.raises(
        FileNotFoundError, match=r"extensions/lilbee\.js.*no-installed.*no-checkout"
    ):
        skill_install.bundled_pi_extension()
