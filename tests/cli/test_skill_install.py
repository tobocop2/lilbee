"""Tests for the shared bundled-skill installer."""

from __future__ import annotations

import csv
import errno
import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest

from lilbee.cli.launchers import skill_install
from lilbee.cli.launchers.skill_install import install_bundled_skill

_OLD = "old text\n"
_NEW = "new text\n"
_SKILL_FIXTURES = Path(__file__).parents[1] / "fixtures" / "skill"
# The SKILL.md that the first release with the skill, a later one and the last release installed.
_RELEASED_SKILLS = [
    _SKILL_FIXTURES / f"SKILL-v{release}.md"
    for release in ("0.6.90b420.dev710", "0.6.90b691", "0.6.90b692")
]
# Every skill file in every release's PyPI wheel: release, path, sha256, installed, installer.
_RELEASED_DIGESTS = _SKILL_FIXTURES / "released-skill-digests.tsv"


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """Point the installer at a temp bundle whose old and new SKILL.md are both known."""
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    known = {
        "SKILL.md": frozenset(hashlib.sha256(t.encode()).hexdigest() for t in (_OLD, _NEW)),
        "reference.md": frozenset({hashlib.sha256(b"reference\n").hexdigest()}),
    }
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", packaged)
    monkeypatch.setattr(skill_install, "_KNOWN_SKILL_FILES", known)

    def _ship(text: str) -> Path:
        # Bytes, so Windows does not turn each newline into CRLF and change the digest.
        (packaged / "SKILL.md").write_bytes(text.encode())
        (packaged / "reference.md").write_bytes(b"reference\n")
        return packaged

    return _ship


@pytest.fixture
def dest(tmp_path) -> Path:
    return tmp_path / "skills" / "lilbee-mcp"


def _files(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == skill_install.__name__]


def test_installs_into_empty_dest(dest):
    assert install_bundled_skill(dest) == dest
    assert _files(dest) == _files(skill_install.bundled_skill_dir())


def test_a_fresh_install_of_the_real_skill_is_not_read_as_edited(dest, caplog):
    install_bundled_skill(dest)
    caplog.set_level("WARNING")
    assert not any(skill_install._is_edited(dest / name) for name in _files(dest))
    assert install_bundled_skill(dest) is None
    assert _warnings(caplog) == []


def test_install_copies_the_packaged_skill_files_and_nothing_else(bundle, dest):
    bundle(_OLD)
    assert install_bundled_skill(dest) == dest
    assert _files(dest) == {"SKILL.md": _OLD.encode(), "reference.md": b"reference\n"}
    assert sorted(p.name for p in dest.parent.iterdir()) == ["lilbee-mcp"]


def test_a_current_install_is_left_alone_without_a_warning(bundle, dest, caplog):
    bundle(_NEW)
    install_bundled_skill(dest)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    assert _warnings(caplog) == []


def test_refreshes_an_unedited_copy_when_the_bundled_skill_changes(bundle, dest, caplog):
    bundle(_OLD)
    install_bundled_skill(dest)
    bundle(_NEW)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) == dest
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == _NEW
    assert _warnings(caplog) == []
    assert sorted(p.name for p in dest.parent.iterdir()) == ["lilbee-mcp"]


def test_refreshes_only_the_files_that_changed(bundle, dest, monkeypatch):
    bundle(_OLD)
    install_bundled_skill(dest)
    bundle(_NEW)
    written: list[str] = []
    real_write = skill_install.write_text_atomically

    def _record(path: Path, text: str) -> None:
        written.append(path.name)
        real_write(path, text)

    monkeypatch.setattr(skill_install, "write_text_atomically", _record)
    install_bundled_skill(dest)
    assert written == ["SKILL.md"]


@pytest.mark.parametrize("fixture", _RELEASED_SKILLS, ids=lambda path: path.stem)
def test_a_copy_from_a_release_is_refreshed_without_a_warning(dest, caplog, fixture):
    released = fixture.read_bytes()
    assert released != (skill_install.bundled_skill_dir() / "SKILL.md").read_bytes()
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_bytes(released)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) == dest
    assert _files(dest) == _files(skill_install.bundled_skill_dir())
    assert _warnings(caplog) == []


def test_keeps_an_edited_copy_and_names_the_folder_and_the_fix(bundle, dest, caplog):
    bundle(_OLD)
    install_bundled_skill(dest)
    (dest / "SKILL.md").write_text("my edit\n", encoding="utf-8")
    bundle(_NEW)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == "my edit\n"
    assert _warnings(caplog) == [
        f"The lilbee skill at {dest} has local changes, so it is not updated. "
        "Delete the folder to get the bundled version."
    ]


def test_an_edit_to_any_owned_file_blocks_the_whole_refresh(bundle, dest):
    bundle(_OLD)
    install_bundled_skill(dest)
    (dest / "reference.md").write_text("my notes\n", encoding="utf-8")
    bundle(_NEW)
    assert install_bundled_skill(dest) is None
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == _OLD


@pytest.mark.parametrize("added", ["mine.md", "notes/mine.md"])
def test_files_the_user_added_are_kept_through_a_refresh(bundle, dest, added):
    bundle(_OLD)
    install_bundled_skill(dest)
    (dest / added).parent.mkdir(parents=True, exist_ok=True)
    (dest / added).write_text("mine\n", encoding="utf-8")
    bundle(_NEW)
    assert install_bundled_skill(dest) == dest
    assert (dest / added).read_text(encoding="utf-8") == "mine\n"
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == _NEW


def test_a_symlinked_skill_folder_is_refreshed_through_the_link(bundle, dest, tmp_path):
    bundle(_OLD)
    target = tmp_path / "dotfiles" / "lilbee-mcp"
    install_bundled_skill(target)
    dest.parent.mkdir(parents=True)
    dest.symlink_to(target, target_is_directory=True)
    bundle(_NEW)
    assert install_bundled_skill(dest) == dest
    assert dest.is_symlink()
    assert dest.resolve() == target.resolve()
    assert (target / "SKILL.md").read_text(encoding="utf-8") == _NEW
    assert sorted(p.name for p in dest.parent.iterdir()) == ["lilbee-mcp"]


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        ("file", "is not a folder"),
        ("dangling", "is not a folder"),
        ("parent_file", "is not a folder"),
    ],
)
def test_a_destination_that_is_not_a_folder_is_skipped_with_a_warning(
    dest, tmp_path, caplog, make, reason
):
    if make == "parent_file":
        dest.parent.parent.mkdir(parents=True, exist_ok=True)
        dest.parent.write_text("not a folder\n", encoding="utf-8")
    else:
        dest.parent.mkdir(parents=True)
    if make == "file":
        dest.write_text("a user file\n", encoding="utf-8")
    if make == "dangling":
        dest.symlink_to(tmp_path / "gone")
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    [message] = _warnings(caplog)
    assert message.startswith(f"The lilbee skill at {dest} {reason}, so it is not updated: ")
    assert not (tmp_path / "gone").exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="POSIX mode bits, not root"
)
@pytest.mark.parametrize("locked", ["skill_folder", "skills_folder"])
def test_a_folder_without_write_permission_is_skipped_and_says_so(bundle, dest, caplog, locked):
    bundle(_OLD)
    if locked == "skill_folder":
        install_bundled_skill(dest)
        bundle(_NEW)
        folder = dest
    else:
        dest.parent.mkdir(parents=True)
        folder = dest.parent
    before = _files(dest.parent)
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    caplog.set_level("WARNING")
    try:
        assert install_bundled_skill(dest) is None
    finally:
        folder.chmod(0o755)
    [message] = _warnings(caplog)
    assert message.startswith(
        f"The lilbee skill at {dest} is not writable (permission denied), so it is not updated: "
    )
    assert "in use" not in message
    assert _files(dest.parent) == before
    assert sorted(p.name for p in dest.parent.iterdir()) == (["lilbee-mcp"] if before else [])


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (errno.EBUSY, "is in use"),
        (errno.EPERM, "is not writable (permission denied)"),
        (errno.EROFS, "is not writable (permission denied)"),
        (errno.ENOSPC, "could not be written"),
    ],
)
def test_a_failed_write_names_its_cause_and_leaves_the_old_copy(
    bundle, dest, monkeypatch, caplog, code, reason
):
    bundle(_OLD)
    install_bundled_skill(dest)
    bundle(_NEW)

    def _fail(path: Path, _text: str) -> None:
        raise OSError(code, os.strerror(code), str(path))

    monkeypatch.setattr(skill_install, "write_text_atomically", _fail)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    assert _warnings(caplog) == [
        f"The lilbee skill at {dest} {reason}, so it is not updated: "
        f"{OSError(code, os.strerror(code), str(dest / 'SKILL.md'))}"
    ]
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == _OLD


def test_an_error_without_an_errno_gets_the_general_reason(bundle, dest, monkeypatch, caplog):
    bundle(_OLD)

    def _fail(_path: Path, _text: str) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(skill_install, "write_text_atomically", _fail)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    assert _warnings(caplog) == [
        f"The lilbee skill at {dest} could not be written, so it is not updated: disk went away"
    ]


@pytest.mark.parametrize(
    ("platform", "winerror", "reason"),
    [
        ("win32", 32, "is in use"),
        ("win32", 33, "is in use"),
        ("win32", 5, "is not writable (permission denied)"),
        ("win32", None, "is not writable (permission denied)"),
        ("linux", 32, "is not writable (permission denied)"),
    ],
)
def test_only_a_windows_sharing_or_lock_violation_on_the_replace_says_in_use(
    bundle, dest, monkeypatch, caplog, platform, winerror, reason
):
    bundle(_OLD)
    install_bundled_skill(dest)
    bundle(_NEW)
    before = _files(dest)
    refused = PermissionError(errno.EACCES, "Access is denied")
    refused.winerror = winerror

    def _refuse(_src, _dst):
        raise refused

    monkeypatch.setattr(os, "replace", _refuse)
    monkeypatch.setattr(skill_install.sys, "platform", platform)
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    raised = PermissionError(errno.EACCES, "Access is denied", str(dest / "SKILL.md"))
    assert _warnings(caplog) == [
        f"The lilbee skill at {dest} {reason}, so it is not updated: {raised}"
    ]
    assert _files(dest) == before


@pytest.mark.parametrize(
    ("winerror", "reason"),
    [(32, "is in use"), (5, "is not writable (permission denied)")],
)
def test_only_a_windows_sharing_violation_on_the_read_says_in_use(
    bundle, dest, monkeypatch, caplog, winerror, reason
):
    bundle(_OLD)
    install_bundled_skill(dest)
    bundle(_NEW)
    held = dest / "SKILL.md"
    refused = PermissionError(errno.EACCES, "Access is denied", str(held))
    refused.winerror = winerror
    real_read = Path.read_bytes

    def _read(path: Path) -> bytes:
        if path == held:
            raise refused
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", _read)
    monkeypatch.setattr(skill_install.sys, "platform", "win32")
    caplog.set_level("WARNING")
    assert install_bundled_skill(dest) is None
    assert _warnings(caplog) == [
        f"The lilbee skill at {dest} {reason}, so it is not updated: {refused}"
    ]


def test_a_missing_bundled_skill_still_raises(dest, tmp_path, monkeypatch):
    monkeypatch.setattr(skill_install, "_PACKAGED_SKILL", tmp_path / "no-installed")
    monkeypatch.setattr(skill_install, "_CHECKOUT_SKILL", tmp_path / "no-checkout")
    with pytest.raises(FileNotFoundError, match=r"no-installed.*no-checkout"):
        install_bundled_skill(dest)
    assert not dest.exists()


def test_every_bundled_skill_file_is_in_the_known_versions():
    bundled = _files(skill_install.bundled_skill_dir())
    assert bundled
    for name, data in bundled.items():
        assert hashlib.sha256(data).hexdigest() in skill_install._KNOWN_SKILL_FILES[name], (
            f"add the sha256 of the bundled {name} to the known skill versions"
        )


def test_the_known_versions_keep_every_file_a_release_installed():
    with _RELEASED_DIGESTS.open(encoding="utf-8", newline="") as table:
        rows = list(csv.DictReader(table, delimiter="\t"))
    installed: dict[str, set[str]] = {}
    for row in rows:
        if row["installed"] == "yes":
            installed.setdefault(Path(row["path"]).name, set()).add(row["digest"])
    assert len({row["release"] for row in rows}) == 59
    assert {name: len(digests) for name, digests in installed.items()} == {
        "SKILL.md": 15,
        "reference-settings.md": 1,
        "reference-tuning.md": 1,
    }
    for name, digests in installed.items():
        missing = digests - skill_install._KNOWN_SKILL_FILES.get(name, frozenset())
        assert not missing, f"keep the sha256 of every released {name}: {sorted(missing)}"


def test_each_released_fixture_has_the_digest_of_its_release():
    with _RELEASED_DIGESTS.open(encoding="utf-8", newline="") as table:
        shipped = {
            (row["release"], row["digest"])
            for row in csv.DictReader(table, delimiter="\t")
            if row["path"].endswith("/SKILL.md")
        }
    for fixture in _RELEASED_SKILLS:
        release = fixture.stem.removeprefix("SKILL-v")
        assert (release, hashlib.sha256(fixture.read_bytes()).hexdigest()) in shipped


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
