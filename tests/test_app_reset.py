"""Unit tests for app.reset._clear_dir."""

from __future__ import annotations

from pathlib import Path

from lilbee.app.reset import _clear_dir


def test_clears_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("y")
    skipped: list[str] = []
    deleted = _clear_dir(tmp_path, skipped)
    assert deleted == 2
    assert skipped == []
    assert list(tmp_path.iterdir()) == []


def test_symlink_pointing_outside_is_removed_not_followed(tmp_path: Path) -> None:
    """A stray symlink whose target is outside must be unlinked (the link only),
    not abort the reset and not delete the target."""
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "keep.md"
    target.write_text("precious")
    base = tmp_path / "base"
    base.mkdir()
    link = base / "link.md"
    link.symlink_to(target)

    skipped: list[str] = []
    deleted = _clear_dir(base, skipped)

    assert skipped == []
    assert not link.exists()  # the link is gone
    assert target.read_text() == "precious"  # the target is untouched
    assert deleted == 1
