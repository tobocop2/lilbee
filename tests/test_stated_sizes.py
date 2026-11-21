"""The stated-size helper reports a size for an empty file and leaves other files alone."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._stated_sizes import state_size

GIB = 1024**3


def test_stating_a_size_without_the_fixture_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="stated_sizes fixture"):
        state_size(tmp_path / "m.gguf", GIB)


@pytest.mark.usefixtures("stated_sizes")
class TestAStatedSize:
    def test_the_file_reports_the_stated_size_and_holds_no_bytes(self, tmp_path: Path) -> None:
        model = tmp_path / "deep" / "m.gguf"
        state_size(model, 20 * GIB)
        assert model.stat().st_size == 20 * GIB
        assert os.stat(model).st_size == 0
        assert model.is_file()

    def test_a_symlink_reports_the_size_of_its_target(self, tmp_path: Path) -> None:
        blob = tmp_path / "blob"
        state_size(blob, GIB)
        link = tmp_path / "link"
        link.symlink_to(blob)
        assert link.stat().st_size == GIB

    def test_another_file_reports_its_own_size(self, tmp_path: Path) -> None:
        state_size(tmp_path / "m.gguf", GIB)
        other = tmp_path / "other.txt"
        other.write_bytes(b"abc")
        assert other.stat().st_size == 3

    def test_the_link_itself_reports_its_own_size(self, tmp_path: Path) -> None:
        blob = tmp_path / "blob"
        state_size(blob, GIB)
        link = tmp_path / "link"
        link.symlink_to(blob)
        assert link.stat(follow_symlinks=False).st_size != GIB
