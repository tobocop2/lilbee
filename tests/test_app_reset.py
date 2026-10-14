"""Unit tests for app.reset."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from filelock import FileLock

from lilbee.app.reset import _clear_dir, perform_reset
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.ingest import skip_marker
from lilbee.data.ingest.discovery import discover_files
from lilbee.data.ingest.skip_marker import (
    SKIP_MARKER_FILENAME,
    load_skip_markers,
    load_skip_reasons,
    write_skip_markers,
    write_skip_reasons,
)
from lilbee.runtime.lock import ResetRefusedError
from tests._records_wait import files_under


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


@pytest.fixture
def isolated_cfg(tmp_path: Path) -> Iterator[Path]:
    """Point cfg at a temp data root and restore every field afterwards."""
    snapshot = cfg.model_copy()
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.linked_roots = {}
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def test_reset_unregisters_linked_roots(isolated_cfg: Path, tmp_path: Path) -> None:
    """Reset removes the source registry, so a following sync re-indexes nothing."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("linked", encoding="utf-8")
    settings.set_value(cfg.data_root, "linked_roots", {"corpus": str(corpus)})
    cfg.linked_roots = {"corpus": str(corpus)}

    perform_reset()

    assert cfg.linked_roots == {}
    assert "linked_roots" not in settings.load(cfg.data_root)
    assert discover_files() == {}
    assert (corpus / "doc.md").read_text(encoding="utf-8") == "linked"  # the corpus is untouched


def test_reset_clears_skip_markers(isolated_cfg: Path) -> None:
    """Reset drops the skip sidecars, so a re-added file is extracted again."""
    write_skip_markers(cfg.data_root, {"stubborn.pdf": "abc123"})
    write_skip_reasons(cfg.data_root, {"stubborn.pdf": "no text extracted"})

    perform_reset()

    assert load_skip_markers(cfg.data_root) == {}
    assert load_skip_reasons(cfg.data_root) == {}


def test_reset_keeps_other_settings(isolated_cfg: Path) -> None:
    """Reset deletes data, not configuration: other config.toml keys survive."""
    settings.set_value(cfg.data_root, "chat_model", "qwen3:8b")
    settings.set_value(cfg.data_root, "linked_roots", {"corpus": "/somewhere"})

    perform_reset()

    assert settings.get(cfg.data_root, "chat_model") == "qwen3:8b"


def test_a_reset_while_another_process_holds_the_records_lock_deletes_nothing(
    isolated_cfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset takes the records lock before it deletes: a held lock refuses it, nothing gone."""
    (cfg.documents_dir / "doc.md").write_text("kept", encoding="utf-8")
    (cfg.data_dir / "chunks.bin").write_text("kept", encoding="utf-8")
    settings.set_value(cfg.data_root, "linked_roots", {"corpus": "/somewhere"})
    cfg.linked_roots = {"corpus": "/somewhere"}
    write_skip_markers(cfg.data_root, {"stubborn.pdf": "abc123"})
    before = files_under(isolated_cfg)
    monkeypatch.setattr(skip_marker, "_RECORDS_LOCK_TIMEOUT_S", 0.05)
    lock_file = str(cfg.data_root / SKIP_MARKER_FILENAME) + ".lock"
    other_process = FileLock(lock_file)
    other_process.acquire()
    try:
        with pytest.raises(ResetRefusedError) as refused:
            perform_reset()
    finally:
        other_process.release()

    assert str(refused.value) == (
        f"Could not lock {lock_file}, so the list of held-out files was not changed. "
        "If no other lilbee process is running, delete that file and try again."
    )
    assert files_under(isolated_cfg) == before
    assert set(before) >= {"documents/doc.md", "data/chunks.bin", SKIP_MARKER_FILENAME}
    assert cfg.linked_roots == {"corpus": "/somewhere"}
