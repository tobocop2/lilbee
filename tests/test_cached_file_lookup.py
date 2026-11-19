"""One exact cached-file lookup; the registry adds a basename fallback for GGUF files."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lilbee.catalog.download import RemoteFile, _cached_shard, cached_file_path
from lilbee.catalog.query import build_adhoc_entry
from lilbee.modelhub.registry import ModelRegistry

_WEIGHTS_SIZE = len(b"weights")
_REPO = "org/repo"
_ROOT_FILE = "a.gguf"
_SUBDIR_FILE = "sub/q.gguf"
_PARTIAL_FILE = "partial.gguf"


def _link(snapshot: Path, blobs: Path, name: str, blob: str) -> Path:
    (blobs / blob).write_bytes(b"weights")
    entry = snapshot / name
    entry.parent.mkdir(parents=True, exist_ok=True)
    depth = len(Path(name).parts) + 1
    entry.symlink_to(Path("../" * depth) / "blobs" / blob)
    return entry


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A models dir whose cache holds ``a.gguf`` and ``sub/q.gguf`` of one revision."""
    repo_dir = tmp_path / "models--org--repo"
    snapshot = repo_dir / "snapshots" / "rev1"
    blobs = repo_dir / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("rev1", encoding="utf-8")
    _link(snapshot, blobs, _ROOT_FILE, "blob-a")
    _link(snapshot, blobs, _SUBDIR_FILE, "blob-q")
    (blobs / "blob-p.incomplete").write_bytes(b"partial")
    return tmp_path


def _snapshot(cache: Path) -> Path:
    return cache / "models--org--repo" / "snapshots" / "rev1"


class TestExactLookup:
    @pytest.mark.parametrize("name", [_ROOT_FILE, _SUBDIR_FILE])
    def test_present_file_gives_its_snapshot_path(self, cache: Path, name: str) -> None:
        assert cached_file_path(_REPO, name, cache) == _snapshot(cache) / name

    @pytest.mark.parametrize(("repo", "name"), [(_REPO, "absent.gguf"), ("org/none", _ROOT_FILE)])
    def test_absent_file_gives_none(self, cache: Path, repo: str, name: str) -> None:
        assert cached_file_path(repo, name, cache) is None

    def test_partial_download_gives_none(self, cache: Path) -> None:
        """The hub writes ``<etag>.incomplete`` to blobs/ and links the snapshot when done."""
        partial = cache / "models--org--repo" / "blobs" / "blob-p.incomplete"
        assert partial.is_file()
        assert not (_snapshot(cache) / _PARTIAL_FILE).exists()
        assert cached_file_path(_REPO, _PARTIAL_FILE, cache) is None

    def test_missing_cache_dir_gives_none(self, tmp_path: Path) -> None:
        assert cached_file_path(_REPO, _ROOT_FILE, tmp_path / "none") is None

    def test_a_name_under_another_directory_is_not_a_hit(self, cache: Path) -> None:
        assert cached_file_path(_REPO, "other/a.gguf", cache) is None

    def test_a_file_the_hub_recorded_as_missing_gives_none(self, cache: Path) -> None:
        marker = cache / "models--org--repo" / ".no_exist" / "rev1" / "gone.gguf"
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b"")
        assert cached_file_path(_REPO, "gone.gguf", cache) is None


class TestCachedShard:
    """The shard lookup reads the cache through the shared exact lookup."""

    @staticmethod
    def _shard(cache: Path, name: str, size: int) -> Path | None:
        entry = build_adhoc_entry(_REPO)
        remote = RemoteFile(size, None)
        with patch("lilbee.catalog.download.fetch_remote_file", return_value=remote):
            return _cached_shard(entry, cache, name, None)

    def test_cache_hit_of_the_reported_size(self, cache: Path) -> None:
        assert self._shard(cache, _ROOT_FILE, _WEIGHTS_SIZE) == _snapshot(cache) / _ROOT_FILE

    def test_cache_hit_of_another_size_is_refused(self, cache: Path) -> None:
        assert self._shard(cache, _ROOT_FILE, _WEIGHTS_SIZE + 1) is None

    def test_absent_shard_gives_none(self, cache: Path) -> None:
        assert self._shard(cache, "absent.gguf", _WEIGHTS_SIZE) is None

    def test_flat_file_of_an_older_install_is_found(self, cache: Path) -> None:
        (cache / "flat.gguf").write_bytes(b"weights")
        assert self._shard(cache, "flat.gguf", _WEIGHTS_SIZE) == cache / "flat.gguf"


class TestRegistryLookup:
    def test_exact_hit_resolves_to_the_blob(self, cache: Path) -> None:
        found = ModelRegistry(cache)._find_cached_gguf(_REPO, _SUBDIR_FILE)
        assert found == (cache / "models--org--repo" / "blobs" / "blob-q").resolve()

    def test_snapshot_lookup_keeps_the_symlink(self, cache: Path) -> None:
        found = ModelRegistry(cache)._snapshot_gguf_path(_REPO, _ROOT_FILE)
        assert found == _snapshot(cache) / _ROOT_FILE
        assert found.is_symlink()

    def test_basename_fallback_finds_a_file_under_another_directory(self, cache: Path) -> None:
        registry = ModelRegistry(cache)
        assert cached_file_path(_REPO, "x/a.gguf", cache) is None
        assert registry._snapshot_gguf_path(_REPO, "x/a.gguf") == _snapshot(cache) / _ROOT_FILE

    def test_partial_or_absent_file_gives_none(self, cache: Path) -> None:
        registry = ModelRegistry(cache)
        assert registry._find_cached_gguf(_REPO, _PARTIAL_FILE) is None
        assert registry._find_cached_gguf(_REPO, "absent.gguf") is None

    def test_missing_cache_dir_gives_none(self, tmp_path: Path) -> None:
        assert ModelRegistry(tmp_path / "none")._find_cached_gguf(_REPO, _ROOT_FILE) is None

    def test_symlink_leaving_the_cache_is_refused_though_the_exact_lookup_sees_it(
        self, cache: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp("outside") / "evil.gguf"
        outside.write_bytes(b"evil")
        (_snapshot(cache) / "evil.gguf").symlink_to(outside)
        assert cached_file_path(_REPO, "evil.gguf", cache) == _snapshot(cache) / "evil.gguf"
        assert ModelRegistry(cache)._snapshot_gguf_path(_REPO, "evil.gguf") is None

    def test_exact_step_follows_the_main_ref_not_the_newest_revision(self, cache: Path) -> None:
        blobs = cache / "models--org--repo" / "blobs"
        newer = cache / "models--org--repo" / "snapshots" / "rev2"
        newer.mkdir()
        _link(newer, blobs, _ROOT_FILE, "blob-a2")
        os.utime(_snapshot(cache) / _ROOT_FILE, (1_000, 1_000), follow_symlinks=False)
        os.utime(newer / _ROOT_FILE, (2_000, 2_000), follow_symlinks=False)
        found = ModelRegistry(cache)._snapshot_gguf_path(_REPO, _ROOT_FILE)
        assert found == _snapshot(cache) / _ROOT_FILE

    def test_exact_step_is_the_shared_lookup(self, cache: Path) -> None:
        shared = _snapshot(cache) / _ROOT_FILE
        with patch("lilbee.modelhub.registry.cached_file_path", return_value=shared) as lookup:
            found = ModelRegistry(cache)._snapshot_gguf_path(_REPO, "unrelated.gguf")
        lookup.assert_called_once_with(_REPO, "unrelated.gguf", cache)
        assert found == shared
