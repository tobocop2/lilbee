"""Tests for registry.py: manifest-keyed by (hf_repo, gguf_filename)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lilbee.catalog.refs import format_native_gguf_ref
from lilbee.modelhub.registry import (
    ModelManifest,
    ModelRegistry,
    _sha256_file,
    _validate_gguf_filename,
    _validate_hf_repo,
    parse_hf_ref,
    repo_to_dir,
)

_REPO = "Qwen/Qwen3-0.6B-GGUF"
_FILENAME = "Qwen3-0.6B-Q4_K_M.gguf"
_REF = f"{_REPO}/{_FILENAME}"


def _make_manifest(
    *,
    hf_repo: str = _REPO,
    gguf_filename: str = _FILENAME,
    size_bytes: int = 1000,
    task: str = "chat",
    blob: str = "",
    downloaded_at: str = "2026-04-25T00:00:00+00:00",
) -> ModelManifest:
    return ModelManifest(
        hf_repo=hf_repo,
        gguf_filename=gguf_filename,
        size_bytes=size_bytes,
        task=task,
        downloaded_at=downloaded_at,
        blob=blob,
    )


def _write_source(tmp_path: Path, content: bytes = b"GGUF\x00\x00") -> Path:
    src = tmp_path / "source.gguf"
    src.write_bytes(content)
    return src


class TestParseHfRef:
    def test_canonical_shape(self) -> None:
        repo, filename = parse_hf_ref(_REF)
        assert repo == _REPO
        assert filename == _FILENAME

    def test_filename_with_dots(self) -> None:
        repo, filename = parse_hf_ref(
            "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        )
        assert repo == "nomic-ai/nomic-embed-text-v1.5-GGUF"
        assert filename == "nomic-embed-text-v1.5.Q4_K_M.gguf"

    def test_bare_name_tag_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a HuggingFace ref"):
            parse_hf_ref("qwen3:0.6b")

    def test_missing_gguf_suffix_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a HuggingFace ref"):
            parse_hf_ref("Qwen/Qwen3-0.6B-GGUF")

    def test_missing_repo_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a HuggingFace ref"):
            parse_hf_ref("standalone.gguf")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_hf_ref("../etc/passwd.gguf")


class TestValidators:
    def test_valid_repo(self) -> None:
        assert _validate_hf_repo(_REPO) == _REPO

    def test_invalid_repo_no_slash(self) -> None:
        with pytest.raises(ValueError):
            _validate_hf_repo("standalone")

    def test_invalid_repo_double_slash(self) -> None:
        with pytest.raises(ValueError):
            _validate_hf_repo("a/b/c")

    def test_repo_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            _validate_hf_repo("../etc/passwd")

    def test_valid_filename(self) -> None:
        assert _validate_gguf_filename(_FILENAME) == _FILENAME

    def test_filename_must_end_in_gguf(self) -> None:
        with pytest.raises(ValueError):
            _validate_gguf_filename("model.bin")

    def test_filename_no_path(self) -> None:
        with pytest.raises(ValueError):
            _validate_gguf_filename("dir/file.gguf")

    def test_filename_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            _validate_gguf_filename("..gguf")


class TestRepoToDir:
    def test_simple(self) -> None:
        assert repo_to_dir(_REPO) == "Qwen--Qwen3-0.6B-GGUF"

    def test_namespace_with_dashes(self) -> None:
        assert repo_to_dir("nomic-ai/x") == "nomic-ai--x"


class TestFormatNativeGgufRef:
    def test_canonical_shape(self) -> None:
        assert format_native_gguf_ref(_REPO, _FILENAME) == _REF

    def test_round_trips_through_parse_hf_ref(self) -> None:
        ref = format_native_gguf_ref(_REPO, _FILENAME)
        assert parse_hf_ref(ref) == (_REPO, _FILENAME)


class TestModelManifest:
    def test_ref_property(self) -> None:
        m = _make_manifest()
        assert m.ref == _REF


class TestSha256File:
    def test_computes_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        assert _sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        assert _sha256_file(p) == hashlib.sha256(b"").hexdigest()


class TestModelRegistryInstall:
    def test_install_writes_manifest_and_blob(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        manifest = _make_manifest(size_bytes=src.stat().st_size)

        blob_path = registry.install(_REPO, _FILENAME, src, manifest)

        assert blob_path.exists()
        manifest_file = tmp_path / "manifests" / repo_to_dir(_REPO) / f"{_FILENAME}.json"
        assert manifest_file.exists()
        data = json.loads(manifest_file.read_text())
        assert data["hf_repo"] == _REPO
        assert data["gguf_filename"] == _FILENAME
        assert data["blob"] == _sha256_file(src)

    def test_install_idempotent_blob(self, tmp_path: Path) -> None:
        """Re-installing the same source content reuses the existing blob."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        manifest = _make_manifest()

        first = registry.install(_REPO, _FILENAME, src, manifest)
        second = registry.install(_REPO, _FILENAME, src, manifest)
        assert first == second

    def test_install_records_blob_digest(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        content = b"GGUF" + b"\x00" * 256
        src = tmp_path / "source.gguf"
        src.write_bytes(content)
        manifest = _make_manifest(size_bytes=len(content))

        registry.install(_REPO, _FILENAME, src, manifest)
        listed = registry.list_installed()
        assert listed[0].blob == hashlib.sha256(content).hexdigest()


class TestModelRegistryResolve:
    def test_resolve_not_installed(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        with pytest.raises(KeyError, match="not installed"):
            registry.resolve(_REF)

    def test_resolve_unparseable_ref_raises_value_error(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        with pytest.raises(ValueError, match="not a HuggingFace ref"):
            registry.resolve("qwen3:0.6b")

    def test_resolve_returns_blob_path(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        path = registry.resolve(_REF)
        assert path.exists()
        assert path.parent.name == "blobs"

    def test_resolve_missing_cache_dir(self, tmp_path: Path) -> None:
        """Manifest exists but the cache folder was deleted out from under us."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        cache_dir = tmp_path / f"models--{repo_to_dir(_REPO)}"
        # Move the entire cache dir away
        moved = tmp_path / "moved_cache"
        cache_dir.rename(moved)
        with pytest.raises(KeyError, match="Cache folder missing"):
            registry.resolve(_REF)

    def test_resolve_missing_blob(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        blob_dir = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        for blob in blob_dir.iterdir():
            blob.unlink()
        with pytest.raises(KeyError, match="Blob file missing"):
            registry.resolve(_REF)

    def test_resolve_no_blob_hash_in_manifest(self, tmp_path: Path) -> None:
        """A manifest written before install computes the digest fails with a
        clear 'install incomplete' error rather than building cache_path / 'blobs' / ''."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        # Write a manifest with blob=None directly to mimic the "download wrote
        # the manifest but install never set the digest" race.
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        manifest_file = tmp_path / "manifests" / repo_to_dir(_REPO) / f"{_FILENAME}.json"
        data = json.loads(manifest_file.read_text())
        data["blob"] = None
        manifest_file.write_text(json.dumps(data))
        with pytest.raises(KeyError, match="install incomplete"):
            registry.resolve(_REF)


class TestModelRegistryIsInstalled:
    def test_is_installed_true(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        assert registry.is_installed(_REF)

    def test_is_installed_false(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert not registry.is_installed(_REF)

    def test_is_installed_unparseable_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        # resolve() raises on unparseable refs; is_installed swallows that.
        assert not registry.is_installed("qwen3:0.6b")


class TestModelRegistryRemove:
    def test_remove_existing(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        removed = registry.remove(_REF)
        assert removed is True
        assert registry.list_installed() == []

    def test_remove_deletes_cached_blob(self, tmp_path: Path) -> None:
        """Manifest and its backing GGUF blob both go away on remove,
        otherwise the user sees no disk space freed when they delete a
        model from the catalog."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        cache_path = tmp_path / f"models--{repo_to_dir(_REPO)}"
        assert any((cache_path / "blobs").iterdir())
        registry.remove(_REF)
        # Blob file is gone and the per-repo cache directory is pruned.
        assert not cache_path.exists()

    def test_remove_wipes_huggingface_cache_cruft(self, tmp_path: Path) -> None:
        """HF's refs/main and snapshots/<rev>/<filename> live alongside
        our blob under models--<repo>/. They have to go away too,
        otherwise the user keeps seeing the per-repo folder after a
        delete and rightly assumes nothing was freed."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        cache_path = tmp_path / f"models--{repo_to_dir(_REPO)}"
        # Simulate the structure huggingface_hub leaves behind.
        (cache_path / "refs").mkdir(parents=True, exist_ok=True)
        (cache_path / "refs" / "main").write_text("deadbeef")
        snapshot_dir = cache_path / "snapshots" / "deadbeef"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / _FILENAME).write_text("symlink stand-in")
        registry.remove(_REF)
        assert not cache_path.exists()

    def test_remove_keeps_blob_when_another_manifest_references_it(self, tmp_path: Path) -> None:
        """Two manifests pointing at the same blob digest is rare but
        must not free the blob until both manifests are gone."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        # Second manifest sharing the same blob digest. install() rehashes
        # the source so re-using the same source produces the same digest.
        second_filename = "Qwen3-0.6B-Q8_0-alias.gguf"
        registry.install(_REPO, second_filename, src, _make_manifest(gguf_filename=second_filename))
        blob_dir = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        registry.remove(_REF)
        assert any(blob_dir.iterdir())  # blob survives because the alias references it

    def test_remove_multi_quant_keeps_other_quants_blob(self, tmp_path: Path) -> None:
        """Removing one quant must not delete a sibling quant's blob.
        Each quant has its own digest so the GC keys on digest equality."""
        registry = ModelRegistry(tmp_path)
        q4_src = tmp_path / "q4.gguf"
        q4_src.write_bytes(b"GGUF" + b"\x01" * 50)
        q8_src = tmp_path / "q8.gguf"
        q8_src.write_bytes(b"GGUF" + b"\x02" * 100)
        registry.install(_REPO, "Q4.gguf", q4_src, _make_manifest(gguf_filename="Q4.gguf"))
        registry.install(_REPO, "Q8.gguf", q8_src, _make_manifest(gguf_filename="Q8.gguf"))
        registry.remove(f"{_REPO}/Q4.gguf")
        assert registry.is_installed(f"{_REPO}/Q8.gguf")

    def test_remove_missing(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        removed = registry.remove(_REF)
        assert removed is False

    def test_remove_invalid_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        removed = registry.remove("not-a-ref")
        assert removed is False

    def test_remove_cleans_empty_repo_dir(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        registry.remove(_REF)
        repo_dir = tmp_path / "manifests" / repo_to_dir(_REPO)
        assert not repo_dir.exists()


class TestModelRegistryList:
    def test_list_empty(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.list_installed() == []

    def test_list_skips_old_layout(self, tmp_path: Path) -> None:
        """Old manifests/<name>/<tag>.json files are ignored (hard cut)."""
        registry = ModelRegistry(tmp_path)
        old_dir = tmp_path / "manifests" / "qwen3"
        old_dir.mkdir(parents=True)
        (old_dir / "0.6b.json").write_text(
            json.dumps({"name": "qwen3", "tag": "0.6b", "size_bytes": 0, "task": "chat"})
        )
        # Old layout files do not match the new "*.gguf.json" glob, so
        # list_installed silently ignores them.
        assert registry.list_installed() == []

    def test_list_multi_quant_same_repo(self, tmp_path: Path) -> None:
        """Two quants from the same repo coexist as distinct manifests."""
        registry = ModelRegistry(tmp_path)
        src1 = tmp_path / "q4.gguf"
        src1.write_bytes(b"GGUF" + b"\x01" * 50)
        src2 = tmp_path / "q8.gguf"
        src2.write_bytes(b"GGUF" + b"\x02" * 100)

        registry.install(
            _REPO,
            "Qwen3-0.6B-Q4_K_M.gguf",
            src1,
            _make_manifest(gguf_filename="Qwen3-0.6B-Q4_K_M.gguf"),
        )
        registry.install(
            _REPO,
            "Qwen3-0.6B-Q8_0.gguf",
            src2,
            _make_manifest(gguf_filename="Qwen3-0.6B-Q8_0.gguf"),
        )

        listed = registry.list_installed()
        refs = [m.ref for m in listed]
        assert f"{_REPO}/Qwen3-0.6B-Q4_K_M.gguf" in refs
        assert f"{_REPO}/Qwen3-0.6B-Q8_0.gguf" in refs

    def test_list_skips_corrupt_manifest(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        repo_dir = tmp_path / "manifests" / repo_to_dir(_REPO)
        repo_dir.mkdir(parents=True)
        (repo_dir / f"{_FILENAME}.json").write_text("{not json")
        assert registry.list_installed() == []

    def test_list_skips_non_directory_entry(self, tmp_path: Path) -> None:
        """A stray file at the top of manifests/ is ignored, not crashed on."""
        registry = ModelRegistry(tmp_path)
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        (manifests_dir / "stray-file.txt").write_text("not a repo dir")
        assert registry.list_installed() == []


class TestModelRegistryWriteManifestErrors:
    def test_write_failure_cleans_up_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If os.replace fails, the temp file is unlinked instead of leaking."""
        from lilbee.modelhub import registry as registry_mod

        registry = ModelRegistry(tmp_path)
        manifest = _make_manifest()

        def boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("simulated rename failure")

        monkeypatch.setattr(registry_mod.os, "replace", boom)
        with pytest.raises(OSError, match="simulated rename failure"):
            registry._write_manifest(manifest)

        repo_dir = tmp_path / "manifests" / repo_to_dir(_REPO)
        leftovers = [p for p in repo_dir.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


class TestModelRegistryGetManifest:
    def test_get_manifest(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest(task="embedding"))
        m = registry.get_manifest(_REF)
        assert m is not None
        assert m.task == "embedding"

    def test_get_manifest_missing(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.get_manifest(_REF) is None

    def test_get_manifest_invalid_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert registry.get_manifest("not-a-ref") is None
