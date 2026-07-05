"""Tests for registry.py: manifest-keyed by (hf_repo, gguf_filename)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from unittest import mock

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
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


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


_FAKE_REV = "0123456789abcdef0123456789abcdef01234567"  # 40-hex commit-hash-shaped revision


def _seed_hf_cache(
    models_dir: Path,
    *,
    repo: str = _REPO,
    filename: str = _FILENAME,
    content: bytes = b"GGUF\x00\x00",
) -> Path:
    """Lay down a faithful HuggingFace cache entry (blobs/ + snapshots/<rev>/ symlink + refs/main).

    Returns the blob path. Mirrors what ``huggingface_hub`` writes so its cache
    helpers (``try_to_load_from_cache`` / ``scan_cache_dir``) resolve it.
    """
    digest = hashlib.sha256(content).hexdigest()
    cache = models_dir / f"models--{repo_to_dir(repo)}"
    (cache / "blobs").mkdir(parents=True, exist_ok=True)
    blob = cache / "blobs" / digest
    blob.write_bytes(content)
    snap = cache / "snapshots" / _FAKE_REV
    # *filename* may carry a quant subdir (e.g. ``Q4_K_M/m-Q4_K_M.gguf``), which
    # HF preserves in the snapshot tree; mirror that nesting here.
    link = snap / filename
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(blob)
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text(_FAKE_REV)
    return blob


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

    def test_subdir_quant_ref(self) -> None:
        # unsloth giant: repo is the first two segments, the rest (incl. the
        # ``Q4_K_M/`` subdir) is the filename.
        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        repo, filename = parse_hf_ref(ref)
        assert repo == "unsloth/MiniMax-M2-GGUF"
        assert filename == "Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"

    def test_subdir_ref_round_trips(self) -> None:
        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        repo, filename = parse_hf_ref(ref)
        assert format_native_gguf_ref(repo, filename) == ref


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

    def test_filename_subdir_accepted(self) -> None:
        # unsloth giants store each quant under a subdir (e.g. ``Q4_K_M/``); the
        # validator keeps the subdir so the manifest key round-trips with the ref.
        subdir = "Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        assert _validate_gguf_filename(subdir) == subdir

    def test_filename_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            _validate_gguf_filename("..gguf")

    def test_filename_parent_traversal_rejected(self) -> None:
        with pytest.raises(ValueError):
            _validate_gguf_filename("../Q4_K_M/m.gguf")

    def test_filename_leading_slash_rejected(self) -> None:
        with pytest.raises(ValueError):
            _validate_gguf_filename("/abs/path/m.gguf")


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


class TestBlobDigest:
    def test_reuses_hf_cache_blob_name_without_hashing(self, tmp_path: Path, monkeypatch) -> None:
        """A snapshot path resolving into blobs/<sha256> must not re-read the file.

        Registration of a 130 GB split GGUF was failing on network volumes because
        the redundant full-file hash hit transient I/O errors; the blob name is the
        digest already.
        """
        from lilbee.modelhub import registry as registry_mod

        digest = hashlib.sha256(b"payload").hexdigest()
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        blob = blobs / digest
        blob.write_bytes(b"payload")
        snapshot = tmp_path / "snapshots" / "rev"
        snapshot.mkdir(parents=True)
        link = snapshot / "model.gguf"
        link.symlink_to(blob)

        def _boom(path):
            raise AssertionError("must not hash a cache blob")

        monkeypatch.setattr(registry_mod, "_sha256_file", _boom)
        assert registry_mod._blob_digest(link) == digest

    def test_plain_file_falls_back_to_hashing(self, tmp_path: Path) -> None:
        from lilbee.modelhub.registry import _blob_digest

        p = tmp_path / "model.gguf"
        p.write_bytes(b"payload")
        assert _blob_digest(p) == hashlib.sha256(b"payload").hexdigest()


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

    def test_install_blob_copy_is_atomic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-copy leaves no partial blob at the final blob path.

        The blob is written to a temp path and atomically renamed, so an
        interrupted install never exposes a truncated blob under its digest.
        """
        from lilbee.modelhub import registry as registry_mod

        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path, content=b"GGUF" + b"\x07" * 256)
        digest = _sha256_file(src)

        def boom(*_a: object, **_k: object) -> None:
            raise OSError("disk full mid-copy")

        monkeypatch.setattr(registry_mod.shutil, "copyfileobj", boom)
        with pytest.raises(OSError, match="disk full mid-copy"):
            registry.install(_REPO, _FILENAME, src, _make_manifest())

        blob_path = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs" / digest
        assert not blob_path.exists()  # no partial blob at the final path

    def test_install_records_blob_digest(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        content = b"GGUF" + b"\x00" * 256
        src = tmp_path / "source.gguf"
        src.write_bytes(content)
        manifest = _make_manifest(size_bytes=len(content))

        registry.install(_REPO, _FILENAME, src, manifest)
        listed = registry.list_installed()
        assert listed[0].blob == hashlib.sha256(content).hexdigest()

    def test_list_installed_includes_quant_subdir_ref(self, tmp_path: Path) -> None:
        # unsloth stores quants under a subdirectory (Q4_K_S/<model>.gguf), so the
        # manifest is written one level deeper than a flat ref. A non-recursive scan
        # dropped it from `model list` + /v1/models, silently sending opencode to
        # its fallback provider, so the recursive scan must surface it.
        repo = "unsloth/MiniMax-M2-GGUF"
        subdir_ref = "Q4_K_S/MiniMax-M2-Q4_K_S-00001-of-00003.gguf"
        registry = ModelRegistry(tmp_path)
        content = b"GGUF" + b"\x00" * 256
        src = tmp_path / "source.gguf"
        src.write_bytes(content)
        manifest = _make_manifest(hf_repo=repo, gguf_filename=subdir_ref, size_bytes=len(content))
        registry.install(repo, subdir_ref, src, manifest)
        assert f"{repo}/{subdir_ref}" in [m.ref for m in registry.list_installed()]

    def test_list_installed_includes_flat_and_subdir_quants_in_one_repo(
        self, tmp_path: Path
    ) -> None:
        # A flat quant and a quant-subdir quant of the same repo both register and
        # surface as distinct refs (the recursive scan must not drop or merge them).
        repo = "unsloth/Some-GGUF"
        registry = ModelRegistry(tmp_path)
        flat = "Some-Q8_0.gguf"
        subdir = "Q4_K_M/Some-Q4_K_M.gguf"
        for filename, byte in ((flat, b"\x01"), (subdir, b"\x02")):
            content = b"GGUF" + byte * 256
            # Name the temp source by the byte's hex, not repr(): repr(b"\x01") is
            # "b'\\x01'", whose backslash is a path separator on Windows.
            src = tmp_path / f"src-{byte.hex()}.gguf"
            src.write_bytes(content)
            registry.install(
                repo,
                filename,
                src,
                _make_manifest(hf_repo=repo, gguf_filename=filename, size_bytes=len(content)),
            )
        refs = {m.ref for m in registry.list_installed()}
        assert refs == {f"{repo}/{flat}", f"{repo}/{subdir}"}


class TestModelRegistryResolve:
    def test_resolve_repo_only_finds_quant_subdir_manifest(self, tmp_path: Path) -> None:
        # A bare org/repo ref must resolve through the fast manifest path even when
        # the installed quant lives in a subdir, not fall through to cache recovery.
        repo = "unsloth/Some-GGUF"
        subdir_ref = "Q4_K_M/Some-Q4_K_M.gguf"
        registry = ModelRegistry(tmp_path)
        content = b"GGUF" + b"\x00" * 256
        src = tmp_path / "source.gguf"
        src.write_bytes(content)
        registry.install(
            repo,
            subdir_ref,
            src,
            _make_manifest(hf_repo=repo, gguf_filename=subdir_ref, size_bytes=len(content)),
        )
        assert registry.resolve(repo) == registry.resolve(f"{repo}/{subdir_ref}")

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

    def test_split_gguf_first_shard_only_not_installed(self, tmp_path: Path) -> None:
        """A split GGUF with only its first shard cached must read as not installed."""
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        _seed_hf_cache(
            tmp_path, repo=repo, filename="m-mxfp4-00001-of-00003.gguf", content=b"shard-1"
        )
        ref = f"{repo}/m-mxfp4-00001-of-00003.gguf"
        assert registry.is_installed(ref) is False
        with pytest.raises(KeyError, match="missing shards"):
            registry.resolve(ref)

    def test_split_gguf_all_shards_present_resolves(self, tmp_path: Path) -> None:
        """Once every shard is cached, the split GGUF resolves and reads installed."""
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"m-mxfp4-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        ref = f"{repo}/m-mxfp4-00001-of-00003.gguf"
        assert registry.is_installed(ref) is True
        assert registry.resolve(ref).exists()

    def test_split_gguf_resolves_to_snapshot_symlink_with_siblings(self, tmp_path: Path) -> None:
        """A split GGUF resolves to its first shard's snapshot symlink, not its blob.

        llama.cpp loads the whole set from the first shard and finds the siblings
        by filename next to it; only the snapshot dir co-locates them under their
        real ``-0000k-of-0000N`` names (blobs are hash-named). Returning the blob
        path is what made gpt-oss-120b fail to load on the H200.
        """
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"m-mxfp4-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        resolved = registry.resolve(f"{repo}/m-mxfp4-00001-of-00003.gguf")
        assert resolved.name == "m-mxfp4-00001-of-00003.gguf"  # the symlink, not a hash blob
        assert resolved.parent.name == _FAKE_REV  # under snapshots/<rev>/, not blobs/
        # Every sibling shard sits next to it under its real name.
        for n in (2, 3):
            assert (resolved.parent / f"m-mxfp4-0000{n}-of-00003.gguf").exists()

    def test_bare_repo_recovers_split_gguf_from_raw_cache(self, tmp_path: Path) -> None:
        """bb-z59: a bare org/repo ref pointing at a raw-cache split GGUF (hf
        download, no manifest) resolves to the first shard's snapshot symlink with
        shard accounting, not shard 1's blob as an unloadable single file."""
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"m-mxfp4-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        resolved = registry.resolve(repo)
        # The bare-repo path lands on the same loadable snapshot symlink as the
        # canonical first-shard ref, not a blob under blobs/.
        assert resolved == registry.resolve(f"{repo}/m-mxfp4-00001-of-00003.gguf")
        assert resolved.parent.name == _FAKE_REV
        # Recovery wrote a manifest with full shard accounting, so it lists
        # installed: shard_blobs holds the trailing shards (2 and 3); the first
        # shard is the primary blob.
        manifest = registry._read_manifest(repo, "m-mxfp4-00001-of-00003.gguf")
        assert manifest is not None
        assert len(manifest.shard_blobs) == 2
        assert manifest.total_size_bytes == sum(len(f"shard-{n}".encode()) for n in (1, 2, 3))

    def test_bare_repo_recovers_subdir_split_gguf_from_raw_cache(self, tmp_path: Path) -> None:
        """Real quant repos (e.g. unsloth) place their shards under a quant subdir;
        a bare-repo ref must recover that split set from a raw cache too, not just a
        flat one."""
        registry = ModelRegistry(tmp_path)
        repo = "unsloth/MiniMax-M2-GGUF"
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"Q4_K_M/MiniMax-M2-Q4_K_M-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        resolved = registry.resolve(repo)
        assert resolved == registry.resolve(f"{repo}/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf")
        assert resolved.parent.name == "Q4_K_M"  # co-located under the quant subdir
        for n in (2, 3):
            assert (resolved.parent / f"MiniMax-M2-Q4_K_M-0000{n}-of-00003.gguf").exists()

    def test_bare_repo_split_that_cannot_resolve_skips_shards_and_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare repo whose only cached GGUFs are an unresolvable split set (a shard
        missing) suppresses shard 1's failure, skips the trailing shards, and ends in
        'not installed' rather than handing back a partial resolve."""
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"m-mxfp4-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        # Every shard reports missing, so _resolve_split raises on shard 1 (suppressed);
        # shards 2 and 3 are skipped as non-first, and the loop exhausts unresolved.
        monkeypatch.setattr(registry, "_split_shards_present", lambda *_a, **_k: False)
        with pytest.raises(KeyError, match="not installed"):
            registry.resolve(repo)

    def test_shard_paths_returns_every_split_shard(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"m-mxfp4-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        paths = registry.shard_paths(f"{repo}/m-mxfp4-00001-of-00003.gguf")
        assert [p.name for p in paths] == [f"m-mxfp4-0000{n}-of-00003.gguf" for n in (1, 2, 3)]
        assert all(p.exists() for p in paths)

    def test_shard_paths_empty_for_single_file_blob_resolution(self, tmp_path: Path) -> None:
        """A single-file GGUF resolves to its hash-named blob, so no sibling exists
        under the real filename and the shard list is empty (callers floor on 0 bytes)."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        assert registry.shard_paths(_REF) == []

    def test_shard_paths_raises_for_uninstalled_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        with pytest.raises(KeyError, match="not installed"):
            registry.shard_paths(_REF)

    def _seed_split(self, tmp_path: Path, repo: str) -> str:
        for n in (1, 2, 3):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"m-mxfp4-0000{n}-of-00003.gguf",
                content=f"shard-{n}".encode(),
            )
        return f"{repo}/m-mxfp4-00001-of-00003.gguf"

    def test_cache_recovered_split_records_shard_accounting(self, tmp_path: Path) -> None:
        # A cache-only split GGUF (no manifest yet)
        # must recover its total size and every shard digest, not just the first.
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        ref = self._seed_split(tmp_path, repo)
        registry.resolve(ref)  # triggers _reregister_from_cache
        manifest = registry._read_manifest(repo, "m-mxfp4-00001-of-00003.gguf")
        assert manifest is not None
        assert manifest.total_size_bytes == sum(len(f"shard-{n}".encode()) for n in (1, 2, 3))
        assert len(manifest.shard_blobs) == 2  # shards 2 and 3
        assert all(_SHA256_HEX_RE.fullmatch(d) for d in manifest.shard_blobs)

    def test_recover_legacy_shard_blobs_finds_extra_shards(self, tmp_path: Path) -> None:
        # A pre-accounting manifest (empty shard_blobs)
        # still frees every shard because removal recovers them from the cache.
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        ref = self._seed_split(tmp_path, repo)
        registry.resolve(ref)
        blobs = registry._recover_legacy_shard_blobs(ref)
        assert len(blobs) == 2  # shards 2 and 3 (primary excluded)
        assert all(_SHA256_HEX_RE.fullmatch(d) for d in blobs)

    def test_recover_legacy_shard_blobs_empty_for_single_file(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        _seed_hf_cache(tmp_path)
        assert registry._recover_legacy_shard_blobs(_REF) == []

    def test_recover_legacy_shard_blobs_empty_when_unresolvable(self, tmp_path: Path) -> None:
        # Uninstalled ref: shard_paths raises, suppressed so removal never breaks.
        registry = ModelRegistry(tmp_path)
        assert registry._recover_legacy_shard_blobs(_REF) == []

    def test_split_gguf_shards_present_but_snapshot_missing_raises_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shards register as present, but if the first shard's snapshot symlink
        can't be located, resolve raises 'not installed' rather than returning None."""
        registry = ModelRegistry(tmp_path)
        repo = "ggml-org/gpt-oss-120b-GGUF"
        # Presence check forced True; snapshot resolution forced None. This is the
        # narrow window where every shard reports present but the first shard's
        # snapshot symlink is gone -- resolve must surface a clean 'not installed'.
        monkeypatch.setattr(registry, "_split_shards_present", lambda *_a, **_k: True)
        monkeypatch.setattr(registry, "_snapshot_gguf_path", lambda *_a, **_k: None)
        with pytest.raises(KeyError, match="not installed"):
            registry.resolve(f"{repo}/m-mxfp4-00001-of-00003.gguf")

    def test_split_shards_present_true_for_single_file_gguf(self, tmp_path: Path) -> None:
        """A non-split filename has exactly one 'shard' and is trivially present:
        the presence check short-circuits without touching the cache."""
        registry = ModelRegistry(tmp_path)
        assert registry._split_shards_present("org/repo-GGUF", "model-Q4_K_M.gguf") is True

    def test_subdir_quant_single_file_resolves(self, tmp_path: Path) -> None:
        """A single-file quant nested in a ``Q4_K_M/`` subdir resolves by basename.

        unsloth-style repos store each quant in its own subdir; the lilbee ref
        keys on the basename, so cache lookup must find it wherever HF nested it.
        """
        registry = ModelRegistry(tmp_path)
        repo = "unsloth/SomeModel-GGUF"
        _seed_hf_cache(
            tmp_path, repo=repo, filename="Q4_K_M/SomeModel-Q4_K_M.gguf", content=b"single"
        )
        resolved = registry.resolve(f"{repo}/SomeModel-Q4_K_M.gguf")
        assert resolved.exists()

    def test_subdir_split_gguf_resolves_to_snapshot_symlink(self, tmp_path: Path) -> None:
        """The glm-air case: a split quant under ``Q4_K_M/`` resolves to its
        snapshot symlink with both shards co-located, addressed by basename ref."""
        registry = ModelRegistry(tmp_path)
        repo = "unsloth/GLM-4.5-Air-GGUF"
        for n in (1, 2):
            _seed_hf_cache(
                tmp_path,
                repo=repo,
                filename=f"Q4_K_M/GLM-4.5-Air-Q4_K_M-0000{n}-of-00002.gguf",
                content=f"glm-shard-{n}".encode(),
            )
        ref = f"{repo}/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
        assert registry.is_installed(ref) is True
        resolved = registry.resolve(ref)
        assert resolved.name == "GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
        assert resolved.parent.name == "Q4_K_M"  # subdir preserved in the snapshot tree
        assert (resolved.parent / "GLM-4.5-Air-Q4_K_M-00002-of-00002.gguf").exists()

    def test_subdir_split_gguf_missing_shard_not_installed(self, tmp_path: Path) -> None:
        """A subdir split quant missing its second shard reads as not installed."""
        registry = ModelRegistry(tmp_path)
        repo = "unsloth/GLM-4.5-Air-GGUF"
        _seed_hf_cache(
            tmp_path,
            repo=repo,
            filename="Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf",
            content=b"glm-shard-1",
        )
        ref = f"{repo}/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"
        assert registry.is_installed(ref) is False
        with pytest.raises(KeyError, match="missing shards"):
            registry.resolve(ref)

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

    def test_resolve_recovers_from_cache_without_manifest(self, tmp_path: Path) -> None:
        """A GGUF in the HF cache resolves even with no lilbee manifest (e.g. after an upgrade)."""
        registry = ModelRegistry(tmp_path)
        blob = _seed_hf_cache(tmp_path)
        assert registry.resolve(_REF) == blob.resolve()
        assert registry.is_installed(_REF)

    def test_resolve_recovery_writes_a_fresh_manifest(self, tmp_path: Path) -> None:
        """Recovering a featured model from the cache also re-registers it, so it
        shows up in ``list_installed`` (and ``lilbee model list`` / the TUI catalog)."""
        registry = ModelRegistry(tmp_path)
        _seed_hf_cache(tmp_path)
        assert not registry.list_installed()  # no manifest yet
        registry.resolve(_REF)
        assert any(m.ref == _REF for m in registry.list_installed())

    def test_resolve_recovery_survives_manifest_write_failure(self, tmp_path: Path) -> None:
        """If re-registering after a cache recovery fails, ``resolve`` still returns the path."""
        registry = ModelRegistry(tmp_path)
        blob = _seed_hf_cache(tmp_path)
        with mock.patch.object(ModelRegistry, "_write_manifest", side_effect=OSError("disk full")):
            assert registry.resolve(_REF) == blob.resolve()
        assert not registry.list_installed()  # the write failed, so still no manifest

    def test_resolve_bare_repo_ref_recovers_from_cache(self, tmp_path: Path) -> None:
        """A bare ``<org>/<repo>`` ref (older builds persisted these) resolves via the HF cache."""
        registry = ModelRegistry(tmp_path)
        blob = _seed_hf_cache(tmp_path)
        assert registry.resolve(_REPO) == blob.resolve()
        assert registry.is_installed(_REPO)

    def test_resolve_split_ref_reregisters_from_cache(self, tmp_path: Path) -> None:
        """A cached split GGUF resolved by file ref gets its manifest rewritten.

        The split path resolves via snapshot symlinks without reading manifests,
        so it previously never re-registered: the model stayed resolvable but
        absent from every listing, and re-pulls short-circuited forever.
        """
        repo = "unsloth/Split-GGUF"
        shards = [f"Split-Q4_K_S-0000{i}-of-00002.gguf" for i in (1, 2)]
        registry = ModelRegistry(tmp_path)
        for shard in shards:
            _seed_hf_cache(tmp_path, repo=repo, filename=shard, content=shard.encode())
        assert not registry.list_installed()
        registry.resolve(f"{repo}/{shards[0]}")
        assert [m.ref for m in registry.list_installed()] == [f"{repo}/{shards[0]}"]

    def test_resolve_recovery_registers_non_catalog_ref_as_chat(self, tmp_path: Path) -> None:
        """A cache recovery of a NON-featured repo still writes a manifest (task=chat).

        Skipping it left the model permanently half-installed: ``is_installed``
        (and so a re-pull's "already installed" short-circuit) saw it, while
        ``list_installed`` (the launcher, the fleet, ``lilbee model list``) did
        not, so it could never be selected or repaired by re-pulling.
        """
        from lilbee.catalog.types import ModelTask

        repo = "unsloth/NotFeatured-GGUF"
        filename = "NotFeatured-Q4_K_S.gguf"
        registry = ModelRegistry(tmp_path)
        _seed_hf_cache(tmp_path, repo=repo, filename=filename)
        assert not registry.list_installed()
        registry.resolve(repo)
        installed = registry.list_installed()
        assert [m.ref for m in installed] == [f"{repo}/{filename}"]
        assert installed[0].task == ModelTask.CHAT

    def test_resolve_bare_repo_ref_uses_manifest_when_present(self, tmp_path: Path) -> None:
        """A bare repo ref prefers a current-format manifest under that repo."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        blob_path = registry.install(_REPO, _FILENAME, src, _make_manifest())
        assert registry.resolve(_REPO) == blob_path

    def test_resolve_bare_repo_ref_skips_unparseable_manifest(self, tmp_path: Path) -> None:
        """A bare repo ref skips an unreadable per-repo manifest and falls back to the cache."""
        registry = ModelRegistry(tmp_path)
        blob = _seed_hf_cache(tmp_path)
        bad = tmp_path / "manifests" / repo_to_dir(_REPO) / f"{_FILENAME}.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("not valid json at all")
        assert registry.resolve(_REPO) == blob.resolve()

    def test_resolve_bare_repo_ref_not_installed(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        with pytest.raises(KeyError, match="not installed"):
            registry.resolve(_REPO)

    def test_resolve_recovers_when_manifest_unparseable(self, tmp_path: Path) -> None:
        """An unreadable / older-format manifest is ignored; the cache is the source of truth."""
        registry = ModelRegistry(tmp_path)
        blob = _seed_hf_cache(tmp_path)
        bad = tmp_path / "manifests" / repo_to_dir(_REPO) / f"{_FILENAME}.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text('{"repo": "old-format", "extra": "fields the current schema lacks"}')
        assert registry.resolve(_REF) == blob.resolve()

    def test_recovery_ignores_snapshot_symlink_escaping_root(self, tmp_path: Path) -> None:
        """A snapshot entry symlinking outside the registry root is not followed."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        registry = ModelRegistry(models_dir)
        outside = tmp_path / "outside.gguf"  # not under models_dir
        outside.write_bytes(b"escaped the cache")
        cache = models_dir / f"models--{repo_to_dir(_REPO)}"
        snap = cache / "snapshots" / _FAKE_REV
        snap.mkdir(parents=True)
        (snap / _FILENAME).symlink_to(outside)
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text(_FAKE_REV)
        with pytest.raises(KeyError, match="not installed"):
            registry.resolve(_REF)


class TestRegisterDownloadedModel:
    def test_subdir_filename_round_trips(self, tmp_path: Path) -> None:
        """A subdir-quant giant registers under its subdir-relative name.

        The snapshot path is ``.../snapshots/<rev>/Q4_K_M/<file>.gguf``; the
        manifest must key on ``Q4_K_M/<file>.gguf`` so the canonical ref
        resolves back to the same blob (F2: subdir giants are first-class).
        """
        from lilbee.catalog.models import CatalogModel
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.registry import register_downloaded_model

        repo = "unsloth/MiniMax-M2-GGUF"
        subdir_name = "Q4_K_M/MiniMax-M2-Q4_K_M.gguf"
        blob = _seed_hf_cache(tmp_path, repo=repo, filename=subdir_name, content=b"giant")
        snapshot_path = (
            tmp_path / f"models--{repo_to_dir(repo)}" / "snapshots" / _FAKE_REV / subdir_name
        )
        entry = CatalogModel(
            hf_repo=repo,
            gguf_filename=subdir_name,
            size_gb=0.0,
            min_ram_gb=2.0,
            description="",
            featured=False,
            downloads=0,
            task=ModelTask.CHAT,
        )
        with mock.patch("lilbee.modelhub.registry.cfg") as cfg_mock:
            cfg_mock.models_dir = tmp_path
            register_downloaded_model(entry, snapshot_path)
            registry = ModelRegistry(tmp_path)
            ref = format_native_gguf_ref(repo, subdir_name)
            manifest = registry.get_manifest(ref)
            assert manifest is not None
            assert manifest.gguf_filename == subdir_name
            assert registry.resolve(ref) == blob


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

    def test_is_installed_false_for_truncated_blob(self, tmp_path: Path) -> None:
        """A blob truncated below its manifest size_bytes is not 'installed'.

        A partial/corrupt blob on disk must not count as usable, otherwise
        the runtime loads a truncated GGUF and fails far from the cause.
        """
        registry = ModelRegistry(tmp_path)
        content = b"GGUF" + b"\x00" * 512
        src = tmp_path / "source.gguf"
        src.write_bytes(content)
        registry.install(_REPO, _FILENAME, src, _make_manifest(size_bytes=len(content)))
        blob_dir = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        blob = next(blob_dir.iterdir())
        blob.write_bytes(content[: len(content) // 2])  # truncate on disk
        assert not registry.is_installed(_REF)


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

    def test_remove_gcs_all_split_shard_blobs(self, tmp_path: Path) -> None:
        """A split GGUF's extra shard blobs are freed on remove.

        A sibling quant keeps the repo cache dir alive, so removal must gc each
        shard blob individually rather than relying on wiping the whole repo dir.
        """
        registry = ModelRegistry(tmp_path)
        sib = tmp_path / "sib.gguf"
        sib.write_bytes(b"GGUF" + b"\x09" * 30)
        registry.install(_REPO, "Sibling.gguf", sib, _make_manifest(gguf_filename="Sibling.gguf"))
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        blobs = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        (blobs / "shard2digest").write_bytes(b"x" * 10)
        (blobs / "shard3digest").write_bytes(b"x" * 10)
        manifest = registry._read_manifest(_REPO, _FILENAME)
        assert manifest is not None
        manifest.shard_blobs = ["shard2digest", "shard3digest"]
        registry._write_manifest(manifest)

        registry.remove(_REF)
        assert not (blobs / "shard2digest").exists()
        assert not (blobs / "shard3digest").exists()
        assert registry.is_installed(f"{_REPO}/Sibling.gguf")

    def test_remove_reads_manifests_once_for_multi_shard(self, tmp_path: Path) -> None:
        """Freeing N shard blobs must not re-walk the manifest tree per shard;
        list_installed is read once and shared across the blob GC calls."""
        registry = ModelRegistry(tmp_path)
        sib = tmp_path / "sib.gguf"
        sib.write_bytes(b"GGUF" + b"\x09" * 30)
        registry.install(_REPO, "Sibling.gguf", sib, _make_manifest(gguf_filename="Sibling.gguf"))
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        blobs = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        (blobs / "shard2digest").write_bytes(b"x" * 10)
        (blobs / "shard3digest").write_bytes(b"x" * 10)
        manifest = registry._read_manifest(_REPO, _FILENAME)
        assert manifest is not None
        manifest.shard_blobs = ["shard2digest", "shard3digest"]
        registry._write_manifest(manifest)

        # Primary + 2 shards = 3 blob GC calls, but only one manifest-tree walk.
        with mock.patch.object(registry, "list_installed", wraps=registry.list_installed) as spy:
            registry.remove(_REF)
        assert spy.call_count == 1

    def test_remove_keeps_shard_blob_referenced_by_sibling(self, tmp_path: Path) -> None:
        """A shard digest still referenced by a sibling's shard_blobs survives."""
        registry = ModelRegistry(tmp_path)
        sib = tmp_path / "sib.gguf"
        sib.write_bytes(b"GGUF" + b"\x09" * 30)
        registry.install(_REPO, "Sibling.gguf", sib, _make_manifest(gguf_filename="Sibling.gguf"))
        sibling = registry._read_manifest(_REPO, "Sibling.gguf")
        assert sibling is not None
        sibling.shard_blobs = ["shared"]
        registry._write_manifest(sibling)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        blobs = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        (blobs / "shared").write_bytes(b"x" * 10)
        (blobs / "unique").write_bytes(b"y" * 10)
        manifest = registry._read_manifest(_REPO, _FILENAME)
        assert manifest is not None
        manifest.shard_blobs = ["shared", "unique"]  # remove() must iterate both
        registry._write_manifest(manifest)

        registry.remove(_REF)
        assert (blobs / "shared").exists()  # still referenced by the sibling
        assert not (blobs / "unique").exists()  # removed: proves remove() gc's shard_blobs

    def test_disk_size_bytes_uses_total_for_split(self, tmp_path: Path) -> None:
        manifest = _make_manifest()
        manifest.size_bytes = 100  # first shard only
        assert manifest.disk_size_bytes == 100  # single-file: falls back to size_bytes
        manifest.total_size_bytes = 600  # six-shard total
        assert manifest.disk_size_bytes == 600

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

    def test_list_skips_manifest_with_null_blob(self, tmp_path: Path) -> None:
        """A manifest whose blob field is null (interrupted install) is hidden.

        Pickers source their options from list_installed, so an entry
        without a blob hash would let the user select an unusable model.
        """
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        manifest_file = tmp_path / "manifests" / repo_to_dir(_REPO) / f"{_FILENAME}.json"
        data = json.loads(manifest_file.read_text())
        data["blob"] = None
        manifest_file.write_text(json.dumps(data))
        assert registry.list_installed() == []

    def test_list_skips_manifest_with_missing_blob_file(self, tmp_path: Path) -> None:
        """Manifest references a digest whose blob no longer exists on disk.

        Happens when the HF cache is cleared externally or a download
        was canceled mid-stream. The manifest survives but the model
        cannot be loaded; pickers must not surface it.
        """
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        blob_dir = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        for blob in blob_dir.iterdir():
            blob.unlink()
        assert registry.list_installed() == []

    def test_list_skips_truncated_blob(self, tmp_path: Path) -> None:
        """A blob truncated below its recorded size_bytes is hidden from pickers."""
        registry = ModelRegistry(tmp_path)
        content = b"GGUF" + b"\x01" * 512
        src = tmp_path / "source.gguf"
        src.write_bytes(content)
        registry.install(_REPO, _FILENAME, src, _make_manifest(size_bytes=len(content)))
        blob_dir = tmp_path / f"models--{repo_to_dir(_REPO)}" / "blobs"
        next(blob_dir.iterdir()).write_bytes(content[:10])
        assert registry.list_installed() == []

    def test_list_skips_manifest_when_cache_dir_missing(self, tmp_path: Path) -> None:
        """The whole per-repo cache dir was wiped; the orphan manifest stays hidden."""
        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        cache_dir = tmp_path / f"models--{repo_to_dir(_REPO)}"
        for child in cache_dir.rglob("*"):
            if child.is_file():
                child.unlink()
        for sub in sorted(cache_dir.rglob("*"), key=lambda p: -len(p.parts)):
            if sub.is_dir():
                sub.rmdir()
        cache_dir.rmdir()
        assert registry.list_installed() == []

    def test_list_keeps_complete_alongside_incomplete(self, tmp_path: Path) -> None:
        """Two manifests, one complete and one with a null blob: only the complete survives."""
        registry = ModelRegistry(tmp_path)
        complete_src = tmp_path / "complete.gguf"
        complete_src.write_bytes(b"GGUF" + b"\x01" * 64)
        registry.install(
            _REPO,
            "Qwen3-0.6B-Q4_K_M.gguf",
            complete_src,
            _make_manifest(gguf_filename="Qwen3-0.6B-Q4_K_M.gguf"),
        )
        broken_src = tmp_path / "broken.gguf"
        broken_src.write_bytes(b"GGUF" + b"\x02" * 64)
        registry.install(
            _REPO,
            "Qwen3-0.6B-Q8_0.gguf",
            broken_src,
            _make_manifest(gguf_filename="Qwen3-0.6B-Q8_0.gguf"),
        )
        broken_manifest = tmp_path / "manifests" / repo_to_dir(_REPO) / "Qwen3-0.6B-Q8_0.gguf.json"
        broken_data = json.loads(broken_manifest.read_text())
        broken_data["blob"] = None
        broken_manifest.write_text(json.dumps(broken_data))
        listed = registry.list_installed()
        assert [m.gguf_filename for m in listed] == ["Qwen3-0.6B-Q4_K_M.gguf"]


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


class TestInstalledRefForRepo:
    def test_returns_installed_quant_ref(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        registry.install(_REPO, _FILENAME, _write_source(tmp_path), _make_manifest())
        assert registry.installed_ref_for_repo(_REPO) == _REF

    def test_multiple_quants_picks_alphabetical_first(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        registry.install(_REPO, "z-Q8_0.gguf", _write_source(tmp_path), _make_manifest())
        registry.install(_REPO, "a-Q4_K_M.gguf", _write_source(tmp_path), _make_manifest())
        assert registry.installed_ref_for_repo(_REPO) == f"{_REPO}/a-Q4_K_M.gguf"

    def test_unknown_repo_returns_none(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        registry.install(_REPO, _FILENAME, _write_source(tmp_path), _make_manifest())
        assert registry.installed_ref_for_repo("other/Repo-GGUF") is None


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


class TestModelRegistryGCBlobPathGuard:
    """``_gc_blob`` refuses to delete cache trees outside ``models_dir``.

    A repo argument whose resolved path falls outside the registry's
    ``_root`` (symlink trickery, ``..`` traversal) hits the
    ``validate_path_within`` guard and the function logs + returns
    instead of removing arbitrary directories.
    """

    def test_refuses_path_outside_models_dir(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from lilbee.modelhub import registry as registry_mod

        registry = ModelRegistry(tmp_path)
        with (
            caplog.at_level(logging.WARNING, logger=registry_mod.__name__),
            mock.patch(
                "lilbee.modelhub.registry.validate_path_within",
                side_effect=ValueError("outside root"),
            ),
        ):
            registry._gc_blob(_REPO, "deadbeef")
        assert any(
            "Refusing to remove cache outside models_dir" in r.message for r in caplog.records
        )

    def test_refuses_blob_digest_with_traversal(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A digest with ``..`` resolves outside the repo's blobs dir; the blob
        path guard must refuse to unlink it. A sibling manifest keeps the repo
        cache alive so the no-siblings rmtree branch isn't taken first."""
        import logging

        from lilbee.modelhub import registry as registry_mod

        registry = ModelRegistry(tmp_path)
        src = _write_source(tmp_path)
        registry.install(_REPO, _FILENAME, src, _make_manifest())
        # A file outside the registry root that a traversal digest would target.
        victim = tmp_path.parent / "victim.bin"
        victim.write_bytes(b"keep me")
        traversal_digest = f"../../../{victim.name}"
        with caplog.at_level(logging.WARNING, logger=registry_mod.__name__):
            registry._gc_blob(_REPO, traversal_digest)
        assert victim.exists()
        assert any(
            "Refusing to remove blob outside models_dir" in r.message for r in caplog.records
        )


class TestShardAccounting:
    def test_single_file_returns_none_and_empty(self, tmp_path: Path) -> None:
        from lilbee.modelhub.registry import _shard_accounting

        f = tmp_path / "model-Q4_K_M.gguf"
        f.write_bytes(b"x" * 10)
        assert _shard_accounting(f) == (None, [])

    def test_multi_shard_sums_size_and_collects_extra_blobs(self, tmp_path: Path) -> None:
        from lilbee.modelhub.registry import _shard_accounting

        names = [
            "m-00001-of-00003.gguf",
            "m-00002-of-00003.gguf",
            "m-00003-of-00003.gguf",
        ]
        for name in names:
            (tmp_path / name).write_bytes(b"x" * 10)
        total, shard_blobs = _shard_accounting(tmp_path / names[0])
        assert total == 30  # all three shards summed
        assert len(shard_blobs) == 2  # shards 2 and 3 (primary tracked separately)
        # Copy/non-symlink mode: digests are content hashes, never snapshot names.
        assert all(_SHA256_HEX_RE.fullmatch(d) for d in shard_blobs)

    def test_multi_shard_skips_missing_shard(self, tmp_path: Path) -> None:
        from lilbee.modelhub.registry import _shard_accounting

        # Only shards 1 and 3 are on disk (2 is missing); the missing one is skipped.
        for name in ("m-00001-of-00003.gguf", "m-00003-of-00003.gguf"):
            (tmp_path / name).write_bytes(b"x" * 10)
        total, shard_blobs = _shard_accounting(tmp_path / "m-00001-of-00003.gguf")
        assert total == 20  # only the two present shards
        assert len(shard_blobs) == 1  # only shard 3 (primary is shard 1)
