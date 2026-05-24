"""Manifest store keyed by ``(hf_repo, gguf_filename)`` over the HF cache.

Canonical ref: ``<hf_repo>/<gguf_filename>``. Two quants of the same
repo are two distinct installations. Manifests live at
``manifests/<repo--repo>/<filename>.json``; blobs at
``models--<repo--repo>/blobs/<sha>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.catalog.refs import format_native_gguf_ref
from lilbee.core.config.model import cfg
from lilbee.core.security import validate_path_within

if TYPE_CHECKING:
    from lilbee.catalog.models import CatalogModel
    from lilbee.catalog.types import ModelTask

log = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 8192  # bytes read per iteration when hashing
_REPO_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._-]+\.gguf$")

REPO_DIR_SEPARATOR = "--"


def _validate_hf_repo(hf_repo: str) -> str:
    """Validate that a HuggingFace repo id has the form ``org/name``."""
    if not hf_repo or not _REPO_SEGMENT_RE.match(hf_repo) or ".." in hf_repo:
        raise ValueError(f"Invalid hf_repo: {hf_repo!r}")
    return hf_repo


def _validate_gguf_filename(filename: str) -> str:
    """Validate that a filename is a safe ``.gguf`` basename (no path separators)."""
    if not filename or not _FILENAME_RE.match(filename) or ".." in filename:
        raise ValueError(f"Invalid gguf_filename: {filename!r}")
    return filename


_REF_SHAPE_HINT = "Use '<org>/<repo>/<filename>.gguf'."


def parse_hf_ref(ref: str) -> tuple[str, str]:
    """Split ``<org>/<repo>/<file>.gguf`` into ``(hf_repo, gguf_filename)``."""
    if not ref.endswith(".gguf") or "/" not in ref:
        raise ValueError(f"Model ref {ref!r} is not a HuggingFace ref. {_REF_SHAPE_HINT}")
    hf_repo, gguf_filename = ref.rsplit("/", 1)
    return _validate_hf_repo(hf_repo), _validate_gguf_filename(gguf_filename)


def repo_to_dir(hf_repo: str) -> str:
    """Encode an HF repo for use as a directory name (HF cache convention)."""
    return hf_repo.replace("/", REPO_DIR_SEPARATOR)


@dataclass
class ModelManifest:
    """One installed model's metadata. Identity: ``(hf_repo, gguf_filename)``."""

    hf_repo: str
    gguf_filename: str
    size_bytes: int
    task: ModelTask
    downloaded_at: str  # ISO 8601
    blob: str | None = None  # SHA-256 hex of the blob in the HF cache; None pre-install

    @property
    def ref(self) -> str:
        return format_native_gguf_ref(self.hf_repo, self.gguf_filename)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    """Read/write manifests and resolve refs to blobs in the HF cache."""

    def __init__(self, models_dir: Path) -> None:
        self._root = models_dir
        self._manifests_dir = models_dir / "manifests"

    def _repo_cache_dir(self, hf_repo: str) -> Path:
        """The HuggingFace cache directory for *hf_repo* under this registry root."""
        return self._root / f"models--{repo_to_dir(hf_repo)}"

    def resolve(self, ref: str) -> Path:
        """Return the blob path for *ref*; ``KeyError`` if not installed.

        The canonical *ref* is ``<org>/<repo>/<file>.gguf`` resolved via the
        lilbee manifest. Two other shapes are accepted as a backwards-compat
        concession for builds already published (whose on-disk layout differs),
        not as the intended contract: a bare ``<org>/<repo>`` (older builds
        persisted these into ``config.toml``) resolves to the one quant of that
        repo that's installed, and a manifest that's missing / unparseable /
        blob-less falls back to whatever GGUF ``huggingface_hub`` reports the
        cache holds for that ref. The HF cache layout is stable, so this lets an
        upgrade keep working without anyone purging their lilbee data dir; it is
        deliberately the exception here, not a pattern to follow elsewhere.
        """
        if not ref.endswith(".gguf") and ref.count("/") == 1:
            return self._resolve_repo_only(_validate_hf_repo(ref))
        hf_repo, gguf_filename = parse_hf_ref(ref)
        manifest = self._read_manifest(hf_repo, gguf_filename)
        # A split GGUF only loads when every shard is on disk. The first shard
        # alone (plus its manifest) used to read as installed, so a pull
        # interrupted between shards registered an unloadable model and a re-pull
        # said "already installed". Treat an incomplete shard set as not installed.
        shards_complete = self._split_shards_present(hf_repo, gguf_filename)
        if manifest is not None and manifest.blob is not None:
            blob_file = self._repo_cache_dir(manifest.hf_repo) / "blobs" / manifest.blob
            if blob_file.exists() and shards_complete:
                return blob_file
        recovered = self._find_cached_gguf(hf_repo, gguf_filename)
        if recovered is not None and shards_complete:
            self._reregister_from_cache(hf_repo, gguf_filename, recovered)
            return recovered
        if not shards_complete:
            raise KeyError(f"Split GGUF {ref} is missing shards; re-pull to fetch the full set")
        if manifest is None:
            raise KeyError(f"Model {ref} not installed")
        # Manifest present but neither it nor the cache yields a blob; keep the
        # specific diagnostic so a corrupted cache stays debuggable.
        cache_path = self._repo_cache_dir(manifest.hf_repo)
        if not cache_path.exists():
            raise KeyError(f"Cache folder missing for {ref}: {cache_path.name}")
        if manifest.blob is None:
            raise KeyError(f"Manifest for {ref} has no blob hash; install incomplete")
        raise KeyError(f"Blob file missing for {ref}: {manifest.blob}")

    def _resolve_repo_only(self, hf_repo: str) -> Path:
        """Resolve a bare ``<org>/<repo>`` ref to the GGUF of that repo on disk.

        Older builds persisted bare repo refs for the chat / embedding model.
        Prefers a current-format manifest under the repo; otherwise asks
        ``huggingface_hub`` what GGUFs the cache holds for the repo and returns
        the first one (alphabetical for determinism if more than one quant is
        installed).
        """
        manifest_dir = self._manifests_dir / repo_to_dir(hf_repo)
        if manifest_dir.is_dir():
            for mf in sorted(manifest_dir.glob("*.gguf.json")):
                manifest = self._load_manifest_file(mf)
                if manifest is None or manifest.blob is None:
                    continue
                blob = self._repo_cache_dir(hf_repo) / "blobs" / manifest.blob
                if blob.exists():
                    return blob
        for filename in sorted(self._cached_gguf_names(hf_repo)):
            recovered = self._find_cached_gguf(hf_repo, filename)
            if recovered is not None:
                self._reregister_from_cache(hf_repo, filename, recovered)
                return recovered
        raise KeyError(f"Model {hf_repo} not installed")

    def _cached_gguf_names(self, hf_repo: str) -> set[str]:
        """``.gguf`` filenames the HuggingFace cache holds for *hf_repo*."""
        if not self._root.is_dir():
            return set()
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(self._root)
        return {
            f.file_name
            for repo in info.repos
            if repo.repo_id == hf_repo
            for rev in repo.revisions
            for f in rev.files
            if f.file_name.endswith(".gguf")
        }

    def _find_cached_gguf(self, hf_repo: str, gguf_filename: str) -> Path | None:
        """Return the cached blob path for ``hf_repo``/``gguf_filename``, or None.

        Uses ``huggingface_hub.try_to_load_from_cache`` so we honor whatever
        cache layout HF uses, then resolves the returned snapshot symlink to the
        blob, bounded to the cache directory.
        """
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(
            repo_id=hf_repo, filename=gguf_filename, cache_dir=str(self._root)
        )
        if not isinstance(hit, str):  # None (not cached) or the _CACHED_NO_EXIST sentinel
            return None
        resolved = Path(hit).resolve()
        try:
            validate_path_within(resolved, self._root)
        except ValueError:
            return None
        return resolved

    def _split_shards_present(self, hf_repo: str, gguf_filename: str) -> bool:
        """True unless *gguf_filename* is a split GGUF missing one of its shards.

        A single-file GGUF is always present here. For a split set
        (``<base>-0000N-of-0000M.gguf``) every shard must be cached, since
        llama.cpp loads the whole set from the first shard but needs them all.
        """
        from lilbee.catalog.download import split_shard_filenames  # deferred: catalog is heavy

        shards = split_shard_filenames(gguf_filename)
        if len(shards) == 1:
            return True
        return all(self._find_cached_gguf(hf_repo, shard) is not None for shard in shards)

    def _reregister_from_cache(self, hf_repo: str, gguf_filename: str, blob_path: Path) -> None:
        """Write a fresh manifest for a model just recovered from the HF cache.

        ``list_installed`` only walks ``manifests/``, so a cache-recovered model
        is resolvable but otherwise invisible (``lilbee model list``, the TUI
        catalog, the pull command's "already installed" check) until a manifest
        exists. The ``task`` comes from the featured catalog; for a non-catalog
        ref it's unknown, so the rewrite is skipped. Best-effort: a read-only
        models dir or a write race must not break the resolve that succeeded.
        """
        from datetime import UTC, datetime

        ref = format_native_gguf_ref(hf_repo, gguf_filename)
        try:
            from lilbee.catalog import (
                find_catalog_entry,
            )  # deferred: lilbee.catalog is a heavy import

            entry = find_catalog_entry(ref)
            if entry is None:
                return
            self._write_manifest(
                ModelManifest(
                    hf_repo=hf_repo,
                    gguf_filename=gguf_filename,
                    size_bytes=blob_path.stat().st_size,
                    task=entry.task,
                    downloaded_at=datetime.now(UTC).isoformat(),
                    blob=blob_path.name,  # the blob's filename is its sha in the HF cache
                )
            )
            log.info("Recovered manifest for %s from the model cache", ref)
        except Exception:  # cache-warming write; the resolve already returned a path
            log.debug("Could not re-register %s from the model cache", ref, exc_info=True)

    def is_installed(self, ref: str) -> bool:
        """Return True if a model is installed and its blob is present."""
        try:
            self.resolve(ref)
            return True
        except (KeyError, ValueError):
            return False

    def install(
        self,
        hf_repo: str,
        gguf_filename: str,
        source_path: Path,
        manifest: ModelManifest,
    ) -> Path:
        """Write a manifest, copying *source_path* into the HF cache if needed."""
        import shutil

        digest = _sha256_file(source_path)
        cache_path = self._repo_cache_dir(hf_repo)
        blobs_dir = cache_path / "blobs"
        blob_path = blobs_dir / digest
        if not blob_path.exists():
            blobs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, blob_path)

        updated = ModelManifest(
            hf_repo=hf_repo,
            gguf_filename=gguf_filename,
            size_bytes=manifest.size_bytes,
            task=manifest.task,
            downloaded_at=manifest.downloaded_at,
            blob=digest,
        )
        self._write_manifest(updated)
        return blob_path

    def remove(self, ref: str) -> bool:
        """Remove a manifest and its backing blob.

        The blob is shared via SHA-256 digest, so it only goes away
        when no other installed manifest references the same digest.
        Empty cache directories (``blobs/``, the per-repo ``models--``
        folder, and the per-repo manifest folder) are pruned so a
        deleted model leaves no orphan bytes behind.
        """
        try:
            hf_repo, gguf_filename = parse_hf_ref(ref)
        except ValueError:
            return False
        manifest = self._read_manifest(hf_repo, gguf_filename)
        if manifest is None:
            return False
        manifest_path = self._manifest_path(hf_repo, gguf_filename)
        manifest_path.unlink()
        repo_dir = manifest_path.parent
        if repo_dir.exists() and not any(repo_dir.iterdir()):
            repo_dir.rmdir()
        if manifest.blob is not None:
            self._gc_blob(manifest.hf_repo, manifest.blob)
        log.info("Removed model %s", ref)
        return True

    def _gc_blob(self, hf_repo: str, digest: str) -> None:
        """Drop blob bytes and HuggingFace cache cruft now that *digest*
        and possibly the whole repo are unused.

        When the per-repo ``models--<repo>/`` directory has no installed
        manifests left, the whole directory is wiped so HF's ``refs/``,
        ``snapshots/``, and stale ``blobs/`` all go with it. Otherwise
        only the specific blob file is removed when no remaining
        manifest still references its digest.
        """
        cache_path = self._repo_cache_dir(hf_repo)
        try:
            validate_path_within(cache_path, self._root)
        except ValueError:
            log.warning("Refusing to remove cache outside models_dir: %s", cache_path)
            return
        siblings = [m for m in self.list_installed() if m.hf_repo == hf_repo]
        if not siblings:
            if cache_path.exists():
                shutil.rmtree(cache_path)
            return
        if any(m.blob == digest for m in siblings):
            return
        blob_file = cache_path / "blobs" / digest
        if blob_file.exists():
            blob_file.unlink()

    def list_installed(self) -> list[ModelManifest]:
        """Return manifests for models whose blob is fully present on disk.

        A manifest with a null blob field or a missing blob file is the
        residue of a canceled or partial download. Surfacing it would
        let the picker offer an unusable selection, so the read filter
        lives here at the source instead of in every UI caller.
        """
        manifests: list[ModelManifest] = []
        if not self._manifests_dir.exists():
            return manifests
        for repo_dir in sorted(self._manifests_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            for tag_file in sorted(repo_dir.glob("*.gguf.json")):
                manifest = self._load_manifest_file(tag_file)
                if manifest is not None and self._blob_present(manifest):
                    manifests.append(manifest)
        return manifests

    def _blob_present(self, manifest: ModelManifest) -> bool:
        """True iff *manifest* points at an existing blob file."""
        if manifest.blob is None:
            return False
        blob_file = self._repo_cache_dir(manifest.hf_repo) / "blobs" / manifest.blob
        return blob_file.exists()

    def get_manifest(self, ref: str) -> ModelManifest | None:
        """Return the manifest for *ref* or None if not installed."""
        try:
            hf_repo, gguf_filename = parse_hf_ref(ref)
        except ValueError:
            return None
        return self._read_manifest(hf_repo, gguf_filename)

    def _manifest_path(self, hf_repo: str, gguf_filename: str) -> Path:
        repo = _validate_hf_repo(hf_repo)
        filename = _validate_gguf_filename(gguf_filename)
        path = self._manifests_dir / repo_to_dir(repo) / f"{filename}.json"
        validate_path_within(path, self._manifests_dir)
        return path

    def _read_manifest(self, hf_repo: str, gguf_filename: str) -> ModelManifest | None:
        return self._load_manifest_file(self._manifest_path(hf_repo, gguf_filename))

    def _write_manifest(self, manifest: ModelManifest) -> None:
        path = self._manifest_path(manifest.hf_repo, manifest.gguf_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(asdict(manifest), indent=2)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".tmp", mode="w", delete=False
            ) as tmp:
                tmp_path = tmp.name
                tmp.write(data)
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)
            raise

    def _load_manifest_file(self, path: Path) -> ModelManifest | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ModelManifest(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            log.warning("Corrupt manifest: %s", path)
            return None


def register_downloaded_model(entry: CatalogModel, file_path: Path) -> None:
    """Write a registry manifest for a freshly downloaded GGUF.

    A failed manifest write is logged, not raised, when the GGUF is still in the
    HF cache (``resolve`` recovers from it); if it isn't, the download itself is
    broken and the failure propagates so the caller reports it.
    """
    from datetime import UTC, datetime

    registry = ModelRegistry(cfg.models_dir)
    manifest = ModelManifest(
        hf_repo=entry.hf_repo,
        gguf_filename=file_path.name,
        size_bytes=file_path.stat().st_size,
        task=entry.task,
        downloaded_at=datetime.now(UTC).isoformat(),
    )
    try:
        registry.install(entry.hf_repo, file_path.name, file_path, manifest)
        log.info("Registered %s/%s in manifest", entry.hf_repo, file_path.name)
    except Exception:
        ref = format_native_gguf_ref(entry.hf_repo, file_path.name)
        if not registry.is_installed(ref):
            raise
        log.warning(
            "Manifest write failed for %s; recovered via the model cache", ref, exc_info=True
        )
