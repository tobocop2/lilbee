"""Manifest store keyed by ``(hf_repo, gguf_filename)`` over the HF cache.

Canonical ref: ``<hf_repo>/<gguf_filename>``. Two quants of the same
repo are two distinct installations. Manifests live at
``manifests/<repo--repo>/<filename>.json``; blobs at
``models--<repo--repo>/blobs/<sha>``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.catalog.download import split_shard_filenames
from lilbee.catalog.query import find_catalog_entry, reclassify_by_name
from lilbee.catalog.refs import (
    NATIVE_GGUF_REF_MIN_SLASHES,
    format_native_gguf_ref,
    is_bare_hf_repo,
)
from lilbee.catalog.types import ModelTask
from lilbee.core.config.model import cfg
from lilbee.core.security import validate_path_within

if TYPE_CHECKING:
    from lilbee.catalog.models import CatalogModel

log = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 8192  # bytes read per iteration when hashing
_REPO_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")
# A GGUF filename, optionally under repo subdirectories (unsloth stores quants
# in e.g. ``Q4_K_M/Model-...-00001-of-00003.gguf``). Path separators are allowed;
# ``..`` and absolute paths are rejected in the validator to stay inside the repo.
_FILENAME_RE = re.compile(r"^[a-zA-Z0-9._/-]+\.gguf$")

REPO_DIR_SEPARATOR = "--"


def _validate_hf_repo(hf_repo: str) -> str:
    """Validate that a HuggingFace repo id has the form ``org/name``."""
    if not hf_repo or not _REPO_SEGMENT_RE.match(hf_repo) or ".." in hf_repo:
        raise ValueError(f"Invalid hf_repo: {hf_repo!r}")
    return hf_repo


def _validate_gguf_filename(filename: str) -> str:
    """Validate a ``.gguf`` filename, allowing repo subdirectories but no traversal."""
    if (
        not filename
        or not _FILENAME_RE.match(filename)
        or ".." in filename
        or filename.startswith("/")
    ):
        raise ValueError(f"Invalid gguf_filename: {filename!r}")
    return filename


_REF_SHAPE_HINT = "Use '<org>/<repo>/<filename>.gguf'."


def parse_hf_ref(ref: str) -> tuple[str, str]:
    """Split ``<org>/<repo>/<file>.gguf`` into ``(hf_repo, gguf_filename)``.

    The repo is always the first two segments (``<org>/<repo>``); everything
    after is the filename, which may include repo subdirectories (unsloth stores
    quants under e.g. ``Q4_K_M/Model-...-00001-of-00003.gguf``).
    """
    if not ref.endswith(".gguf") or ref.count("/") < NATIVE_GGUF_REF_MIN_SLASHES:
        raise ValueError(f"Model ref {ref!r} is not a HuggingFace ref. {_REF_SHAPE_HINT}")
    parts = ref.split("/")
    hf_repo = "/".join(parts[:NATIVE_GGUF_REF_MIN_SLASHES])
    gguf_filename = "/".join(parts[NATIVE_GGUF_REF_MIN_SLASHES:])
    return _validate_hf_repo(hf_repo), _validate_gguf_filename(gguf_filename)


def repo_to_dir(hf_repo: str) -> str:
    """Encode an HF repo for use as a directory name (HF cache convention)."""
    return hf_repo.replace("/", REPO_DIR_SEPARATOR)


@dataclass
class ModelManifest:
    """One installed model's metadata. Identity: ``(hf_repo, gguf_filename)``."""

    hf_repo: str
    gguf_filename: str
    size_bytes: int  # primary (first-shard) blob size; validated against the blob on disk
    task: ModelTask
    downloaded_at: str  # ISO 8601
    blob: str | None = None  # SHA-256 hex of the blob in the HF cache; None pre-install
    # A split GGUF has further shard blobs beyond ``blob``. ``total_size_bytes``
    # is the sum across every shard (None = single file, use ``size_bytes``);
    # ``shard_blobs`` are the non-primary shard digests, so removal frees them all.
    total_size_bytes: int | None = None
    shard_blobs: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return format_native_gguf_ref(self.hf_repo, self.gguf_filename)

    @property
    def disk_size_bytes(self) -> int:
        """Total bytes this model occupies on disk, across all shards."""
        return self.total_size_bytes if self.total_size_bytes is not None else self.size_bytes


def _copy_atomic(source_path: Path, blob_path: Path) -> None:
    """Copy *source_path* to *blob_path* via a temp file + atomic rename.

    A crash mid-copy leaves only the temp file, never a partial blob at
    the final digest path that callers would treat as complete.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(blob_path.parent), suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as dst, source_path.open("rb") as src:
            shutil.copyfileobj(src, dst)
        os.replace(tmp_path, blob_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _blob_size_matches(blob_file: Path, expected_size: int) -> bool:
    """True iff *blob_file* exists and its byte size equals *expected_size*.

    A blob shorter than the manifest's recorded size is a truncated /
    interrupted download and must not count as installed.
    """
    try:
        return blob_file.stat().st_size == expected_size
    except OSError:
        return False


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


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _blob_digest(source_path: Path) -> str:
    """Digest for *source_path*, reusing the HF-cache blob name when possible.

    huggingface_hub names cache blobs by their sha256, so a snapshot path that
    resolves into a ``blobs/`` dir already carries its digest. Re-hashing a
    multi-GB GGUF only to recompute that name is slow and, on network volumes,
    I/O-fragile enough to fail registration outright. Plain files still hash.
    """
    real = source_path.resolve()
    if real.parent.name == "blobs" and _SHA256_HEX.fullmatch(real.name):
        return real.name
    return _sha256_file(source_path)


class ModelRegistry:
    """Read/write manifests and resolve refs to blobs in the HF cache."""

    def __init__(self, models_dir: Path) -> None:
        self._root = models_dir
        self._manifests_dir = models_dir / "manifests"

    def _repo_cache_dir(self, hf_repo: str) -> Path:
        """The HuggingFace cache directory for *hf_repo* under this registry root."""
        return self._root / f"models--{repo_to_dir(hf_repo)}"

    def resolve(self, ref: str) -> Path:
        """Return the loadable GGUF path for *ref*; ``KeyError`` if not installed.

        A single-file GGUF resolves to its content-hashed blob; a split GGUF to
        its first shard's snapshot symlink (so llama.cpp finds the sibling shards).

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
        if is_bare_hf_repo(ref):
            return self._resolve_repo_only(_validate_hf_repo(ref))
        hf_repo, gguf_filename = parse_hf_ref(ref)
        shards = split_shard_filenames(gguf_filename)
        if len(shards) > 1:
            return self._resolve_split(ref, hf_repo, shards)
        manifest = self._read_manifest(hf_repo, gguf_filename)
        if manifest is not None and manifest.blob is not None:
            blob_file = self._repo_cache_dir(manifest.hf_repo) / "blobs" / manifest.blob
            if _blob_size_matches(blob_file, manifest.size_bytes):
                return blob_file
        recovered = self._find_cached_gguf(hf_repo, gguf_filename)
        if recovered is not None:
            self._reregister_from_cache(hf_repo, gguf_filename, recovered)
            return recovered
        if manifest is None:
            raise KeyError(f"Model {ref} not installed")
        # Manifest present but neither it nor the cache yields a blob; keep the
        # specific diagnostic so a corrupted cache stays debuggable.
        cache_path = self._repo_cache_dir(manifest.hf_repo)
        if not cache_path.exists():
            raise KeyError(f"Cache folder missing for {ref}: {cache_path.name}")
        if manifest.blob is None:
            raise KeyError(f"Manifest for {ref} has no blob hash; install incomplete")
        blob_file = cache_path / "blobs" / manifest.blob
        if blob_file.exists():
            raise KeyError(
                f"Blob for {ref} is truncated: {blob_file.stat().st_size} of "
                f"{manifest.size_bytes} bytes; re-download required"
            )
        raise KeyError(f"Blob file missing for {ref}: {manifest.blob}")

    def _resolve_split(self, ref: str, hf_repo: str, shards: list[str]) -> Path:
        """Resolve a split GGUF to its first shard's snapshot symlink.

        llama.cpp loads the whole set from the first shard, locating the siblings
        by filename next to it. Only the snapshot dir co-locates the shards under
        their real names (the blobs dir names them by hash), so hand back the
        symlink, not the blob. Every shard must be present first: the first shard
        alone used to read as installed, registering an unloadable model that a
        re-pull then skipped.
        """
        if not self._split_shards_present(hf_repo, shards[0]):
            raise KeyError(f"Split GGUF {ref} is missing shards; re-pull to fetch the full set")
        first_shard = self._snapshot_gguf_path(hf_repo, shards[0])
        if first_shard is None:
            raise KeyError(f"Model {ref} not installed")
        if self._read_manifest(hf_repo, shards[0]) is None:
            # Same cache recovery as the single-file path; resolve the symlink so
            # the manifest records the content-hashed blob, not the link, and pass
            # the snapshot path so the shard accounting is recovered too.
            self._reregister_from_cache(
                hf_repo, shards[0], first_shard.resolve(), snapshot_path=first_shard
            )
        return first_shard

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
            # rglob, like list_installed: a quant-subdir ref writes its manifest one
            # directory deeper, so a non-recursive scan would miss it and fall through
            # to the slower huggingface_hub cache recovery.
            for mf in sorted(manifest_dir.rglob("*.gguf.json")):
                manifest = self._load_manifest_file(mf)
                if manifest is None or manifest.blob is None:
                    continue
                blob = self._repo_cache_dir(hf_repo) / "blobs" / manifest.blob
                if blob.exists():
                    return blob
        for filename in sorted(self._cached_gguf_names(hf_repo)):
            shards = split_shard_filenames(filename)
            if len(shards) > 1:
                # A split set in the cache: skip its non-first shards and resolve
                # the whole set from shard 1 so we hand back the snapshot symlink
                # (siblings co-located, loadable) with shard accounting, not
                # shard 1's blob as an unloadable single file.
                if filename != shards[0]:
                    continue
                with contextlib.suppress(KeyError):
                    return self._resolve_split(
                        format_native_gguf_ref(hf_repo, filename), hf_repo, shards
                    )
                continue
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

    def _snapshot_gguf_path(self, hf_repo: str, gguf_filename: str) -> Path | None:
        """Return the snapshot *symlink* path for a cached GGUF, or None.

        Returns the symlink, not the blob, so a split GGUF loads from a dir where
        its sibling shards are co-located under their real names.
        """
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(
            repo_id=hf_repo, filename=gguf_filename, cache_dir=str(self._root)
        )
        candidate: Path | None = None
        if isinstance(hit, str):  # exact repo-relative match
            candidate = Path(hit)
        else:  # None or the _CACHED_NO_EXIST sentinel: locate the basename instead
            snapshots = self._repo_cache_dir(hf_repo) / "snapshots"
            if snapshots.is_dir():
                basename = Path(gguf_filename).name
                # Several cached revisions can hold the basename; prefer the most
                # recently materialized one over an arbitrary lexicographic pick.
                candidate = max(
                    snapshots.rglob(basename), key=lambda p: p.lstat().st_mtime, default=None
                )
        if candidate is None:
            return None
        try:
            validate_path_within(candidate.resolve(), self._root)
        except ValueError:
            return None
        return candidate

    def _find_cached_gguf(self, hf_repo: str, gguf_filename: str) -> Path | None:
        """Return the cached blob path for ``hf_repo``/``gguf_filename``, or None.

        Locates the snapshot symlink (subdir-aware) and resolves it to its blob,
        bounded to the cache directory.
        """
        symlink = self._snapshot_gguf_path(hf_repo, gguf_filename)
        return symlink.resolve() if symlink is not None else None

    def _split_shards_present(self, hf_repo: str, gguf_filename: str) -> bool:
        """True unless *gguf_filename* is a split GGUF missing one of its shards.

        A single-file GGUF is always present here. For a split set
        (``<base>-0000N-of-0000M.gguf``) every shard must be cached, since
        llama.cpp loads the whole set from the first shard but needs them all.
        """
        shards = split_shard_filenames(gguf_filename)
        if len(shards) == 1:
            return True
        return all(self._find_cached_gguf(hf_repo, shard) is not None for shard in shards)

    def shard_paths(self, ref: str) -> list[Path]:
        """On-disk paths of *ref*'s GGUF shards that exist next to its resolved path.

        A split GGUF resolves to its first shard's snapshot symlink with the
        siblings co-located, so every shard is returned; a single-file GGUF
        resolves to its content-hashed blob, where no sibling exists under the
        real filename. Raises ``KeyError`` / ``ValueError`` like :meth:`resolve`.
        """
        first = self.resolve(ref)
        _repo, filename = parse_hf_ref(ref)
        candidates = (
            first.parent / Path(shard).name for shard in split_shard_filenames(Path(filename).name)
        )
        return [path for path in candidates if path.exists()]

    def _reregister_from_cache(
        self,
        hf_repo: str,
        gguf_filename: str,
        blob_path: Path,
        snapshot_path: Path | None = None,
    ) -> None:
        """Best-effort manifest write for a cache-recovered model so listings see it.

        *snapshot_path* is the first shard's snapshot path (siblings co-located);
        when given, the split-shard accounting is recovered too, so a cache-only
        split GGUF still frees every shard and reports its full size.
        """
        ref = format_native_gguf_ref(hf_repo, gguf_filename)
        try:
            entry = find_catalog_entry(ref)
            if entry is not None:
                task = entry.task
            else:
                task = ModelTask(reclassify_by_name(ref, ModelTask.CHAT))
            total_size, shard_blobs = (
                _shard_accounting(snapshot_path) if snapshot_path is not None else (None, [])
            )
            self._write_manifest(
                ModelManifest(
                    hf_repo=hf_repo,
                    gguf_filename=gguf_filename,
                    size_bytes=blob_path.stat().st_size,
                    task=task,
                    downloaded_at=datetime.now(UTC).isoformat(),
                    blob=blob_path.name,  # the blob's filename is its sha in the HF cache
                    total_size_bytes=total_size,
                    shard_blobs=shard_blobs,
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
        digest = _blob_digest(source_path)
        cache_path = self._repo_cache_dir(hf_repo)
        blobs_dir = cache_path / "blobs"
        blob_path = blobs_dir / digest
        if not blob_path.exists():
            blobs_dir.mkdir(parents=True, exist_ok=True)
            _copy_atomic(source_path, blob_path)

        updated = ModelManifest(
            hf_repo=hf_repo,
            gguf_filename=gguf_filename,
            # Record the size install actually wrote, not the caller's claim,
            # so the on-disk size check has a trustworthy reference.
            size_bytes=source_path.stat().st_size,
            task=manifest.task,
            downloaded_at=manifest.downloaded_at,
            blob=digest,
            # Carry the split-shard accounting through unchanged (computed by the
            # caller from the full shard set); install only rewrites the primary.
            total_size_bytes=manifest.total_size_bytes,
            shard_blobs=manifest.shard_blobs,
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
        # Manifests written before shard accounting existed have no shard_blobs, so
        # recover them from the cache *before* unlinking (resolve needs the manifest).
        shard_blobs = manifest.shard_blobs or self._recover_legacy_shard_blobs(ref)
        manifest_path = self._manifest_path(hf_repo, gguf_filename)
        manifest_path.unlink()
        repo_dir = manifest_path.parent
        if repo_dir.exists() and not any(repo_dir.iterdir()):
            repo_dir.rmdir()
        # Free the primary blob and every extra shard blob; a split GGUF has more
        # than one, and leaving the others orphans them when a sibling quant keeps
        # the repo cache dir alive. The surviving manifests for this repo are read
        # once here rather than per digest (list_installed walks the whole tree).
        siblings = [m for m in self.list_installed() if m.hf_repo == manifest.hf_repo]
        for digest in [manifest.blob, *shard_blobs]:
            if digest is not None:
                self._gc_blob(manifest.hf_repo, digest, siblings=siblings)
        log.info("Removed model %s", ref)
        return True

    def _recover_legacy_shard_blobs(self, ref: str) -> list[str]:
        """Extra shard blob digests for a pre-accounting split GGUF, best-effort.

        Older manifests recorded only the first shard, so removing them would
        orphan the rest. Derive the sibling shards from the cache; empty on any
        failure or for a single-file model, so removal never breaks.
        """
        with contextlib.suppress(Exception):
            shards = self.shard_paths(ref)
            return [_blob_digest(path) for path in shards[1:]]
        return []

    def _gc_blob(
        self, hf_repo: str, digest: str, *, siblings: list[ModelManifest] | None = None
    ) -> None:
        """Drop blob bytes and HuggingFace cache cruft now that *digest*
        and possibly the whole repo are unused.

        When the per-repo ``models--<repo>/`` directory has no installed
        manifests left, the whole directory is wiped so HF's ``refs/``,
        ``snapshots/``, and stale ``blobs/`` all go with it. Otherwise
        only the specific blob file is removed when no remaining
        manifest still references its digest.

        ``siblings`` is the surviving-manifest list for *hf_repo*; callers
        freeing several blobs at once pass it in so the manifest tree is walked
        once instead of per digest. Defaults to reading it when omitted.
        """
        cache_path = self._repo_cache_dir(hf_repo)
        try:
            validate_path_within(cache_path, self._root)
        except ValueError:
            log.warning("Refusing to remove cache outside models_dir: %s", cache_path)
            return
        if siblings is None:
            siblings = [m for m in self.list_installed() if m.hf_repo == hf_repo]
        if not siblings:
            if cache_path.exists():
                shutil.rmtree(cache_path)
            return
        if any(digest == m.blob or digest in m.shard_blobs for m in siblings):
            return
        blob_file = cache_path / "blobs" / digest
        try:
            validate_path_within(blob_file, self._root)
        except ValueError:
            log.warning("Refusing to remove blob outside models_dir: %s", blob_file)
            return
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
            # rglob, not glob: a quant-subdir ref (unsloth stores quants under e.g.
            # Q4_K_S/<model>.gguf) writes its manifest one level deeper, so a
            # non-recursive scan omitted it from `model list` and /v1/models, and
            # opencode silently fell back to its own provider.
            for tag_file in sorted(repo_dir.rglob("*.gguf.json")):
                manifest = self._load_manifest_file(tag_file)
                if manifest is not None and self._blob_present(manifest):
                    manifests.append(manifest)
        return manifests

    def _blob_present(self, manifest: ModelManifest) -> bool:
        """True iff *manifest* points at a blob whose on-disk size matches."""
        if manifest.blob is None:
            return False
        blob_file = self._repo_cache_dir(manifest.hf_repo) / "blobs" / manifest.blob
        return _blob_size_matches(blob_file, manifest.size_bytes)

    def get_manifest(self, ref: str) -> ModelManifest | None:
        """Return the manifest for *ref* or None if not installed."""
        try:
            hf_repo, gguf_filename = parse_hf_ref(ref)
        except ValueError:
            return None
        return self._read_manifest(hf_repo, gguf_filename)

    def installed_ref_for_repo(self, hf_repo: str) -> str | None:
        """Full ``<repo>/<file>.gguf`` ref of an installed quant of *hf_repo*, or None.

        Alphabetical-first when several quants are installed, matching
        ``_resolve_repo_only``'s determinism.
        """
        refs = sorted(m.ref for m in self.list_installed() if m.hf_repo == hf_repo)
        return refs[0] if refs else None

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


_HF_SNAPSHOTS_DIR = "snapshots"


def _repo_relative_gguf_name(file_path: Path) -> str:
    """Recover the repo-relative GGUF filename, keeping any subdir prefix.

    HF caches a file at ``models--<repo>/snapshots/<rev>/[<subdir>/]<name>``. A
    subdir-quant giant (unsloth stores quants under e.g. ``Q4_K_M/``) must
    register under that subdir-relative name so its manifest key round-trips with
    the ref; ``file_path.name`` alone would drop the subdir. Falls back to the
    basename when the path is not under a snapshot revision dir.
    """
    parts = file_path.parts
    if _HF_SNAPSHOTS_DIR not in parts:
        return file_path.name
    rev_index = parts.index(_HF_SNAPSHOTS_DIR) + 1
    relative_parts = parts[rev_index + 1 :]
    return "/".join(relative_parts) if relative_parts else file_path.name


def _shard_accounting(first_shard_path: Path) -> tuple[int | None, list[str]]:
    """Total on-disk size and non-primary shard blob digests for a split GGUF.

    ``(None, [])`` for a single-file model. For a split GGUF, the sibling shards
    live next to the first shard; ``_blob_digest`` yields each shard's blob digest
    (the HF cache blob name, or a content hash in copy/non-symlink mode), so the
    shards are summed and the digests of shards 2..N collected for removal-time
    garbage collection.
    """
    shard_names = split_shard_filenames(first_shard_path.name)
    if len(shard_names) <= 1:
        return None, []
    total = 0
    shard_blobs: list[str] = []
    for index, name in enumerate(shard_names):
        shard_path = first_shard_path.with_name(name)
        if not shard_path.exists():
            continue
        total += shard_path.stat().st_size
        if index > 0:  # the primary blob is tracked separately as manifest.blob
            shard_blobs.append(_blob_digest(shard_path))
    return total, shard_blobs


def register_downloaded_model(entry: CatalogModel, file_path: Path) -> None:
    """Write a registry manifest for a freshly downloaded GGUF.

    A failed manifest write is logged, not raised, when the GGUF is still in the
    HF cache (``resolve`` recovers from it); if it isn't, the download itself is
    broken and the failure propagates so the caller reports it.
    """
    registry = ModelRegistry(cfg.models_dir)
    gguf_filename = _repo_relative_gguf_name(file_path)
    total_size, shard_blobs = _shard_accounting(file_path)
    manifest = ModelManifest(
        hf_repo=entry.hf_repo,
        gguf_filename=gguf_filename,
        size_bytes=file_path.stat().st_size,
        task=entry.task,
        downloaded_at=datetime.now(UTC).isoformat(),
        total_size_bytes=total_size,
        shard_blobs=shard_blobs,
    )
    try:
        registry.install(entry.hf_repo, gguf_filename, file_path, manifest)
        log.info("Registered %s/%s in manifest", entry.hf_repo, gguf_filename)
    except Exception:
        ref = format_native_gguf_ref(entry.hf_repo, gguf_filename)
        if not registry.is_installed(ref):
            raise
        log.warning(
            "Manifest write failed for %s; recovered via the model cache", ref, exc_info=True
        )
