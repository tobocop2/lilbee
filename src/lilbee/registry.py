"""Model registry -- manifest-based resolution over HuggingFace cache.

Inspired by Ollama's model management. Model names (e.g., "nomic-embed-text")
resolve through manifests to files in the HF cache.

Storage layout::

    models_dir/
    +-- manifests/
    |   +-- nomic-embed-text/
    |   |   +-- latest.json
    |   +-- qwen3/
    |       +-- latest.json
    |       +-- 0.6b.json
    +-- models--ORG--NAME/blobs/
        +-- sha256-abc123...
        +-- sha256-def456...
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lilbee.security import validate_path_within

log = logging.getLogger(__name__)

_HASH_ALGORITHM = "sha256"
_HASH_CHUNK_SIZE = 8192  # bytes read per iteration when hashing
_REF_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9 ._/-]+$")


def _validate_ref_segment(segment: str, label: str) -> str:
    """Validate that a model ref segment contains only safe characters.

    Allows alphanumeric, hyphens, dots, underscores, slashes (namespaced models),
    and spaces (display names). Rejects path traversal sequences.
    """
    if not segment or not _REF_SEGMENT_RE.match(segment):
        raise ValueError(f"Invalid model {label}: {segment!r}")
    if ".." in segment:
        raise ValueError(f"Invalid model {label}: {segment!r}")
    return segment


@dataclass(frozen=True)
class ModelRef:
    """Parsed model reference like 'qwen3:8b' or 'nomic-embed-text:latest'."""

    name: str
    tag: str = "latest"

    @classmethod
    def parse(cls, s: str) -> ModelRef:
        """Parse 'name:tag' string. Default tag is 'latest'."""
        if ":" in s:
            name, tag = s.rsplit(":", 1)
            return cls(
                name=_validate_ref_segment(name, "name"),
                tag=_validate_ref_segment(tag, "tag"),
            )
        return cls(name=_validate_ref_segment(s, "name"))

    def __str__(self) -> str:
        return f"{self.name}:{self.tag}"


@dataclass
class ModelManifest:
    """Manifest for an installed model version."""

    name: str
    tag: str
    size_bytes: int
    task: str  # "chat", "embedding", or "vision"
    source_repo: str  # HuggingFace repo
    source_filename: str  # original .gguf filename
    downloaded_at: str  # ISO 8601 timestamp
    blob: str = ""  # SHA-256 hash (filename in blobs/), set by install()


def _coerce_ref(ref: str | ModelRef) -> ModelRef:
    """Normalize a string or ModelRef into a ModelRef."""
    if isinstance(ref, str):
        return ModelRef.parse(ref)
    return ref


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
    """Content-addressable model storage with manifest resolution.

    Now references HF cache directly instead of copying files.
    """

    def __init__(self, models_dir: Path) -> None:
        self._root = models_dir
        self._manifests_dir = models_dir / "manifests"

    def resolve(self, ref: str | ModelRef) -> Path:
        """Resolve model name to file path in HF cache.

        Raises ``KeyError`` if the model is not installed.
        """
        r = _coerce_ref(ref)
        manifest = self._read_manifest(r)
        if manifest is None:
            raise KeyError(f"Model {r} not installed")

        # Find the file in HF cache structure: models--ORG--NAME/blobs/*
        cache_path = self._root / f"models--{manifest.source_repo.replace('/', '--')}"
        if not cache_path.exists():
            raise KeyError(f"Cache folder missing for {r}: {cache_path.name}")

        # Find the blobs directory
        blobs_dir = cache_path / "blobs"
        if not blobs_dir.exists():
            raise KeyError(f"Blobs directory missing for {r}")

        # Look for file matching the blob hash
        blob_file = blobs_dir / manifest.blob
        if not blob_file.exists():
            raise KeyError(f"Blob file missing for {r}: {manifest.blob}")

        return blob_file

    def is_installed(self, ref: str | ModelRef) -> bool:
        """Check if a model is installed (manifest exists and file is in cache)."""
        try:
            self.resolve(ref)
            return True
        except (KeyError, ValueError):
            return False

    def install(self, ref: ModelRef, source_path: Path, manifest: ModelManifest) -> Path:
        """Write manifest for a model already in HF cache.

        The file is already in the cache dir (thanks to cache_dir setting).
        This just creates the manifest to track it.
        """
        # Compute hash to store in manifest
        digest = _sha256_file(source_path)

        # Resolve the cache path to get the blob filename
        cache_path = self._root / f"models--{manifest.source_repo.replace('/', '--')}"
        blob_path = cache_path / "blobs" / digest

        # Verify file exists in cache
        if not blob_path.exists():
            raise FileNotFoundError(f"Downloaded file not found in cache: {blob_path}")

        # Write manifest
        updated = ModelManifest(
            name=manifest.name,
            tag=manifest.tag,
            size_bytes=manifest.size_bytes,
            task=manifest.task,
            source_repo=manifest.source_repo,
            source_filename=manifest.source_filename,
            downloaded_at=manifest.downloaded_at,
            blob=digest,
        )
        self._write_manifest(ref, updated)
        return blob_path

    def remove(self, ref: str | ModelRef) -> bool:
        """Remove a model manifest. Does NOT delete cache files (managed by HF)."""
        try:
            r = _coerce_ref(ref)
            manifest = self._read_manifest(r)
        except ValueError:
            return False
        if manifest is None:
            return False
        manifest_path = self._manifest_path(r)
        manifest_path.unlink()
        # Remove empty name directory
        name_dir = manifest_path.parent
        if name_dir.exists() and not any(name_dir.iterdir()):
            name_dir.rmdir()
        log.info("Removed manifest for %s (cache file untouched)", r)
        return True

    def list_installed(self) -> list[ModelManifest]:
        """List all installed models."""
        manifests: list[ModelManifest] = []
        if not self._manifests_dir.exists():
            return manifests
        for name_dir in sorted(self._manifests_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            for tag_file in sorted(name_dir.glob("*.json")):
                manifest = self._load_manifest_file(tag_file)
                if manifest is not None:
                    manifests.append(manifest)
        return manifests

    def migrate_legacy(self) -> int:
        """Scan for .gguf files in root dir and register them.

        On first run, finds existing .gguf files, moves them to HF cache
        structure, and creates manifests. Returns the number of models migrated.
        """
        if not self._root.exists():
            return 0
        gguf_files = sorted(self._root.glob("*.gguf"))
        if not gguf_files:
            return 0
        count = 0
        for path in gguf_files:
            name, task, repo = _match_catalog_entry(path.name)
            ref = ModelRef.parse(name)
            digest = _sha256_file(path)

            if repo:
                cache_path = self._root / f"models--{repo.replace('/', '--')}"
            else:
                safe_name = name.replace(" ", "_").replace("/", "_")
                cache_path = self._root / f"models--{safe_name}"
            blobs_dir = cache_path / "blobs"
            blobs_dir.mkdir(parents=True, exist_ok=True)
            blob_path = blobs_dir / digest
            path.rename(blob_path)

            manifest = ModelManifest(
                name=name,
                tag="latest",
                size_bytes=blob_path.stat().st_size,
                task=task,
                source_repo=repo,
                source_filename=path.name,
                downloaded_at=datetime.now(tz=UTC).isoformat(),
                blob=digest,
            )
            self._write_manifest(ref, manifest)
            log.info("Migrated legacy model %s -> %s", path.name, ref)
            count += 1
        return count

    def _manifest_path(self, ref: ModelRef) -> Path:
        path = self._manifests_dir / ref.name / f"{ref.tag}.json"
        validate_path_within(path, self._manifests_dir)
        return path

    def _read_manifest(self, ref: ModelRef) -> ModelManifest | None:
        return self._load_manifest_file(self._manifest_path(ref))

    def _write_manifest(self, ref: ModelRef, manifest: ModelManifest) -> None:
        path = self._manifest_path(ref)
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

    def _blob_referenced(self, digest: str) -> bool:
        """Check if any manifest still references the given blob digest."""
        return any(manifest.blob == digest for manifest in self.list_installed())


def _match_catalog_entry(filename: str) -> tuple[str, str, str]:
    """Match a .gguf filename to a catalog entry, returning (name, task, repo).

    Falls back to deriving a name from the filename itself.
    """
    from lilbee.catalog import FEATURED_ALL

    filename_lower = filename.lower()
    for entry in FEATURED_ALL:
        # Match on repo name fragment or exact filename
        repo_stem = entry.hf_repo.split("/")[-1].lower()
        if filename_lower.startswith(repo_stem) or entry.gguf_filename == filename:
            return entry.name, entry.task, entry.hf_repo
    # Fallback: strip extension and quant suffix for a reasonable name
    stem = filename.rsplit(".", 1)[0]
    return stem, "chat", ""
