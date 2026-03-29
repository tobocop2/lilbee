"""Model registry -- content-addressable storage with manifest-based resolution.

Inspired by Ollama's model management. Model names (e.g., "nomic-embed-text")
resolve through manifests to content-addressed .gguf blobs.

Storage layout::

    models_dir/
    +-- manifests/
    |   +-- nomic-embed-text/
    |   |   +-- latest.json
    |   +-- qwen3/
    |       +-- latest.json
    |       +-- 0.6b.json
    +-- blobs/
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
    """Content-addressable model storage with manifest resolution."""

    def __init__(self, models_dir: Path) -> None:
        self._root = models_dir
        self._manifests_dir = models_dir / "manifests"
        self._blobs_dir = models_dir / "blobs"

    def resolve(self, ref: str | ModelRef) -> Path:
        """Resolve model name to blob file path.

        Raises ``KeyError`` if the model is not installed.
        """
        r = _coerce_ref(ref)
        manifest = self._read_manifest(r)
        if manifest is None:
            raise KeyError(f"Model {r} not installed")
        blob_path = self._blobs_dir / f"{_HASH_ALGORITHM}-{manifest.blob}"
        if not blob_path.exists():
            raise KeyError(f"Blob missing for {r}: {blob_path.name}")
        return blob_path

    def is_installed(self, ref: str | ModelRef) -> bool:
        """Check if a model is installed (manifest exists and blob is present)."""
        try:
            self.resolve(ref)
            return True
        except (KeyError, ValueError):
            return False

    def install(self, ref: ModelRef, source_path: Path, manifest: ModelManifest) -> Path:
        """Move a downloaded file into blob storage and write manifest.

        Returns the blob path.
        """
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        digest = _sha256_file(source_path)
        blob_path = self._blobs_dir / f"{_HASH_ALGORITHM}-{digest}"
        os.replace(source_path, blob_path)
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
        """Remove a model manifest. Delete blob if no other manifests reference it."""
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
        # Garbage-collect blob if unreferenced
        if not self._blob_referenced(manifest.blob):
            blob_path = self._blobs_dir / f"{_HASH_ALGORITHM}-{manifest.blob}"
            blob_path.unlink(missing_ok=True)
            log.info("Deleted unreferenced blob %s", blob_path.name)
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

        On first run, finds existing .gguf files and creates manifests.
        Uses catalog to match display names. Returns the number of models migrated.
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
            manifest = ModelManifest(
                name=name,
                tag="latest",
                size_bytes=path.stat().st_size,
                task=task,
                source_repo=repo,
                source_filename=path.name,
                downloaded_at=datetime.now(tz=UTC).isoformat(),
            )
            self.install(ref, path, manifest)
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
