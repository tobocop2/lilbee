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
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from lilbee.core.security import validate_path_within

log = logging.getLogger(__name__)

_HASH_ALGORITHM = "sha256"
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
    task: str  # use lilbee.models.ModelTask values
    downloaded_at: str  # ISO 8601
    blob: str = ""  # SHA-256 hex of the blob in the HF cache

    @property
    def ref(self) -> str:
        return f"{self.hf_repo}/{self.gguf_filename}"


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

    def resolve(self, ref: str) -> Path:
        """Return the blob path for *ref*; ``KeyError`` if not installed."""
        hf_repo, gguf_filename = parse_hf_ref(ref)
        manifest = self._read_manifest(hf_repo, gguf_filename)
        if manifest is None:
            raise KeyError(f"Model {ref} not installed")
        cache_path = self._root / f"models--{repo_to_dir(manifest.hf_repo)}"
        if not cache_path.exists():
            raise KeyError(f"Cache folder missing for {ref}: {cache_path.name}")
        blob_file = cache_path / "blobs" / manifest.blob
        if not blob_file.exists():
            raise KeyError(f"Blob file missing for {ref}: {manifest.blob}")
        return blob_file

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
        cache_path = self._root / f"models--{repo_to_dir(hf_repo)}"
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
        """Remove a manifest. Does not delete the cached blob."""
        try:
            hf_repo, gguf_filename = parse_hf_ref(ref)
        except ValueError:
            return False
        manifest_path = self._manifest_path(hf_repo, gguf_filename)
        if not manifest_path.exists():
            return False
        manifest_path.unlink()
        repo_dir = manifest_path.parent
        if repo_dir.exists() and not any(repo_dir.iterdir()):
            repo_dir.rmdir()
        log.info("Removed manifest for %s (cache file untouched)", ref)
        return True

    def list_installed(self) -> list[ModelManifest]:
        """Return manifests for all installed models."""
        manifests: list[ModelManifest] = []
        if not self._manifests_dir.exists():
            return manifests
        for repo_dir in sorted(self._manifests_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            for tag_file in sorted(repo_dir.glob("*.gguf.json")):
                manifest = self._load_manifest_file(tag_file)
                if manifest is not None:
                    manifests.append(manifest)
        return manifests

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
