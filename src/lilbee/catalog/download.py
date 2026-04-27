"""GGUF download, mmproj resolution, registry registration."""

import fnmatch
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from lilbee.catalog.featured import _DEFAULT_MMPROJ_PATTERN, VISION_MMPROJ_FILES
from lilbee.catalog.hf_client import (
    _DEFAULT_TIMEOUT,
    ProgressCallback,
    _hf_headers,
    _hf_token,
    _ProgressTracker,
)
from lilbee.catalog.models import CatalogModel
from lilbee.modelhub.models import ModelTask
from lilbee.modelhub.registry import ModelManifest, ModelRegistry
from lilbee.runtime.cancellation import TaskCancelled

log = logging.getLogger(__name__)


def _cfg() -> Any:
    """Lazy accessor for the global ``cfg`` singleton (see circular-import note)."""
    # circular: lilbee.catalog -> lilbee.config via cfg
    from lilbee.core.config import cfg

    return cfg


class DownloadConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    repo_id: str
    filename: str
    token: str | None
    force_download: bool = False
    cache_dir: str | None = None
    tqdm_class: Any = None


def download_model(entry: CatalogModel, *, on_progress: ProgressCallback | None = None) -> Path:
    """Download a GGUF model from HuggingFace to cfg.models_dir.
    Uses huggingface_hub for resumable downloads, caching, and auth.
    The optional *on_progress(downloaded, total)* callback receives byte counts.
    For vision models, also downloads the mmproj (CLIP projection) file.

    Raises:
        PermissionError: gated repo requiring authentication
        RuntimeError: repo not found or download failure with details
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    _cfg().models_dir.mkdir(parents=True, exist_ok=True)

    filename = resolve_filename(entry)
    dest = _cfg().models_dir / filename
    if dest.exists():
        log.info("Model already downloaded: %s", dest)
        if on_progress is not None:
            size = dest.stat().st_size
            on_progress(size, size)  # Report 100% immediately
        return _finalize_download(entry, dest, on_progress=on_progress)

    log.info("Downloading %s/%s → %s", entry.hf_repo, filename, _cfg().models_dir)
    token = _hf_token()

    tracker = _ProgressTracker(on_progress) if on_progress else None
    config = DownloadConfig(
        repo_id=entry.hf_repo,
        filename=filename,
        token=token,
        cache_dir=str(_cfg().models_dir),
        tqdm_class=tracker.make_tqdm_class() if tracker else None,
    )

    try:
        # HF_HUB_DISABLE_XET is set in lilbee/__init__.py at import time.
        # Setting it here is too late — huggingface_hub.constants already
        # captured the value when this module first imported it.
        cached = Path(hf_hub_download(**config.model_dump(exclude_none=True)))
    except TaskCancelled:
        raise
    except GatedRepoError:
        raise PermissionError(
            f"{entry.hf_repo} requires HuggingFace authentication. "
            "Set HF_TOKEN env var or visit the repo page to request access."
        ) from None
    except RepositoryNotFoundError:
        raise RuntimeError(f"Repository {entry.hf_repo!r} not found on HuggingFace.") from None
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise RuntimeError(f"Network error downloading {entry.hf_repo}: {exc}") from None
    except OSError as exc:
        raise RuntimeError(f"I/O error downloading {entry.hf_repo}: {exc}") from None
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {entry.hf_repo}: {type(exc).__name__}: {exc}"
        ) from None

    if on_progress:
        actual_size = cached.stat().st_size
        if not tracker or not tracker.was_used:
            log.info("Model found in HuggingFace cache: %s", cached)
        on_progress(actual_size, actual_size)
    dest = cached
    return _finalize_download(entry, dest, on_progress=on_progress)


def _finalize_download(
    entry: CatalogModel,
    dest: Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Register the model in the manifest and download mmproj for vision models."""
    _register_model(entry, dest)
    if entry.task == ModelTask.VISION:
        _download_mmproj(entry, on_progress=on_progress)
    return dest


def _register_model(entry: CatalogModel, file_path: Path) -> None:
    """Create a registry manifest for a downloaded model."""
    registry = ModelRegistry(_cfg().models_dir)
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
        log.warning("Failed to register manifest for %s", entry.hf_repo, exc_info=True)


def _download_mmproj(
    entry: CatalogModel,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path | None:
    """Download the mmproj (CLIP projection) file for a vision model.
    Returns the path to the downloaded file, or None if no mmproj is configured.
    The optional ``on_progress`` callback receives ``(downloaded, total)`` byte
    counts and is wired through the same tqdm hook used by the main download.
    """
    mmproj_pattern = VISION_MMPROJ_FILES.get(entry.hf_repo, _DEFAULT_MMPROJ_PATTERN)

    mmproj_filename = _resolve_mmproj_filename(entry.hf_repo, mmproj_pattern)
    if not mmproj_filename:
        log.warning("Could not resolve mmproj file for %s", entry.hf_repo)
        return None

    from huggingface_hub import hf_hub_download

    tracker = _ProgressTracker(on_progress) if on_progress else None
    log.info("Downloading mmproj %s/%s → %s", entry.hf_repo, mmproj_filename, _cfg().models_dir)
    path = Path(
        hf_hub_download(
            repo_id=entry.hf_repo,
            filename=mmproj_filename,
            cache_dir=str(_cfg().models_dir),
            token=_hf_token(),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        )
    )
    if on_progress is not None and (not tracker or not tracker.was_used):
        # Cache hit — HF returned the cached path without invoking tqdm.
        size = path.stat().st_size
        on_progress(size, size)
    return path


def _resolve_mmproj_filename(hf_repo: str, pattern: str) -> str | None:
    """Resolve an mmproj filename pattern to a concrete filename via the HF API."""
    if "*" not in pattern:
        return pattern

    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{hf_repo}",
            timeout=_DEFAULT_TIMEOUT,
            headers=_hf_headers(),
        )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
    except Exception as exc:
        log.warning("Cannot query mmproj files for %s: %s", hf_repo, exc)
        return None

    mmproj_files: list[str] = [
        s.get("rfilename", "") for s in siblings if fnmatch.fnmatch(s.get("rfilename", ""), pattern)
    ]
    if not mmproj_files:
        return None

    # Prefer F16 over F32 (smaller), and any over BF16
    for preference in ("f16", "F16"):
        for f in mmproj_files:
            if preference in f:
                return f
    return mmproj_files[0]


def _mmproj_in_models_dir_matching(pattern: str) -> Path | None:
    """Return the first ``*.gguf`` under ``_cfg().models_dir`` that matches."""
    models_dir: Path = _cfg().models_dir
    for p in models_dir.rglob("*.gguf"):
        if fnmatch.fnmatch(p.name, pattern) or "mmproj" in p.name.lower():
            return p
    return None


def find_mmproj_file(model_ref: str) -> Path | None:
    """Find the mmproj for a ``FEATURED_VISION`` entry under ``_cfg().models_dir``.

    *model_ref* is matched against each featured vision entry's
    ``hf_repo``. Returns ``None`` when nothing matches. Never falls back
    to an arbitrary mmproj: that cross-contaminates non-vision chat
    models (e.g. a chat model would inherit a vision model's mmproj and
    be misreported as vision-capable).
    """
    # Local import to avoid pulling featured.py into hf_client/ etc.
    from lilbee.catalog.featured import FEATURED_VISION

    if not _cfg().models_dir.exists():
        return None
    for entry in FEATURED_VISION:
        if model_ref not in entry.hf_repo and entry.hf_repo not in model_ref:
            continue
        pattern = VISION_MMPROJ_FILES.get(entry.hf_repo, _DEFAULT_MMPROJ_PATTERN)
        match = _mmproj_in_models_dir_matching(pattern)
        if match is not None:
            return match
    return None


_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K", "Q3_K_M")


def resolve_filename(entry: CatalogModel) -> str:
    """Resolve a GGUF filename pattern to the best concrete filename.
    For exact filenames, return as-is. For wildcards, query the HF API
    and pick the best quantization (prefer Q4_K_M for balance of size/quality).
    """
    if "*" not in entry.gguf_filename:
        return entry.gguf_filename

    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{entry.hf_repo}",
            timeout=_DEFAULT_TIMEOUT,
            headers=_hf_headers(),
        )
        if resp.status_code == 401:
            raise PermissionError(
                f"{entry.hf_repo} requires HuggingFace authentication. "
                "Set HF_TOKEN env var or visit the repo page to request access."
            )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
    except PermissionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Cannot query files for {entry.hf_repo}: {exc}") from exc

    gguf_files = [
        s.get("rfilename", "") for s in siblings if s.get("rfilename", "").endswith(".gguf")
    ]
    if not gguf_files:
        raise RuntimeError(f"No GGUF files found in {entry.hf_repo}")

    return _pick_best_gguf(gguf_files)


def _pick_best_gguf(filenames: list[str]) -> str:
    """Pick the best GGUF file by quantization preference."""
    for quant in _QUANT_PREFERENCE:
        for f in filenames:
            if quant in f:
                return f
    return filenames[0]


def fetch_model_file_size(hf_repo: str) -> float:
    """Fetch the best GGUF file size from HuggingFace tree API.
    Returns size in GB, or 0.0 if unavailable.
    """
    try:
        resp = httpx.get(
            f"https://huggingface.co/api/models/{hf_repo}/tree/main",
            timeout=_DEFAULT_TIMEOUT,
            headers=_hf_headers(),
        )
        resp.raise_for_status()
        files = resp.json()
    except Exception:
        return 0.0

    gguf_files = [
        (f.get("path", ""), f.get("size", 0) or f.get("lfs", {}).get("size", 0))
        for f in files
        if isinstance(f, dict) and f.get("path", "").endswith(".gguf")
    ]
    if not gguf_files:
        return 0.0

    best_name = _pick_best_gguf([name for name, _ in gguf_files])
    size_bytes = next((s for n, s in gguf_files if n == best_name), 0)
    return round(size_bytes / (1024**3), 1) if size_bytes else 0.0
