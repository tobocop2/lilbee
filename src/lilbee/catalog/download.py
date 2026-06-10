"""GGUF download, mmproj resolution, post-download hooks."""

import fnmatch
import logging
import re
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from lilbee.catalog.download_progress import ProgressCallback, _ProgressTracker
from lilbee.catalog.featured import DEFAULT_MMPROJ_PATTERN, VISION_MMPROJ_FILES
from lilbee.catalog.hf_client import DEFAULT_TIMEOUT, HF_API_URL, hf_headers, hf_token
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.refs import pick_best_gguf
from lilbee.catalog.types import ModelTask
from lilbee.runtime.cancellation import TaskCancelledError

CompleteCallback = Callable[[CatalogModel, Path], None]


def _models_dir() -> Path:
    """Deferred cfg read: a module-level cfg import is circular via Config()'s
    model-ref validator (config -> model_ref -> catalog -> here -> config)."""
    from lilbee.core.config.model import cfg

    return cfg.models_dir


log = logging.getLogger(__name__)


class DownloadConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    repo_id: str
    filename: str
    token: str | None
    force_download: bool = False
    cache_dir: str | None = None
    tqdm_class: Any = None


_HTTP_TOO_LARGE_MARKER = "too large to be downloaded using the regular download method"


def _download_with_xet(config: DownloadConfig) -> Path:
    """Re-run the download with xet enabled, for files past the HTTP size cap.

    lilbee disables xet by default (``HF_HUB_DISABLE_XET``) so download progress
    bars stay smooth, but huggingface_hub refuses files over its HTTP size cap on
    the regular path and only xet can fetch them. ``is_xet_available()`` reads the
    constant live, so flip it for this one download and restore it after. hf_xet
    is a hard dependency, so the xet path is always available.
    """
    from huggingface_hub import constants, hf_hub_download

    original = constants.HF_HUB_DISABLE_XET
    constants.HF_HUB_DISABLE_XET = False
    try:
        log.info("File exceeds the HTTP download cap; retrying %s via xet.", config.repo_id)
        return Path(hf_hub_download(**config.model_dump(exclude_none=True)))
    finally:
        constants.HF_HUB_DISABLE_XET = original


def _hf_download_or_translate(entry: CatalogModel, config: DownloadConfig) -> Path:
    """Run the HF download and translate every error class into a clean exception."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        # HF_HUB_DISABLE_XET is set in lilbee/__init__.py at import time; the
        # _download_with_xet fallback flips the constant directly (not the env)
        # for files that only xet can deliver.
        return Path(hf_hub_download(**config.model_dump(exclude_none=True)))
    except TaskCancelledError:
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
    except ValueError as exc:
        if _HTTP_TOO_LARGE_MARKER in str(exc):
            return _download_with_xet(config)
        raise RuntimeError(f"Failed to download {entry.hf_repo}: ValueError: {exc}") from None
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {entry.hf_repo}: {type(exc).__name__}: {exc}"
        ) from None


_SPLIT_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def split_shard_filenames(filename: str) -> list[str]:
    """Return every shard of a split GGUF in order, or ``[filename]`` if it isn't split.

    A split GGUF names its parts ``<base>-00001-of-0000N.gguf`` through
    ``<base>-0000N-of-0000N.gguf``. llama.cpp loads the whole set from the first
    shard but needs every part on disk, so the catalog must fetch all of them and
    only consider the model installed once the full set is present.
    """
    match = _SPLIT_SHARD_RE.match(filename)
    if match is None:
        return [filename]
    base = match.group("base")
    total = int(match.group("total"))
    return [f"{base}-{index:05d}-of-{total:05d}.gguf" for index in range(1, total + 1)]


def download_model(
    entry: CatalogModel,
    *,
    on_progress: ProgressCallback | None = None,
    on_complete: CompleteCallback | None = None,
) -> Path:
    """Download a GGUF model from HuggingFace to the models dir.
    Uses huggingface_hub for resumable downloads, caching, and auth.
    The optional *on_progress(downloaded, total)* callback receives byte counts.
    The optional *on_complete(entry, file_path)* callback runs after the file
    is on disk; modelhub uses it to write a registry manifest. For vision
    models, also downloads the mmproj (CLIP projection) file.

    A split GGUF has every shard fetched before the model is finalized, so the
    registry manifest (and thus "installed") only lands once the full set is on
    disk; an interrupted multi-part pull leaves the model not-installed and
    re-pullable rather than registered-but-unloadable.

    Raises:
        PermissionError: gated repo requiring authentication
        RuntimeError: repo not found or download failure with details
    """
    _models_dir().mkdir(parents=True, exist_ok=True)

    filename = resolve_filename(entry)
    shards = split_shard_filenames(filename)
    dest = _models_dir() / shards[0]
    if all(
        (_models_dir() / shard).exists()
        and _cached_file_is_complete(entry.hf_repo, shard, _models_dir() / shard)
        for shard in shards
    ):
        log.info("Model already downloaded: %s", dest)
        if on_progress is not None:
            size = dest.stat().st_size
            on_progress(size, size)  # Report 100% immediately
        return _finalize_download(entry, dest, on_progress=on_progress, on_complete=on_complete)

    tracker = _ProgressTracker(on_progress) if on_progress else None
    shard_paths: list[Path] = []
    for shard in shards:
        log.info("Downloading %s/%s → %s", entry.hf_repo, shard, _models_dir())
        config = DownloadConfig(
            repo_id=entry.hf_repo,
            filename=shard,
            token=hf_token(),
            cache_dir=str(_models_dir()),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        )
        shard_paths.append(_hf_download_or_translate(entry, config))
    first_shard_path = shard_paths[0]  # the 00001-of-N shard llama.cpp loads from

    if on_progress:
        actual_size = first_shard_path.stat().st_size
        if not tracker or not tracker.was_used:
            log.info("Model found in HuggingFace cache: %s", first_shard_path)
        on_progress(actual_size, actual_size)
    return _finalize_download(
        entry, first_shard_path, on_progress=on_progress, on_complete=on_complete
    )


def _finalize_download(
    entry: CatalogModel,
    dest: Path,
    *,
    on_progress: ProgressCallback | None = None,
    on_complete: CompleteCallback | None = None,
) -> Path:
    """Run post-download hooks: registry write (via on_complete) + mmproj fetch."""
    if on_complete is not None:
        on_complete(entry, dest)
    if entry.task == ModelTask.VISION:
        _download_mmproj(entry, on_progress=on_progress)
    return dest


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
    mmproj_pattern = VISION_MMPROJ_FILES.get(entry.hf_repo, DEFAULT_MMPROJ_PATTERN)

    mmproj_filename = _resolve_mmproj_filename(entry.hf_repo, mmproj_pattern)
    if not mmproj_filename:
        log.warning("Could not resolve mmproj file for %s", entry.hf_repo)
        return None

    from huggingface_hub import hf_hub_download

    tracker = _ProgressTracker(on_progress) if on_progress else None
    log.info("Downloading mmproj %s/%s → %s", entry.hf_repo, mmproj_filename, _models_dir())
    path = Path(
        hf_hub_download(
            repo_id=entry.hf_repo,
            filename=mmproj_filename,
            cache_dir=str(_models_dir()),
            token=hf_token(),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        )
    )
    if on_progress is not None and (not tracker or not tracker.was_used):
        # Cache hit: HF returned the cached path without invoking tqdm.
        size = path.stat().st_size
        on_progress(size, size)
    return path


def _resolve_mmproj_filename(hf_repo: str, pattern: str) -> str | None:
    """Resolve an mmproj filename pattern to a concrete filename via the HF API."""
    if "*" not in pattern:
        return pattern

    try:
        resp = httpx.get(
            f"{HF_API_URL}/{hf_repo}",
            timeout=DEFAULT_TIMEOUT,
            headers=hf_headers(),
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
    """Return the first ``*.gguf`` under the models dir that matches."""
    models_dir: Path = _models_dir()
    for p in models_dir.rglob("*.gguf"):
        if fnmatch.fnmatch(p.name, pattern) or "mmproj" in p.name.lower():
            return p
    return None


def find_mmproj_file(model_ref: str) -> Path | None:
    """Find the mmproj for a ``FEATURED_VISION`` entry under the models dir.

    *model_ref* is matched against each featured vision entry's
    ``hf_repo``. Returns ``None`` when nothing matches. Never falls back
    to an arbitrary mmproj: that cross-contaminates non-vision chat
    models (e.g. a chat model would inherit a vision model's mmproj and
    be misreported as vision-capable).
    """
    # Local import to avoid pulling featured.py into hf_client/ etc.
    from lilbee.catalog.featured import FEATURED_VISION

    if not _models_dir().exists():
        return None
    for entry in FEATURED_VISION:
        if model_ref not in entry.hf_repo and entry.hf_repo not in model_ref:
            continue
        pattern = VISION_MMPROJ_FILES.get(entry.hf_repo, DEFAULT_MMPROJ_PATTERN)
        match = _mmproj_in_models_dir_matching(pattern)
        if match is not None:
            return match
    return None


def resolve_filename(entry: CatalogModel) -> str:
    """Resolve a GGUF filename pattern to the best concrete filename.
    For exact filenames, return as-is. For wildcards, query the HF API
    and pick the best quantization (prefer Q4_K_M for balance of size/quality).
    """
    if "*" not in entry.gguf_filename:
        return entry.gguf_filename

    try:
        resp = httpx.get(
            f"{HF_API_URL}/{entry.hf_repo}",
            timeout=DEFAULT_TIMEOUT,
            headers=hf_headers(),
        )
        if resp.status_code == HTTPStatus.UNAUTHORIZED:
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

    return pick_best_gguf(gguf_files)


_SIZE_UNKNOWN = 0


def _cached_file_is_complete(hf_repo: str, filename: str, dest: Path) -> bool:
    """Decide whether an existing cached file may be accepted as complete.

    Verifies the on-disk byte size against the size HuggingFace reports for
    *filename*. A mismatch means a truncated / corrupt download, so the file
    is rejected and re-fetched. When the size can't be fetched (offline, API
    error) it stays unknown and the cached file is accepted: there's nothing
    to verify against and refusing would block all offline reuse.
    """
    expected = fetch_expected_file_size(hf_repo, filename)
    if expected == _SIZE_UNKNOWN:
        return True
    actual = dest.stat().st_size
    if actual == expected:
        return True
    log.warning(
        "Cached %s is %d bytes but HuggingFace reports %d; re-downloading",
        dest,
        actual,
        expected,
    )
    return False


def _hf_file_size(hf_repo: str, filename: str) -> int | None:
    """Byte size huggingface_hub resolves for *filename* (None if unreported)."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    return get_hf_file_metadata(hf_hub_url(hf_repo, filename), token=hf_token()).size


def fetch_expected_file_size(hf_repo: str, filename: str) -> int:
    """Return the byte size huggingface_hub reports for *filename*, or _SIZE_UNKNOWN.

    Resolves via hf_hub's own file metadata (correct revision, redirects, and
    LFS/Xet handled uniformly) instead of scraping the repo tree. Returns 0 when
    offline or unresolvable, in which case the caller keeps the cached file.
    """
    try:
        return _hf_file_size(hf_repo, filename) or _SIZE_UNKNOWN
    except Exception:
        return _SIZE_UNKNOWN


def fetch_model_file_size(hf_repo: str) -> float:
    """Fetch the best GGUF file size from HuggingFace tree API.
    Returns size in GB, or 0.0 if unavailable.
    """
    try:
        resp = httpx.get(
            f"{HF_API_URL}/{hf_repo}/tree/main",
            timeout=DEFAULT_TIMEOUT,
            headers=hf_headers(),
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

    best_name = pick_best_gguf([name for name, _ in gguf_files])
    size_bytes = next((s for n, s in gguf_files if n == best_name), 0)
    return round(size_bytes / (1024**3), 1) if size_bytes else 0.0
