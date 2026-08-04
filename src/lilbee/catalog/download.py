"""GGUF download, mmproj resolution, post-download hooks."""

import fnmatch
import logging
import re
import shutil
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from lilbee.catalog.download_progress import ProgressCallback, _ProgressTracker
from lilbee.catalog.hf_client import (
    DEFAULT_TIMEOUT,
    HF_API_URL,
    hf_headers,
    hf_token,
    repo_has_mmproj,
)
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.refs import DEFAULT_MMPROJ_PATTERN, pick_best_gguf
from lilbee.catalog.types import ModelTask
from lilbee.runtime.cancellation import TaskCancelledError

CompleteCallback = Callable[[CatalogModel, Path], None]

log = logging.getLogger(__name__)


def _models_dir() -> Path:
    """Deferred cfg read: a module-level cfg import is circular via Config()'s
    model-ref validator (config -> model_ref -> catalog -> here -> config)."""
    from lilbee.core.config.model import cfg

    return cfg.models_dir


class DownloadConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    repo_id: str
    filename: str
    token: str | None
    force_download: bool = False
    cache_dir: str | None = None
    tqdm_class: Any = None


_BYTES_PER_GB = 1024**3


def _repo_partial_bytes(models_dir: Path, hf_repo: str) -> int:
    """Bytes an interrupted attempt at *hf_repo* already holds on disk.

    A resume needs only the remainder, so these count toward available space.
    """
    from huggingface_hub.file_download import repo_folder_name

    repo_dir = models_dir / repo_folder_name(repo_id=hf_repo, repo_type="model")
    if not repo_dir.is_dir():
        return 0
    return sum(f.stat().st_size for f in repo_dir.glob("blobs/*.incomplete") if f.is_file())


def _free_bytes(path: Path) -> int | None:
    """Free space on the volume that will hold *path*, which need not exist yet.

    Measured at the nearest existing ancestor, since shutil.disk_usage raises on
    a missing path.
    """
    probe = path.resolve()
    while True:
        try:
            return shutil.disk_usage(probe).free
        except OSError:
            if probe.parent == probe:
                return None
            probe = probe.parent


def disk_shortfall(models_dir: Path, hf_repo: str, needed: int) -> str | None:
    """Describe why *needed* bytes will not fit, or None when they will."""
    if needed == _SIZE_UNKNOWN:
        return None  # offline or unresolvable; nothing to compare against
    free = _free_bytes(models_dir)
    if free is None:
        return None  # unmeasurable volume; let the download report the truth
    available = free + _repo_partial_bytes(models_dir, hf_repo)
    if needed <= available:
        return None
    return (
        f"Not enough disk space for {hf_repo}: needs "
        f"{needed / _BYTES_PER_GB:.1f} GB, {available / _BYTES_PER_GB:.1f} GB free."
    )


def _require_disk_space(entry: CatalogModel, models_dir: Path, needed: int) -> None:
    """Refuse a download the disk cannot hold, naming the shortfall.

    huggingface_hub only warns, and the xet path reports a full disk as a
    reconstruction error naming neither the disk nor the file.
    """
    message = disk_shortfall(models_dir, entry.hf_repo, needed)
    if message is not None:
        raise RuntimeError(message)


_LOW_DISK_FLOOR = 512 * 1024**2
"""Free bytes below which a failed download is reported as a full disk.

Catches a volume that filled mid-transfer, which the pre-flight cannot see."""


def _raise_if_disk_exhausted(entry: CatalogModel, config: DownloadConfig) -> None:
    """Re-raise a failed download as a disk problem when the volume is full."""
    if config.cache_dir is None:
        return
    try:
        free = shutil.disk_usage(config.cache_dir).free
    except OSError:
        return  # the path went away with the failure; leave the original error
    if free >= _LOW_DISK_FLOOR:
        return
    raise RuntimeError(
        f"Ran out of disk space downloading {entry.hf_repo}: {free / _BYTES_PER_GB:.1f} GB free."
    ) from None


_XET_CANCELLED_MARKER = "Operation cancelled"


def abort_active_download() -> None:
    """Stop the xet transfer running in this process.

    Aborts at session granularity; hf_xet exposes nothing finer.
    """
    try:
        from huggingface_hub.utils._xet import abort_xet_session
    except ImportError:
        return  # xet unavailable; the HTTP path cancels on its own
    abort_xet_session()


def _hf_download_or_translate(entry: CatalogModel, config: DownloadConfig) -> Path:
    """Run the HF download and translate every error class into a clean exception."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
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
    except Exception as exc:
        if _XET_CANCELLED_MARKER in str(exc):
            # An aborted session surfaces as a bare RuntimeError.
            raise TaskCancelledError(str(exc)) from None
        _raise_if_disk_exhausted(entry, config)
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
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = resolve_filename(entry)
    shards = split_shard_filenames(filename)
    dest = models_dir / shards[0]
    if all(
        (models_dir / shard).exists()
        and _cached_file_is_complete(entry.hf_repo, shard, models_dir / shard)
        for shard in shards
    ):
        log.info("Model already downloaded: %s", dest)
        if on_progress is not None:
            size = sum((models_dir / shard).stat().st_size for shard in shards)
            on_progress(size, size)  # Report 100% immediately (every shard)
        return _finalize_download(entry, dest, on_progress=on_progress, on_complete=on_complete)

    shard_sizes = [fetch_expected_file_size(entry.hf_repo, shard) for shard in shards]
    sizes_known = all(size != _SIZE_UNKNOWN for size in shard_sizes)
    _require_disk_space(entry, models_dir, sum(shard_sizes) if sizes_known else 0)

    # Sum the shard sizes up front so a multi-shard pull reports one monotonic
    # 0->100% against the real total, not N separate per-shard cycles. Only use
    # the sum when every shard size is known (0 = unresolved/offline); a partial
    # sum would undercount the total and let progress run past 100%.
    grand_total = sum(shard_sizes) if len(shards) > 1 and sizes_known else 0
    tracker = _ProgressTracker(on_progress, grand_total=grand_total) if on_progress else None
    shard_paths: list[Path] = []
    for shard in shards:
        log.info("Downloading %s/%s → %s", entry.hf_repo, shard, models_dir)
        config = DownloadConfig(
            repo_id=entry.hf_repo,
            filename=shard,
            token=hf_token(),
            cache_dir=str(models_dir),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        )
        shard_path = _hf_download_or_translate(entry, config)
        shard_paths.append(shard_path)
        if tracker is not None:
            tracker.shard_done(shard_path.stat().st_size)
    first_shard_path = shard_paths[0]  # the 00001-of-N shard llama.cpp loads from

    if on_progress:
        total_size = sum(path.stat().st_size for path in shard_paths)
        if not tracker or not tracker.was_used:
            log.info("Model found in HuggingFace cache: %s", first_shard_path)
        on_progress(total_size, total_size)
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
    """Run post-download hooks: registry write (via on_complete) + mmproj fetch.

    The mmproj is fetched whenever the repo ships one, not only for VISION-task
    entries: dual-use VL repos (Qwen-VL, InternVL, SmolVLM, gemma-3) classify as
    chat by name and arch, and without their projector the vision role dies at
    plan time with a missing-mmproj warning a re-pull cannot cure.
    """
    if on_complete is not None:
        on_complete(entry, dest)
    if entry.task == ModelTask.VISION or repo_has_mmproj(entry.hf_repo):
        download_mmproj(entry, on_progress=on_progress)
    return dest


def download_mmproj(
    entry: CatalogModel,
    *,
    on_progress: ProgressCallback | None = None,
) -> Path | None:
    """Download the mmproj (CLIP projection) file for a vision model.
    Returns the path to the downloaded file, or None if no mmproj is configured.
    The optional ``on_progress`` callback receives ``(downloaded, total)`` byte
    counts and is wired through the same tqdm hook used by the main download.
    """
    mmproj_filename = _resolve_mmproj_filename(entry.hf_repo, DEFAULT_MMPROJ_PATTERN)
    if not mmproj_filename:
        log.warning("Could not resolve mmproj file for %s", entry.hf_repo)
        return None

    models_dir = _models_dir()
    tracker = _ProgressTracker(on_progress) if on_progress else None
    log.info("Downloading mmproj %s/%s → %s", entry.hf_repo, mmproj_filename, models_dir)
    _require_disk_space(entry, models_dir, fetch_expected_file_size(entry.hf_repo, mmproj_filename))
    # The projector gets the same error translation as the GGUF.
    path = _hf_download_or_translate(
        entry,
        DownloadConfig(
            repo_id=entry.hf_repo,
            filename=mmproj_filename,
            token=hf_token(),
            cache_dir=str(models_dir),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        ),
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

    # Prefer an F16 mmproj when one is offered; otherwise take the first match.
    for preference in ("f16", "F16"):
        for f in mmproj_files:
            if preference in f:
                return f
    return mmproj_files[0]


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
