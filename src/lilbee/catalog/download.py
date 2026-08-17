"""GGUF download, mmproj resolution, post-download hooks."""

import fnmatch
import logging
import os
import re
import shutil
import sys
import threading
import time
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


def _raise_if_disk_exhausted(
    entry: CatalogModel, config: DownloadConfig, cause: BaseException
) -> None:
    """Re-raise a failed download as a disk problem when the volume is full.

    Low free space is a heuristic, not a diagnosis, so *cause* stays in the
    message.
    """
    if config.cache_dir is None:
        return
    try:
        free = shutil.disk_usage(config.cache_dir).free
    except OSError:
        return  # the path went away with the failure; leave the original error
    if free >= _LOW_DISK_FLOOR:
        return
    raise RuntimeError(
        f"Ran out of disk space downloading {entry.hf_repo}: "
        f"{free / _BYTES_PER_GB:.1f} GB free. {type(cause).__name__}: {cause}"
    ) from None


_XET_CANCELLED_MARKER = "Operation cancelled"


def abort_active_download() -> None:
    """Stop the xet transfer running in this process.

    Aborts at session granularity; hf_xet exposes nothing finer.
    """
    from huggingface_hub.utils._xet import abort_xet_session

    abort_xet_session()


_XET_HIGH_PERFORMANCE_ENV = "HF_XET_HIGH_PERFORMANCE"

_XET_DISABLE_ENV = "HF_HUB_DISABLE_XET"


def _disable_xet_where_it_stalls() -> None:
    """Fall back to the plain HTTP download path on Windows.

    hf_xet transfers stall or deadlock on Windows (xet-core issues #446,
    #789, #850), while the plain path downloads at line speed. Everywhere
    else xet stays on deliberately: it is the fast path. A user who
    exported the variable keeps whatever they chose. huggingface_hub
    parses the variable once at import, so the hub constant must change
    too; the environment write covers worker subprocesses, which parse it
    fresh.
    """
    if sys.platform != "win32":
        return
    if _XET_DISABLE_ENV in os.environ:
        return
    from huggingface_hub import constants

    os.environ[_XET_DISABLE_ENV] = "1"
    constants.HF_HUB_DISABLE_XET = True


def _apply_fast_download_mode() -> None:
    """Publish the high-performance setting to xet before it builds a session.

    hf_xet reads it from the environment in Rust and caches it when the session
    is built, so a change lands on restart.
    """
    # circular: catalog.download -> core.config via cfg, the same cycle
    # _models_dir documents (config -> model_ref -> catalog -> here).
    from lilbee.core.config.model import cfg

    if cfg.fast_model_downloads:
        os.environ[_XET_HIGH_PERFORMANCE_ENV] = "1"
    else:
        os.environ.pop(_XET_HIGH_PERFORMANCE_ENV, None)


_STALL_WINDOW_S = 60.0
"""Seconds per measurement window; a transfer below the byte floor for a
whole window counts as stalled.

Well past the hub's own 10s read timeout and its resume retries, so the
guard only fires on transfers those mechanisms cannot wake."""

_STALL_FLOOR_BYTES = 256 * 1024
"""Minimum bytes per window for a transfer to count as alive.

A wedged connection can trickle a few bytes a minute, which an any-activity
check reads as progress; ~4 KB/s is far below any usable model download."""

_STALL_POLL_S = 5.0

_STALL_RETRIES = 2


def _abort_stalled_transfer() -> None:
    """Break a wedged transfer so the blocked download thread raises.

    Covers both transports: the xet session abort stops a deadlocked Rust
    transfer (a no-op without one), and closing the hub's shared client
    closes the plain path's socket under its blocked read. The next hub
    call builds a fresh client.
    """
    from huggingface_hub.utils._http import close_session

    abort_active_download()
    close_session()


class _StallGuard:
    """Aborts a transfer that reports no bytes for the stall window.

    A wedged transfer blocks forever with the task showing active: hf_xet
    can deadlock before its first byte, a dead socket never wakes the
    plain path's read, and a dying connection can trickle bytes too slowly
    to ever finish. The guard rides the same progress stream the task bar
    shows; when a window passes under the byte floor, the abort makes the
    blocked thread raise, and the caller resumes from the .incomplete file.
    """

    def __init__(
        self,
        window_s: float = _STALL_WINDOW_S,
        poll_s: float = _STALL_POLL_S,
        floor_bytes: int = _STALL_FLOOR_BYTES,
    ) -> None:
        self._window_s = window_s
        self._poll_s = poll_s
        self._floor_bytes = floor_bytes
        self._window_start = time.monotonic()
        self._window_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired = False

    def pulse(self, n: float = 0) -> None:
        """Count transferred bytes; called from the download thread's tqdm."""
        self._window_bytes += int(n)

    def wrap_tqdm(self, tqdm_class: Any) -> Any:
        """Subclass *tqdm_class* (or the hub's default) to pulse on every update.

        ``update_transfer`` is only defined when the base has it: the hub
        feature-detects the method, so adding it to a base that lacks it
        would advertise a stream the base cannot aggregate.
        """
        from huggingface_hub.utils.tqdm import tqdm as hub_tqdm

        guard = self
        base = tqdm_class if tqdm_class is not None else hub_tqdm

        class _Pulsing(base):  # type: ignore[misc, valid-type]
            def update(self, n: float = 1) -> bool | None:
                guard.pulse(n)
                super().update(n)
                return None

        if not hasattr(base, "update_transfer"):
            return _Pulsing

        class _PulsingTransfer(_Pulsing):
            def update_transfer(self, n: float = 1) -> bool | None:
                guard.pulse(n)
                super().update_transfer(n)
                return None

        return _PulsingTransfer

    def _watch(self) -> None:
        while not self._stop.wait(self._poll_s):
            if not self._keep_watching():
                return

    def _keep_watching(self) -> bool:
        """One tick: True to keep watching, False once fired or stopped."""
        now = time.monotonic()
        if now - self._window_start < self._window_s:
            return True
        if self._stop.is_set():
            return False  # the transfer finished while this tick was deciding
        if self._window_bytes >= self._floor_bytes:
            self._window_start = now
            self._window_bytes = 0
            return True
        self.fired = True
        _abort_stalled_transfer()
        return False

    def __enter__(self) -> "_StallGuard":
        self._thread = threading.Thread(
            target=self._watch, name="download-stall-guard", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_s + 1)


def _download_with_stall_guard(entry: CatalogModel, config: DownloadConfig) -> Path:
    """Run the transfer under the stall guard, resuming after each stall.

    huggingface_hub resumes from the .incomplete file, so a retry costs only
    the bytes since the stall. A failure with the guard quiet is a real
    error and propagates on the first attempt; cancellation always does.
    """
    last_error: Exception | None = None
    for attempt in range(_STALL_RETRIES + 1):
        guard = _StallGuard()
        guarded = config.model_copy(update={"tqdm_class": guard.wrap_tqdm(config.tqdm_class)})
        try:
            with guard:
                return _hf_download_or_translate(entry, guarded)
        except TaskCancelledError:
            raise
        except Exception as exc:
            if not guard.fired:
                raise
            last_error = exc
            log.warning(
                "Transfer of %s stalled (attempt %d/%d); resuming.",
                entry.hf_repo,
                attempt + 1,
                _STALL_RETRIES + 1,
            )
    raise RuntimeError(
        f"Download of {entry.hf_repo} stalled {_STALL_RETRIES + 1} times with almost "
        "no data arriving. Check the network connection and retry; the finished part "
        "is kept and the download resumes where it stopped."
    ) from last_error


def _hf_download_or_translate(entry: CatalogModel, config: DownloadConfig) -> Path:
    """Run the HF download and translate every error class into a clean exception."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    _apply_fast_download_mode()
    _disable_xet_where_it_stalls()
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
        _raise_if_disk_exhausted(entry, config, exc)
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
        shard_path = _download_with_stall_guard(entry, config)
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
    # The projector gets the same error translation and stall guard as the GGUF.
    path = _download_with_stall_guard(
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
