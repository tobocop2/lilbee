"""GGUF download, mmproj resolution, post-download hooks."""

import fnmatch
import logging
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from http import HTTPStatus
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import httpx
from pydantic import BaseModel

from lilbee.catalog.compat import UnsupportedQuantError, classify, file_header
from lilbee.catalog.download_progress import ProgressCallback, _ProgressTracker
from lilbee.catalog.hf_client import (
    DEFAULT_TIMEOUT,
    HF_API_URL,
    hf_headers,
    hf_token,
    repo_has_mmproj,
)
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.refs import (
    DEFAULT_MMPROJ_PATTERN,
    FLOAT_QUANTS,
    WILDCARD,
    quant_label,
    rank_gguf_candidates,
    split_shard_filenames,
)
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError

_T = TypeVar("_T")

CompleteCallback = Callable[[CatalogModel, Path], None]
# Raises UnsupportedQuantError when the engine cannot decode the named file.
LoadCheck = Callable[[str, str], None]
# Called once per file resolved with the Hub, before any bytes transfer, with
# the cache blob that file occupies (None when the Hub reports no blob).
ProbeCallback = Callable[[str | None], None]

log = logging.getLogger(__name__)


class RemoteFile(NamedTuple):
    """The Hub's answer about one repo file.

    *blob* is the name the file takes in the cache's blob directory, and it is
    None when the Hub reports no entity tag for the file.
    """

    size: int
    blob: str | None


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
    """Describe why *needed* bytes will not fit, or None when they will.

    A partial blob from an interrupted attempt is not counted: huggingface_hub
    writes each transfer to a fresh temporary file, so those bytes are spent.
    """
    if needed == _SIZE_UNKNOWN:
        return None  # offline or unresolvable; nothing to compare against
    free = _free_bytes(models_dir)
    if free is None:
        return None  # unmeasurable volume; let the download report the truth
    if needed <= free:
        return None
    return (
        f"Not enough disk space for {hf_repo}: needs "
        f"{needed / _BYTES_PER_GB:.1f} GB, {free / _BYTES_PER_GB:.1f} GB free."
    )


def discard_partial_blobs(models_dir: Path, hf_repo: str, blobs: Iterable[str]) -> None:
    """Delete the temporary files of *blobs*, which no later download reads.

    huggingface_hub writes every transfer to a fresh ``<blob>.<unique>.incomplete``
    name and unlinks it on the way out, so a leftover belongs to an attempt that
    never unwound: a terminated child, a power loss, a build that predates this
    sweep. Left alone the bytes are lost for the life of the cache.

    Scoped to the blobs the caller resolved, because another quant of the same
    repo downloads into the same directory at the same time and its temporary
    file is live.
    """
    from huggingface_hub.file_download import repo_folder_name

    repo_dir = models_dir / repo_folder_name(repo_id=hf_repo, repo_type="model")
    for blob in blobs:
        for partial in repo_dir.glob(f"blobs/{blob}.*.incomplete"):
            try:
                partial.unlink()
            except OSError:
                log.warning("Left a partial download behind: %s", partial)


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


_TRANSFER_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 5


class _TransientDownloadError(RuntimeError):
    """A transfer fault another attempt can clear: a network or an I/O error."""


def _retry_transient(work: Callable[[], _T]) -> _T:
    """Run *work*, retrying only the faults another attempt can clear.

    Every other error propagates on the first attempt, cancellation included,
    so a defect or a configuration error is never hidden behind a retry. A
    transfer that goes quiet raises nothing here, so it is the parent process
    that ends it, not this loop.
    """
    last_error: Exception | None = None
    for attempt in range(_TRANSFER_RETRIES + 1):
        try:
            return work()
        except _TransientDownloadError as exc:
            last_error = exc
            log.warning("%s (attempt %d/%d)", exc, attempt + 1, _TRANSFER_RETRIES + 1)
        if attempt < _TRANSFER_RETRIES:
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(
        f"{last_error}. The transfer failed {_TRANSFER_RETRIES + 1} times. Check the "
        "network connection and retry; the files that finished are kept."
    ) from last_error


def _download_with_retry(entry: CatalogModel, config: DownloadConfig) -> Path:
    """Run one file's transfer, retrying the faults another attempt can clear."""
    return _retry_transient(lambda: _hf_download_or_translate(entry, config))


def _hf_download_or_translate(entry: CatalogModel, config: DownloadConfig) -> Path:
    """Run the HF download and translate every error class into a clean exception."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

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
    except EntryNotFoundError:
        raise RuntimeError(_missing_file_message(entry.hf_repo, config.filename)) from None
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise _TransientDownloadError(f"Network error downloading {entry.hf_repo}: {exc}") from None
    except OSError as exc:
        raise _TransientDownloadError(f"I/O error downloading {entry.hf_repo}: {exc}") from None
    except Exception as exc:
        _raise_if_disk_exhausted(entry, config, exc)
        raise RuntimeError(
            f"Failed to download {entry.hf_repo}: {type(exc).__name__}: {exc}"
        ) from None


class _NeverCancelled:
    """Cancel signal for a caller that has no way to stop the download."""

    def is_set(self) -> bool:
        """Always False; nothing holds a handle that could set this."""
        return False


_NEVER_CANCELLED = _NeverCancelled()


def download_model(
    entry: CatalogModel,
    *,
    on_progress: ProgressCallback | None = None,
    on_complete: CompleteCallback | None = None,
    cancel: CancelSignal | None = None,
) -> Path:
    """Download a GGUF model from HuggingFace to the models dir.
    Uses huggingface_hub for caching and auth; a stalled file starts again
    from the top, and files already finished are kept.
    The optional *on_progress(downloaded, total)* callback receives byte counts.
    The optional *on_complete(entry, file_path)* callback runs after every file
    is on disk; modelhub uses it to write a registry manifest. For vision
    models, also downloads the mmproj (CLIP projection) file.

    The transfer always runs in its own child process, because terminating that
    process is the only stop a wedged transfer cannot refuse. A caller that
    passes no *cancel* signal gets one that is never set, so the watchdog that
    ends a quiet child covers every download.

    A split GGUF has every shard fetched before the model is finalized, so the
    registry manifest (and thus "installed") only lands once the full set is on
    disk; an interrupted multi-part pull leaves the model not-installed and
    re-pullable rather than registered-but-unloadable.

    Raises:
        PermissionError: gated repo requiring authentication
        RuntimeError: repo not found or download failure with details
        TaskCancelledError: the cancel signal was set
    """
    _apply_fast_download_mode()
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    # circular: download -> download_process via fetch_model_files
    from lilbee.catalog.download_process import download_in_subprocess

    dest = download_in_subprocess(
        entry,
        models_dir,
        token,
        on_progress=on_progress,
        cancel=_NEVER_CANCELLED if cancel is None else cancel,
    )
    if on_complete is not None:
        on_complete(entry, dest)
    return dest


def fetch_model_files(
    entry: CatalogModel,
    models_dir: Path,
    token: str | None,
    *,
    on_progress: ProgressCallback | None = None,
    on_probe: ProbeCallback | None = None,
) -> Path:
    """Fetch *entry*'s GGUF shards, plus its projector when the repo ships one.

    Takes the models dir and token as arguments so a download child process
    can run it without reading cfg. Writes no registry state. *on_probe* fires
    once per file resolved with the Hub, which is the only sign of life a
    caller gets before the first bytes, and it names the cache blob so the
    caller can clear that file's leftovers.
    """
    filename = resolve_filename(entry)
    shards = split_shard_filenames(filename)
    dest = models_dir / shards[0]
    if all(_shard_is_cached(entry, models_dir, shard, on_probe) for shard in shards):
        log.info("Model already downloaded: %s", dest)
        if on_progress is not None:
            size = sum((models_dir / shard).stat().st_size for shard in shards)
            on_progress(size, size)  # Report 100% immediately (every shard)
        _ensure_projector(entry, models_dir, token, on_progress=on_progress, on_probe=on_probe)
        return dest

    remote = [_probed_shard(entry, shard, on_probe) for shard in shards]
    shard_sizes = [file.size for file in remote]
    sizes_known = all(size != _SIZE_UNKNOWN for size in shard_sizes)
    # Reclaim first: a leftover partial is unreadable bytes that still occupy
    # the volume the space check is about to measure.
    discard_partial_blobs(models_dir, entry.hf_repo, _blobs(remote))
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
            token=token,
            cache_dir=str(models_dir),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        )
        shard_path = _download_with_retry(entry, config)
        shard_paths.append(shard_path)
        if tracker is not None:
            tracker.shard_done(shard_path.stat().st_size)
    first_shard_path = shard_paths[0]  # the 00001-of-N shard llama.cpp loads from

    if on_progress:
        total_size = sum(path.stat().st_size for path in shard_paths)
        if not tracker or not tracker.was_used:
            log.info("Model found in HuggingFace cache: %s", first_shard_path)
        on_progress(total_size, total_size)
    _ensure_projector(entry, models_dir, token, on_progress=on_progress, on_probe=on_probe)
    return first_shard_path


def _blobs(files: Iterable[RemoteFile]) -> list[str]:
    """The cache blob names among *files*, dropping the ones the Hub did not report."""
    return [file.blob for file in files if file.blob is not None]


def _probe(on_probe: ProbeCallback | None, file: RemoteFile) -> RemoteFile:
    """Report *file* as resolved, then hand it back to the caller."""
    if on_probe is not None:
        on_probe(file.blob)
    return file


def _shard_is_cached(
    entry: CatalogModel, models_dir: Path, shard: str, on_probe: ProbeCallback | None
) -> bool:
    """Whether *shard* is on disk at the size the Hub reports for it."""
    file = _probe(on_probe, fetch_remote_file(entry.hf_repo, shard))
    path = models_dir / shard
    return path.exists() and _size_matches(path, file.size)


def _probed_shard(entry: CatalogModel, shard: str, on_probe: ProbeCallback | None) -> RemoteFile:
    """The Hub's answer about *shard*, reported to *on_probe* as it arrives."""
    return _probe(on_probe, fetch_remote_file(entry.hf_repo, shard))


def _ensure_projector(
    entry: CatalogModel,
    models_dir: Path,
    token: str | None,
    *,
    on_progress: ProgressCallback | None = None,
    on_probe: ProbeCallback | None = None,
) -> None:
    """Fetch the projector whenever the repo ships one, not only for VISION entries.

    Dual-use VL repos (Qwen-VL, InternVL, SmolVLM, gemma-3) classify as chat by
    name and arch, and without their projector the vision role dies at plan
    time with a missing-mmproj warning a re-pull cannot cure.
    """
    if entry.task == ModelTask.VISION or repo_has_mmproj(entry.hf_repo):
        _fetch_mmproj(entry, models_dir, token, on_progress=on_progress, on_probe=on_probe)


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
    _apply_fast_download_mode()
    return _fetch_mmproj(entry, _models_dir(), hf_token(), on_progress=on_progress)


def _fetch_mmproj(
    entry: CatalogModel,
    models_dir: Path,
    token: str | None,
    *,
    on_progress: ProgressCallback | None = None,
    on_probe: ProbeCallback | None = None,
) -> Path | None:
    """Fetch *entry*'s mmproj into *models_dir*, or None when the repo names none."""
    mmproj_filename = _resolve_mmproj_filename(entry.hf_repo, DEFAULT_MMPROJ_PATTERN)
    if not mmproj_filename:
        log.warning("Could not resolve mmproj file for %s", entry.hf_repo)
        return None

    tracker = _ProgressTracker(on_progress) if on_progress else None
    log.info("Downloading mmproj %s/%s → %s", entry.hf_repo, mmproj_filename, models_dir)
    projector = _probe(on_probe, fetch_remote_file(entry.hf_repo, mmproj_filename))
    discard_partial_blobs(models_dir, entry.hf_repo, _blobs([projector]))
    _require_disk_space(entry, models_dir, projector.size)
    # The projector gets the same error translation and retry as the GGUF.
    path = _download_with_retry(
        entry,
        DownloadConfig(
            repo_id=entry.hf_repo,
            filename=mmproj_filename,
            token=token,
            cache_dir=str(models_dir),
            tqdm_class=tracker.make_tqdm_class() if tracker else None,
        ),
    )
    if on_progress is not None and (not tracker or not tracker.was_used):
        # Cache hit: HF returned the cached path without invoking tqdm.
        size = path.stat().st_size
        on_progress(size, size)
    return path


def _repo_sibling_files(hf_repo: str) -> list[str]:
    """Every filename the HuggingFace API lists for *hf_repo*.

    Raises:
        PermissionError: the repo is gated and needs authentication.
        RuntimeError: the listing could not be fetched.
    """
    try:
        resp = httpx.get(
            f"{HF_API_URL}/{hf_repo}",
            timeout=DEFAULT_TIMEOUT,
            headers=hf_headers(),
        )
        if resp.status_code == HTTPStatus.UNAUTHORIZED:
            raise PermissionError(
                f"{hf_repo} requires HuggingFace authentication. "
                "Set HF_TOKEN env var or visit the repo page to request access."
            )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
    except PermissionError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Cannot query files for {hf_repo}: {exc}") from exc
    return [s.get("rfilename", "") for s in siblings]


def _mmproj_rank(filename: str) -> tuple[bool, str]:
    """Sort key preferring an unquantized projector, ties broken by name."""
    return (quant_label(filename) not in FLOAT_QUANTS, filename)


def _resolve_mmproj_filename(hf_repo: str, pattern: str) -> str | None:
    """Resolve an mmproj filename pattern to a concrete filename via the HF API."""
    if WILDCARD not in pattern:
        return pattern
    try:
        names = _repo_sibling_files(hf_repo)
    except (PermissionError, RuntimeError) as exc:
        log.warning("Cannot query mmproj files for %s: %s", hf_repo, exc)
        return None
    matches = [name for name in names if fnmatch.fnmatch(name, pattern)]
    return min(matches, key=_mmproj_rank) if matches else None


def resolve_filename(entry: CatalogModel, *, can_load: LoadCheck | None = None) -> str:
    """The repo file a pull of *entry* fetches, gated on each candidate's GGUF header.

    Quant labels only order the candidates. A file whose header calls it a
    projector or an adapter is never the model, so a repo that labels its
    projector ``Q8_0`` cannot install it as one.

    *can_load* is the engine's verdict on one file, supplied by the layer that
    owns the engine. It runs last because it is the expensive question, and it
    decides between candidates rather than judging the one already chosen: a
    repo publishing the same weights in several packings holds files the engine
    reads and files it cannot, and only one of them is worth downloading.

    Where no architecture is supported the best-ranked weights still come back.
    Refusing here would report a generic error and disable ``--allow-unsupported``,
    so that verdict belongs to the architecture guard.

    Raises:
        PermissionError: the repo is gated and needs authentication.
        UnsupportedQuantError: every candidate carries weights the engine cannot
            decode; the first such refusal is re-raised, naming its file.
        RuntimeError: the repo listing failed, or it holds no model weights.
    """
    named = entry.gguf_filename
    if WILDCARD not in named and file_header(entry.hf_repo, named).is_model:
        return named
    unsupported: str | None = None
    refused: UnsupportedQuantError | None = None
    for candidate in rank_gguf_candidates(_repo_sibling_files(entry.hf_repo)):
        header = file_header(entry.hf_repo, candidate)
        if not header.is_model:
            continue
        if classify(header.architecture) is ModelCompat.UNSUPPORTED:
            unsupported = unsupported or candidate
            continue
        if can_load is not None:
            try:
                can_load(entry.hf_repo, candidate)
            except UnsupportedQuantError as exc:
                refused = refused or exc
                continue
        return candidate
    if unsupported is not None:
        return unsupported
    if refused is not None:
        raise refused
    raise RuntimeError(f"No GGUF model weights found in {entry.hf_repo}")


_SIZE_UNKNOWN = 0


def _size_matches(dest: Path, expected: int) -> bool:
    """Whether *dest* holds the *expected* byte count, accepting an unknown one.

    An unknown size is accepted because there is nothing to verify against and
    refusing would block every offline reuse. A size that disagrees is a
    truncated or corrupt file, so the caller fetches it again.
    """
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


def _hf_file_metadata(hf_repo: str, filename: str) -> tuple[int | None, str | None]:
    """Byte size and cache blob huggingface_hub resolves for *filename*."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    metadata = get_hf_file_metadata(hf_hub_url(hf_repo, filename), token=hf_token())
    return metadata.size, metadata.etag


def _missing_file_message(hf_repo: str, filename: str) -> str:
    """User-facing error for a file the Hub reports as nonexistent."""
    return (
        f"File {filename!r} does not exist in {hf_repo} on HuggingFace. "
        "Check the filename on the repo page."
    )


def fetch_remote_file(hf_repo: str, filename: str) -> RemoteFile:
    """Return the size and cache blob huggingface_hub reports for *filename*.

    Resolves via hf_hub's own file metadata (correct revision, redirects, and
    LFS/Xet handled uniformly) instead of scraping the repo tree. Reports an
    unknown size when offline or unresolvable, in which case the caller keeps
    the cached file. A file the Hub reports as nonexistent raises instead: that
    answer is definitive, and treating it as unknown let a pull of a mistyped
    filename accept a stale local file and report success without downloading
    anything.
    """
    from huggingface_hub.errors import RemoteEntryNotFoundError

    try:
        size, blob = _hf_file_metadata(hf_repo, filename)
    except RemoteEntryNotFoundError:
        raise RuntimeError(_missing_file_message(hf_repo, filename)) from None
    except Exception:
        return RemoteFile(size=_SIZE_UNKNOWN, blob=None)
    return RemoteFile(size=size or _SIZE_UNKNOWN, blob=blob)


def download_bytes(hf_repo: str, filename: str) -> int:
    """Bytes a pull of *filename* fetches, every shard summed, or 0 when unknown.

    The exact figure HuggingFace reports, not the catalog row's approximation:
    a disk check refuses a real download, so it asks about the real file. A
    single unresolvable shard makes the sum unknown rather than short.
    """
    sizes = [fetch_remote_file(hf_repo, shard).size for shard in split_shard_filenames(filename)]
    return sum(sizes) if all(sizes) else _SIZE_UNKNOWN
