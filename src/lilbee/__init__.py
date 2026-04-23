"""lilbee — Local RAG knowledge base."""

from __future__ import annotations

import os
import threading

# Disable huggingface_hub's xet transfer layer before any HF submodule loads.
# huggingface_hub.constants reads HF_HUB_DISABLE_XET at import time, so this
# must run before the first `import huggingface_hub` anywhere in the process.
# Workaround for HF issue #4058: xet-core reports progress in 3-4 coarse jumps
# instead of continuously, making download bars appear stuck on large files.
# Forcing the HTTP path restores smooth per-chunk tqdm updates. Users can still
# opt back into xet by setting HF_HUB_DISABLE_XET=0 in their environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Suppress HF-default tqdm bars (metadata probes, snapshot summaries) that
# leak cursor escapes into the TUI. Our custom tqdm_class is NOT a subclass
# of huggingface_hub.utils.tqdm, so huggingface_hub's `_create_progress_bar`
# instantiates it directly without honoring this flag. Download callbacks
# continue to fire. See lilbee/catalog.py::_CallbackProgressBar.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _install_thread_only_tqdm_lock() -> None:
    """Pin tqdm's class lock to a threading.RLock before anything touches tqdm.

    Textual's App swaps stdout/stderr for a wrapper whose ``fileno()`` returns
    -1. The first call to ``tqdm.get_lock()`` lazily builds a
    ``TqdmDefaultWriteLock``, which ends up spawning multiprocessing's
    resource tracker via ``_posixsubprocess.fork_exec``. fork_exec rejects
    fd -1 with ``ValueError: bad value(s) in fds_to_keep``, which in lilbee
    surfaces as every vision OCR page failing the moment the TUI is the
    frontmost process. Setting ``_lock`` on the base class short-circuits
    the lazy path for every tqdm instance in the process, including the
    ones HF hub, sentence-transformers, and llama-cpp create on their own.
    Matches the approach taken upstream in huggingface_hub PR #4065.
    """
    try:
        from tqdm.std import tqdm as _tqdm_base
    except ImportError:
        return
    if getattr(_tqdm_base, "_lock", None) is None:
        _tqdm_base._lock = threading.RLock()


def _prestart_mp_resource_tracker() -> None:
    """Spawn multiprocessing's resource tracker now, while stderr is still real.

    ``popen_spawn_posix._launch`` calls ``resource_tracker.getfd()`` every
    time we start a worker subprocess, and the *first* call to ``getfd()``
    fork_execs the tracker. Textual's App wrapper makes ``sys.stderr.fileno()
    == -1``, which propagates into fork_exec's fds_to_keep tuple and raises
    ``ValueError: bad value(s) in fds_to_keep`` before the tracker's own
    pipe is even reached. Starting the tracker here, before Textual swaps
    stderr, keeps the fd tracker alive for the rest of the process so every
    later ``Process.start()`` in the TUI only sees a cached fd.
    """
    try:
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
    except Exception:
        # Best-effort: if the tracker already crashed or cannot be started
        # in the current env, leave the state alone. The worker's own
        # spawn will surface a real error at call time.
        pass


_install_thread_only_tqdm_lock()
_prestart_mp_resource_tracker()


def _shrink_hf_download_chunk_size() -> None:
    """Shrink huggingface_hub's 10MB download chunk to 200KB.

    Default DOWNLOAD_CHUNK_SIZE=10MB means the tqdm callback only fires
    every ~7 seconds on a 1.5MB/s connection, making downloads look stuck
    between jumps. 200KB chunks drive the callback several times per
    second at typical home-internet rates, so the UI renders smooth
    real-time progress. Monkey-patched here because HF exposes no env
    override. Runtime cost: tqdm call overhead is negligible (~µs) and
    HTTP iter_bytes accumulates into chunks of this size, so smaller
    chunks do not produce more network round-trips.
    """
    try:
        from huggingface_hub import constants as _hf_constants

        _hf_constants.DOWNLOAD_CHUNK_SIZE = 200 * 1024
    except ImportError:
        pass


_shrink_hf_download_chunk_size()

from typing import TYPE_CHECKING  # noqa: E402 — must follow HF environment / constants setup above

if TYPE_CHECKING:
    from lilbee.api import Lilbee

__all__ = ["Lilbee"]


def __getattr__(name: str) -> object:
    if name == "Lilbee":
        from lilbee.api import Lilbee

        return Lilbee
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
