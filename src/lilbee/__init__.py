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
    """Pin ``tqdm.std.tqdm._lock`` to a threading RLock.

    Bypasses tqdm's lazy multiprocessing-lock init, which tries to
    fork_exec the MP resource tracker with ``sys.stderr.fileno() == -1``
    under Textual and crashes with ``bad value(s) in fds_to_keep``.
    Matches huggingface_hub PR #4065 but applied at the base class so
    every tqdm instance in the process inherits the lock via MRO.
    """
    try:
        from tqdm.std import tqdm as _tqdm_base
    except ImportError:
        return
    if getattr(_tqdm_base, "_lock", None) is None:
        _tqdm_base._lock = threading.RLock()


def _prestart_mp_resource_tracker() -> None:
    """Start the multiprocessing resource tracker before Textual swaps stderr.

    Later ``Process.start()`` calls reuse the cached tracker fd and
    never hit the fork_exec with a bad fd. No-op on Windows, which
    doesn't use ``_posixsubprocess``, and no-op under PyInstaller frozen
    bundles, where ``sys.executable`` is the lilbee exe itself and the
    tracker's spawn would re-enter typer with ``-B -s -E`` as CLI args.
    """
    import sys as _sys

    if _sys.platform == "win32":
        return
    if getattr(_sys, "frozen", False):
        return
    try:
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
    except (OSError, RuntimeError, ValueError, ImportError):
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
