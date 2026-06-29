"""lilbee: local knowledge base."""

from __future__ import annotations

import os
import threading

# Disable huggingface_hub's xet transfer by default so the download bar stays
# smooth: xet reports progress in a few coarse jumps (the bar looks stuck),
# while the plain HTTP path with small chunks updates several times a second.
# The catalog re-enables xet per-download for large files (catalog/download.py),
# where xet's speed is worth the coarser bar. Set HF_HUB_DISABLE_XET=0 to force
# xet for every download.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Suppress HF-default tqdm bars (metadata probes, snapshot summaries) that
# leak cursor escapes into the TUI. Our custom tqdm_class is NOT a subclass
# of huggingface_hub.utils.tqdm, so huggingface_hub's `_create_progress_bar`
# instantiates it directly without honoring this flag. Download callbacks
# continue to fire. See lilbee/catalog/download_progress.py::_CallbackProgressBar.
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
    doesn't use ``_posixsubprocess``, and no-op under a frozen build
    (Nuitka onefile), where ``sys.executable`` is the lilbee exe itself
    and the tracker's spawn would re-enter typer with ``-B -s -E`` as
    CLI args (``__main__._dispatch_frozen_child`` handles that case).

    Detecting the frozen build via :func:`lilbee._frozen.is_frozen` is
    load-bearing: Nuitka never sets ``sys.frozen``, so a check on that
    attribute alone would let the tracker spawn fire inside the onefile
    binary and surface the typer error.
    """
    import sys as _sys

    from lilbee._frozen import is_frozen

    if _sys.platform == "win32":
        return
    if is_frozen():
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
        pass  # huggingface_hub may be absent in stripped-down environments


_shrink_hf_download_chunk_size()


# HF and LiteLLM log filters live next to their respective implementations
# (catalog/hf_client.py and providers/litellm_sdk.py). They install themselves
# on module import; this package's __init__.py stays free of HF/LiteLLM
# implementation detail.


# Must follow HF environment / constants setup above.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from lilbee.api import Lilbee

__all__ = ["Lilbee"]


def __getattr__(name: str) -> object:
    if name == "Lilbee":
        from lilbee.api import Lilbee

        return Lilbee
    # Resolve real submodules on attribute access (e.g. ``lilbee.providers`` from a
    # mock.patch string target) without forcing eager imports at package load.
    # Unknown names still raise AttributeError.
    import importlib

    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
