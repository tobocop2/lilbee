"""llama.cpp log dispatcher and native-stderr suppression.

Routes llama.cpp's C-stream log messages through Python logging, coalescing
CONT continuation chunks until a newline. The pending buffer is owned by the
``_LogDispatcher`` singleton because it tracks request-scoped state for a
ctypes callback.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from lilbee.providers.base import ProviderError

_llama_log = logging.getLogger("lilbee.llama_cpp")

# ggml.h log levels (not exposed by llama-cpp-python).
_GGML_LOG_LEVEL_INFO = 1
_GGML_LOG_LEVEL_WARN = 2
_GGML_LOG_LEVEL_ERROR = 3
_GGML_LOG_LEVEL_DEBUG = 4
_GGML_LOG_LEVEL_CONT = 5

# WARN demotes to INFO so noisy auto-corrections stay silent at the default WARNING level.
_GGML_TO_PY_LEVEL = {
    _GGML_LOG_LEVEL_INFO: logging.DEBUG,
    _GGML_LOG_LEVEL_WARN: logging.INFO,
    _GGML_LOG_LEVEL_ERROR: logging.ERROR,
    _GGML_LOG_LEVEL_DEBUG: logging.DEBUG,
}

# llama.cpp emits these at GGML_LOG_LEVEL_ERROR but they're advisory: the model
# still loads correctly. Demote to WARNING so users don't think the setup is
# broken.
_GGML_ERROR_SOFT_DEMOTE = (
    "special_eos_id is not in special_eog_ids",
    "embeddings required but some input tokens were not marked as outputs",
    # 'n_ctx_seq (X) > n_ctx_train (Y)' -- our embed clamp prevents misuse.
    "n_ctx_seq",
    "tokenizer config may be incorrect",
    # Reported at ERROR every model load on pre-M5/A19 macs (i.e. most macs);
    # cosmetic.
    "tensor API disabled for pre-M5",
    # 'control-looking token: ... was not control-type'. Tokenizer-config
    # quirk; llama.cpp auto-corrects.
    "was not control-type",
    # 'special_eog_ids contains "<|tool_response>", removing "</s>"'.
    # Auto-correction of EOG list.
    "removing '</s>' token from EOG list",
    # Format-string bug in upstream tensor-name logging; harmless.
    "cannot properly format tensor name",
    # SWA full-size cache fallback for iSWA models; informational.
    "using full-size SWA cache",
)

_STDERR_LOCK = threading.Lock()


def _resolve_ggml_level(ggml_level: int, text: str) -> int:
    """Translate ggml log level to Python, demoting known-advisory ERRORs to WARNING."""
    py_level = _GGML_TO_PY_LEVEL.get(ggml_level, logging.DEBUG)
    if py_level == logging.ERROR and any(s in text for s in _GGML_ERROR_SOFT_DEMOTE):
        return logging.WARNING
    return py_level


@dataclass
class _LogDispatchSnapshot:
    """Frozen snapshot of dispatcher state for test save/restore."""

    callback: Any
    installed: bool
    pending: dict[int, str]
    pending_level: int


class _LogDispatcher:
    """Owns the mutable state for routing llama.cpp logs through Python logging.

    All four pieces of state (callback handle, installed flag, pending
    buffer, pending-level) live here as instance fields so tests can
    snapshot / restore the whole dispatcher in one call instead of
    poking module globals directly.
    """

    def __init__(self) -> None:
        # ctypes does not retain a Python reference to the wrapped callback;
        # hanging it off the dispatcher keeps it alive for process lifetime.
        self.callback: Any = None
        self.installed: bool = False
        # Request-scoped buffer for a ctypes callback; not on Services because
        # the C side has no notion of dependency injection.
        self.pending: dict[int, str] = {}
        self.pending_level: int = _GGML_LOG_LEVEL_INFO

    def dispatch(self, level: int, text_bytes: bytes, _user_data: Any) -> None:
        """Dispatch one llama.cpp log message; CONT chunks coalesce on newline."""
        try:
            text = text_bytes.decode("utf-8", errors="replace") if text_bytes else ""
        except Exception:  # pragma: no cover
            return

        if level == _GGML_LOG_LEVEL_CONT:
            self.pending[0] = self.pending.get(0, "") + text
        else:
            if 0 in self.pending:
                buffered = self.pending.pop(0).rstrip()
                if buffered:
                    _llama_log.log(_resolve_ggml_level(self.pending_level, buffered), buffered)
            self.pending_level = level
            self.pending[0] = text

        if "\n" in self.pending.get(0, ""):
            full = self.pending.pop(0).rstrip()
            if full:
                _llama_log.log(_resolve_ggml_level(self.pending_level, full), full)

    def install(self) -> None:
        """Route llama.cpp logs through Python logging. Idempotent."""
        if self.installed:
            return
        llama_cpp = import_llama_cpp()
        self.callback = llama_cpp.llama_log_callback(self.dispatch)
        llama_cpp.llama_log_set(self.callback, None)
        self.installed = True

    def snapshot(self) -> _LogDispatchSnapshot:
        """Return a frozen snapshot of dispatcher state for test save/restore."""
        return _LogDispatchSnapshot(
            callback=self.callback,
            installed=self.installed,
            pending=dict(self.pending),
            pending_level=self.pending_level,
        )

    def restore(self, snap: _LogDispatchSnapshot) -> None:
        """Reset dispatcher state to *snap*."""
        self.callback = snap.callback
        self.installed = snap.installed
        self.pending = dict(snap.pending)
        self.pending_level = snap.pending_level

    def reset(self) -> None:
        """Clear dispatcher state (for tests that want a known-clean baseline)."""
        self.callback = None
        self.installed = False
        self.pending.clear()
        self.pending_level = _GGML_LOG_LEVEL_INFO


# Process-lifetime singleton. Tests treat this as the dispatcher boundary.
_dispatcher = _LogDispatcher()


_MISSING_VULKAN_HINT = (
    "lilbee's Linux wheel currently links against libvulkan.so.1 at runtime. "
    "Install your distro's Vulkan loader package (Arch: vulkan-icd-loader, "
    "Fedora: vulkan-loader, Debian/Ubuntu: libvulkan1) and try again. See "
    "https://github.com/tobocop2/lilbee#linux-runtime-requirements."
)


def import_llama_cpp() -> Any:
    """Import ``llama_cpp`` with an actionable error when libvulkan is missing.

    Bare Arch / Fedora installs lack the Vulkan loader and the raw
    ``OSError: libvulkan.so.1: cannot open shared object file`` is opaque.
    Re-raise as :class:`ProviderError` with install guidance. Idempotent
    after the first successful import (Python caches it in ``sys.modules``),
    so callers can sprinkle it at every llama_cpp entry point cheaply.
    """
    _apply_gpu_device_env()
    try:
        import llama_cpp

        return llama_cpp
    except OSError as exc:
        if "libvulkan" in str(exc):
            raise ProviderError(_MISSING_VULKAN_HINT, provider="llama-cpp") from exc
        raise


# Backend env vars set from ``cfg.gpu_devices`` before the first llama_cpp
# import. Vulkan, CUDA, and ROCm each read their own; setting all four
# lets the same config field work across every wheel flavor lilbee ships.
_GPU_VISIBLE_ENV_VARS = (
    "GGML_VK_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)


def _apply_gpu_device_env() -> None:
    """Apply ``cfg.gpu_devices`` (or autodetect) to every backend's visibility env var.

    Must run before ``import llama_cpp``: the Vulkan / CUDA / ROCm
    runtimes snapshot their visible-device lists at library load time
    and ignore subsequent changes. Resolution order:

    1. Explicit env var (``GGML_VK_VISIBLE_DEVICES`` etc.) -- always
       wins, including when the user set it intentionally to empty.
    2. ``cfg.gpu_devices`` -- the user's lilbee-level pin.
    3. Autodetection -- pick the highest-ranked Vulkan device
       (discrete > integrated) when more than one is visible.

    Steps 2 and 3 only set vars that aren't already in the
    environment, so step 1 is preserved.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.llama_cpp.gpu_select import autoselect_best_gpu_index

    devices = cfg.gpu_devices or autoselect_best_gpu_index()
    if not devices:
        return
    for name in _GPU_VISIBLE_ENV_VARS:
        os.environ.setdefault(name, devices)


def install_llama_log_handler() -> None:
    """Route llama.cpp logs through Python logging. Idempotent."""
    _dispatcher.install()


@contextmanager
def stderr_suppressed() -> Iterator[None]:
    """Redirect fd 2 to /dev/null for the duration of the block.

    Holds ``_STDERR_LOCK`` so concurrent fd-2 manipulation can't corrupt
    each other's saved-stderr value. Use this around the *full* embed or
    rerank loop, not per call: per-call wrapping serializes hundreds of
    syscalls + lock acquires inside the inner loop and starves the TUI
    threads of CPU.
    """
    with _STDERR_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)


def suppress_native_stderr(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* with C-level stderr suppressed.

    Thin wrapper around :func:`stderr_suppressed` for one-shot call sites
    (Llama() construction, GGUF metadata read).
    """
    with stderr_suppressed():
        return fn(*args, **kwargs)
