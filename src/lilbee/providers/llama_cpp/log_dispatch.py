"""llama.cpp log dispatcher and native-stderr suppression.

Routes llama.cpp's C-stream log messages through Python logging, coalescing
CONT continuation chunks until a newline. The pending buffer is module-private
because it tracks request-scoped state for a ctypes callback.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

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
)

_STDERR_LOCK = threading.Lock()

# ctypes does not retain a Python reference to the wrapped callback;
# this module-level handle keeps it alive for the process lifetime.
_llama_log_callback: Any = None
_llama_log_installed = False
# Module-private buffer; not in Services because it's request-scoped state for a ctypes callback.
_llama_log_pending: dict[int, str] = {}
_llama_log_pending_level: int = _GGML_LOG_LEVEL_INFO


def _llama_log_dispatch(level: int, text_bytes: bytes, _user_data: Any) -> None:
    """Dispatch one llama.cpp log message; CONT chunks are coalesced on newline."""
    global _llama_log_pending_level
    try:
        text = text_bytes.decode("utf-8", errors="replace") if text_bytes else ""
    except Exception:  # pragma: no cover
        return

    if level == _GGML_LOG_LEVEL_CONT:
        _llama_log_pending[0] = _llama_log_pending.get(0, "") + text
    else:
        if 0 in _llama_log_pending:
            buffered = _llama_log_pending.pop(0).rstrip()
            if buffered:
                _llama_log.log(_resolve_ggml_level(_llama_log_pending_level, buffered), buffered)
        _llama_log_pending_level = level
        _llama_log_pending[0] = text

    if "\n" in _llama_log_pending.get(0, ""):
        full = _llama_log_pending.pop(0).rstrip()
        if full:
            _llama_log.log(_resolve_ggml_level(_llama_log_pending_level, full), full)


def _resolve_ggml_level(ggml_level: int, text: str) -> int:
    """Translate ggml log level to Python, demoting known-advisory ERRORs to WARNING."""
    py_level = _GGML_TO_PY_LEVEL.get(ggml_level, logging.DEBUG)
    if py_level == logging.ERROR and any(s in text for s in _GGML_ERROR_SOFT_DEMOTE):
        return logging.WARNING
    return py_level


def install_llama_log_handler() -> None:
    """Route llama.cpp logs through Python logging. Idempotent."""
    global _llama_log_callback, _llama_log_installed
    if _llama_log_installed:
        return
    import llama_cpp

    _llama_log_callback = llama_cpp.llama_log_callback(_llama_log_dispatch)
    llama_cpp.llama_log_set(_llama_log_callback, None)
    _llama_log_installed = True


def suppress_native_stderr(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* with C-level stderr suppressed.
    llama.cpp prints noisy messages (e.g. 'init: embeddings required...')
    that bypass Python logging. This redirects fd 2 to /dev/null for the
    duration of the call. A lock serializes access to fd 2 so concurrent
    threads don't corrupt each other's file descriptors.
    """
    with _STDERR_LOCK:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            return fn(*args, **kwargs)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
