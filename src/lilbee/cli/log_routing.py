"""CLI diagnostics routing: keep answer output clean, log records go to a file."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lilbee.core.config import cfg

_LOG_DIR_NAME = "logs"
_CLI_LOG_FILE_NAME = "cli.log"
_MAX_BYTES = 1_048_576  # 1 MiB
_BACKUP_COUNT = 5
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

class _DiagnosticsRouter:
    """Holds whether the user asked for a log level this invocation."""

    def __init__(self) -> None:
        self.explicit_verbosity = False


_router = _DiagnosticsRouter()


def set_explicit_verbosity(explicit: bool) -> None:
    """Record whether the user requested a log level for this invocation."""
    _router.explicit_verbosity = explicit


def attach_rotating_file_handler(
    log_path: Path,
    *,
    max_bytes: int = _MAX_BYTES,
    backup_count: int = _BACKUP_COUNT,
    level: int = logging.NOTSET,
) -> None:
    """Add a RotatingFileHandler for *log_path* to the root logger. Idempotent."""
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path:
            return
    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(level)
    root.addHandler(handler)


def route_diagnostics_to_log_file() -> Path | None:
    """Send log records and Python warnings to ``cfg.data_root/logs/cli.log``.

    Answer-facing commands (ask, search, chat) call this after their overrides
    resolve the data root, so stdout carries only the answer. Warnings become
    ``py.warnings`` log records via ``logging.captureWarnings``; nothing is
    filtered and no logger level changes. Skipped entirely when --log-level or
    LILBEE_LOG_LEVEL was given: diagnostics then stay on stderr as usual.
    """
    if _router.explicit_verbosity:
        return None
    log_dir = cfg.data_root / _LOG_DIR_NAME
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    log_path = log_dir / _CLI_LOG_FILE_NAME
    attach_rotating_file_handler(log_path)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and handler.stream in (sys.stderr, sys.stdout)
        ):
            root.removeHandler(handler)
    logging.captureWarnings(True)
    return log_path
