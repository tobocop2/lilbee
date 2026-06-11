"""Server log-file routing for ``lilbee serve``: rotating file under the data root."""

from __future__ import annotations

import faulthandler
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import IO

from lilbee.core.config import cfg

logger = logging.getLogger(__name__)

_LOG_DIR_NAME = "logs"
_SERVER_LOG_FILE_NAME = "server.log"
_FAULT_LOG_FILE_NAME = "server-fault.log"
_MAX_BYTES = 2_097_152  # 2 MiB
_BACKUP_COUNT = 3
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_server_log_file() -> Path:
    """Install a RotatingFileHandler at ``cfg.data_root/logs/server.log``. Idempotent."""
    log_dir = cfg.data_root / _LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _SERVER_LOG_FILE_NAME

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path:
            return log_path

    file_handler = RotatingFileHandler(log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)
    if root.level > logging.INFO:
        # NOTSET stream handlers delegate to root; pin them so stderr verbosity is unchanged.
        for handler in root.handlers:
            if handler is not file_handler and handler.level == logging.NOTSET:
                handler.setLevel(root.level)
        root.setLevel(logging.INFO)
    return log_path


class _FaultLog:
    """Holds the fault-log handle alive; faulthandler writes to the raw fd."""

    def __init__(self) -> None:
        self._handle: IO[str] | None = None

    def enable(self) -> Path:
        """Open the fault log if needed and point faulthandler at it."""
        log_dir = cfg.data_root / _LOG_DIR_NAME
        log_dir.mkdir(parents=True, exist_ok=True)
        fault_path = log_dir / _FAULT_LOG_FILE_NAME
        if self._handle is None or self._handle.closed:
            self._handle = fault_path.open("a")
            faulthandler.enable(file=self._handle)
        return fault_path


_fault_log = _FaultLog()


def enable_fault_log() -> Path:
    """Point ``faulthandler`` at ``cfg.data_root/logs/server-fault.log``. Idempotent."""
    return _fault_log.enable()


def install_excepthook() -> None:
    """Log unhandled exceptions before the previous hook runs."""
    previous = sys.excepthook

    def _hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        logger.critical("unhandled exception", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = _hook
