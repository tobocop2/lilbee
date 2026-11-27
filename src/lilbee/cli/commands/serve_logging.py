"""Server log-file routing for ``lilbee serve``: rotating file under the data root."""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import os
import signal
import sys
import threading
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import IO

from lilbee.core.config import cfg, load_warnings

logger = logging.getLogger(__name__)

_LOG_DIR_NAME = "logs"
_SERVER_LOG_FILE_NAME = "server.log"
_FAULT_LOG_FILE_NAME = "server-fault.log"

_DEADLINE_EXIT_STATUS = 1
_BACKSTOP_LEAD_S = 0.5
_MAX_BYTES = 2_097_152  # 2 MiB
_BACKUP_COUNT = 3
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_UVICORN_ACCESS_LOGGER = "uvicorn.access"
_HEALTH_ACCESS_PATH = "/api/health"


def setup_server_log_file(earlier: Iterable[str] = ()) -> Path:
    """Install a RotatingFileHandler at ``cfg.data_root/logs/server.log``. Idempotent.

    A new handler starts with the *earlier* warnings, logged before any handler existed.
    """
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
    for message in earlier:
        file_handler.handle(
            logging.makeLogRecord(
                {
                    "name": logger.name,
                    "levelno": logging.WARNING,
                    "levelname": "WARNING",
                    "msg": message,
                }
            )
        )
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return log_path


def _fault_log_path() -> Path:
    log_dir = cfg.data_root / _LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _FAULT_LOG_FILE_NAME


class _FaultLog:
    """Holds the fault-log handle alive; faulthandler writes to the raw fd."""

    def __init__(self) -> None:
        self._handle: IO[str] | None = None
        self._alarm_handle: IO[str] | None = None

    def enable(self) -> Path:
        """Open the fault log if needed and point faulthandler at it."""
        fault_path = _fault_log_path()
        if self._handle is None or self._handle.closed:
            self._handle = fault_path.open("a", encoding="utf-8")
            faulthandler.enable(file=self._handle)
        return fault_path

    def dump_on_alarm(self) -> None:
        """Dump every thread to the fault log on SIGALRM; the process then dies of it.

        The first call opens the alarm handle and registers the signal; a later call finds both.
        """
        if self._alarm_handle is not None and not self._alarm_handle.closed:
            return
        with contextlib.suppress(OSError):  # pragma: no cover - POSIX only, no SIGALRM on Windows
            handle = _fault_log_path().open("a", encoding="utf-8")
            faulthandler.register(signal.SIGALRM, file=handle, all_threads=True, chain=True)
            self._alarm_handle = handle


_fault_log = _FaultLog()


def enable_fault_log() -> Path:
    """Point ``faulthandler`` at ``cfg.data_root/logs/server-fault.log``. Idempotent."""
    return _fault_log.enable()


def _exit_at_the_deadline(seconds: float) -> None:
    """Dump every thread to the fault log while this one holds the GIL, then exit."""
    try:
        with _fault_log_path().open("a", encoding="utf-8") as fault_log:
            fault_log.write(f"Timeout ({seconds:g} s)!\n")
            fault_log.flush()
            faulthandler.dump_traceback(file=fault_log, all_threads=True)
    finally:
        os._exit(_DEADLINE_EXIT_STATUS)


class _ExitDeadline:
    """Holds the one live exit timer; arming again cancels the earlier timer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def arm(self, seconds: float) -> None:
        timer = threading.Timer(seconds, _exit_at_the_deadline, args=(seconds,))
        timer.name = "exit-deadline"
        timer.daemon = True
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = timer
            timer.start()


_exit_deadline = _ExitDeadline()


def arm_exit_deadline(seconds: float) -> None:
    """End the process with status 1 after *seconds*; the fault log names what held it.

    A later call replaces the earlier deadline. A kernel timer fires 0.5 s after
    it, so the Python timer thread wins when it can run. The kernel timer needs
    no GIL: it writes every thread to the fault log, then kills the process. It
    names a native call that holds the GIL across the deadline, and a hang in
    finalization.
    """
    _exit_deadline.arm(seconds)
    if sys.platform != "win32":  # pragma: no cover - Windows has no interval timer
        signal.setitimer(signal.ITIMER_REAL, seconds + _BACKSTOP_LEAD_S)
        _fault_log.dump_on_alarm()


class _ExceptHook:
    """Tracks whether the logging excepthook is installed."""

    def __init__(self) -> None:
        self.installed = False

    def install(self) -> None:
        """Install the logging excepthook once, chaining to the previous hook."""
        if self.installed:
            return
        previous = sys.excepthook

        def _hook(
            exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
        ) -> None:
            logger.critical("unhandled exception", exc_info=(exc_type, exc, tb))
            previous(exc_type, exc, tb)

        sys.excepthook = _hook
        self.installed = True


_except_hook = _ExceptHook()


def install_excepthook() -> None:
    """Log unhandled exceptions before the previous hook runs. Idempotent."""
    _except_hook.install()


def _show_info_on_console() -> None:
    """Lower each root console handler to INFO so startup and signal lines show at any CLI level."""
    for handler in logging.getLogger().handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            and handler.level > logging.INFO
        ):
            handler.setLevel(logging.INFO)


def setup_server_logging() -> None:
    """Console, file log, fault log, and excepthook for a ``serve`` process."""
    setup_server_log_file(load_warnings)
    _show_info_on_console()
    enable_fault_log()
    install_excepthook()


class _HealthAccessFilter(logging.Filter):
    """Drop uvicorn access records for the health endpoint."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _HEALTH_ACCESS_PATH not in record.getMessage()


def install_health_access_filter() -> None:
    """Quiet health-poll lines on the uvicorn access logger. Idempotent."""
    access_logger = logging.getLogger(_UVICORN_ACCESS_LOGGER)
    for filt in access_logger.filters:
        if isinstance(filt, _HealthAccessFilter):
            return
    access_logger.addFilter(_HealthAccessFilter())
