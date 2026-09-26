"""TUI log file routing: rotating file handler that survives the stderr strip."""

from __future__ import annotations

import logging
from pathlib import Path

from lilbee.cli.log_routing import attach_rotating_file_handler
from lilbee.core.config import cfg

_TUI_LOG_DIR_NAME = "logs"
_TUI_LOG_FILE_NAME = "tui.log"
_MAX_BYTES = 1_048_576  # 1 MiB
_BACKUP_COUNT = 5


def tui_log_path() -> Path:
    """Where the TUI's log file is, or will be, at ``cfg.data_root/logs/tui.log``."""
    return cfg.data_root / _TUI_LOG_DIR_NAME / _TUI_LOG_FILE_NAME


def setup_tui_log_file() -> Path:
    """Install a RotatingFileHandler at ``cfg.data_root/logs/tui.log``. Idempotent."""
    log_path = tui_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attach_rotating_file_handler(
        log_path,
        max_bytes=_MAX_BYTES,
        backup_count=_BACKUP_COUNT,
        level=logging.WARNING,
    )
    return log_path
