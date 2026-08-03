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


def setup_tui_log_file() -> Path:
    """Install a RotatingFileHandler at ``cfg.data_root/logs/tui.log``. Idempotent."""
    log_dir = cfg.data_root / _TUI_LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _TUI_LOG_FILE_NAME
    attach_rotating_file_handler(
        log_path,
        max_bytes=_MAX_BYTES,
        backup_count=_BACKUP_COUNT,
        level=logging.WARNING,
    )
    return log_path
