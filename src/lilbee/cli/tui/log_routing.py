"""TUI log file routing: rotating file handler that survives the stderr strip."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lilbee.core.config import cfg

_TUI_LOG_DIR_NAME = "logs"
_TUI_LOG_FILE_NAME = "tui.log"
_MAX_BYTES = 1_048_576  # 1 MiB
_BACKUP_COUNT = 5
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_tui_log_file() -> Path:
    """Install a RotatingFileHandler at ``cfg.data_root/logs/tui.log``. Idempotent."""
    log_dir = cfg.data_root / _TUI_LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _TUI_LOG_FILE_NAME

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path:
            return log_path

    handler = RotatingFileHandler(log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logging.WARNING)
    root.addHandler(handler)
    return log_path
