"""Server log-file routing for ``lilbee serve``: rotating file under the data root."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lilbee.core.config import cfg

_LOG_DIR_NAME = "logs"
_SERVER_LOG_FILE_NAME = "server.log"
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
