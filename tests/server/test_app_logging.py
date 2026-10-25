"""Building the server app leaves process logging and threads as they were."""

import logging
import threading

from lilbee.server.app import create_app

_BUILDS = 5


def test_building_the_app_starts_no_thread():
    before = set(threading.enumerate())

    for _ in range(_BUILDS):
        create_app()

    started = [thread.name for thread in set(threading.enumerate()) - before]
    assert started == []


def test_building_the_app_keeps_an_attached_file_handler_open(tmp_path):
    log_path = tmp_path / "attached.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        create_app()
        logging.getLogger("lilbee.test_app_logging").error("after the build")
        handler.flush()
        assert handler.stream is not None
        assert "after the build" in log_path.read_text(encoding="utf-8")
        assert handler in root.handlers
    finally:
        root.removeHandler(handler)
        handler.close()


def test_building_the_app_keeps_the_root_logger_level():
    root = logging.getLogger()
    level = root.level
    handlers = list(root.handlers)
    try:
        root.setLevel(logging.WARNING)

        create_app()

        assert root.level == logging.WARNING
        assert root.handlers == handlers
    finally:
        root.setLevel(level)
