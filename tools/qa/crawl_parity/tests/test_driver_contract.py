"""The file names and modes a driver writes are the ones the harness reads."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from tools.qa.crawl_parity import sides
from tools.qa.crawl_parity.model import Mode


def driver_io() -> ModuleType:
    """The drivers' helper module, loaded by path as a driver loads it."""
    path = sides.DRIVERS_DIR / "_driver_io.py"
    spec = importlib.util.spec_from_file_location("_driver_io_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A dataclass looks its own module up by name while the module body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_harness_reads_the_files_a_driver_writes(tmp_path: Path) -> None:
    helper = driver_io()
    assert (helper.PAGES_FILE, helper.RUN_FILE) == (sides.PAGES_FILE, sides.RUN_FILE)
    assert set(helper.MODES) == {mode.value for mode in Mode}
    helper.write_pages(tmp_path, [helper.PageOut("http://h/a?x=1", "é text", None, 2.0)])
    run = helper.RunOut(started=1.0, crawl_started=1.5, crawl_ended=3.0, threads_before=1)
    run.threads_after = 2
    helper.write_run(tmp_path, run)
    pages = sides._read_pages(tmp_path, "http://h")
    assert pages["/a?x=1"].markdown == "é text" and pages["/a?x=1"].saved_at == 2.0
    record = sides._read_run(tmp_path)
    assert record is not None
    assert (record.crawl_started, record.crawl_ended) == (1.5, 3.0)
    assert record.threads_after - record.threads_before == 1
    written = json.loads((tmp_path / sides.RUN_FILE).read_text(encoding="utf-8"))
    assert "native_threads_at_end" in written


def test_a_driver_that_did_not_reach_its_end_has_no_run_record(tmp_path: Path) -> None:
    assert sides._read_run(tmp_path) is None
    assert sides._read_pages(tmp_path, "http://h") == {}


def test_of_two_results_for_one_page_the_one_with_markdown_is_kept(tmp_path: Path) -> None:
    helper = driver_io()
    failed = helper.PageOut("http://h/a", None, "first try failed")
    saved = helper.PageOut("http://h/a", "text", None, 2.0)
    for order in ([failed, saved], [saved, failed]):
        helper.write_pages(tmp_path, order)
        assert sides._read_pages(tmp_path, "http://h")["/a"].markdown == "text"
