"""Tests for the `lilbee export` / `lilbee import` CLI commands."""

from __future__ import annotations

import _thread
import errno
import json
import threading
from contextlib import contextmanager
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.cli import app
from lilbee.core.config import active_config, cfg
from lilbee.data.store import Store
from lilbee.runtime.lock import ResetRefusedError, _no_sync_running
from tests.conftest import make_mock_services, read_takes_hooked, reset_held_on_a_thread

runner = CliRunner()
# How long a held reset lasts when a test hangs; no test waits for it.
_HANG_LIMIT_S = 30.0
_ONE_PAGE = '{"source":"doc.pdf","page":1,"text":"body","content_type":"pdf"}\n'


class _StubEmbedder:
    truncated_total = 0

    def embed_batch(self, texts, **_kwargs):
        return [[0.1] * cfg.embedding_dim for _ in texts]


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    """A data root of this test alone; a command takes its locks there.

    Each command resolves the data root from LILBEE_DATA, which CI sets one time
    for every test worker.
    """
    root = tmp_path / "library"
    monkeypatch.setenv("LILBEE_DATA", str(root))
    return root


@pytest.fixture()
def store(tmp_path):
    real = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb"}))
    svc_mod.set_services(make_mock_services(store=real, embedder=_StubEmbedder()))
    yield real
    svc_mod.set_services(None)


def _seed(store, source="doc.pdf"):
    store.add_page_texts(
        [
            {"source": source, "page": 1, "text": "page one", "content_type": "pdf"},
            {"source": source, "page": 2, "text": "page two", "content_type": "pdf"},
        ]
    )
    store.upsert_source(source, "h", 2)


class TestExportCommand:
    def test_export_writes_file(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.parquet"
        result = runner.invoke(app, ["--json", "export", str(out)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == {
            "command": "export",
            "format": "parquet",
            "output": str(out),
            "pages": 2,
            "sources": 1,
        }
        assert out.exists()

    def test_export_jsonl_by_suffix(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        result = runner.invoke(app, ["export", str(out)])
        assert result.exit_code == 0, result.output
        assert "Wrote 2 pages" in result.output
        assert out.exists()

    def test_a_cancelled_export_reports_it_and_leaves_no_file(self, store, tmp_path, monkeypatch):
        """Ctrl-C stops between row groups; half a dataset must not be left behind."""
        from lilbee.runtime.cancellation import TaskCancelledError

        _seed(store)
        out = tmp_path / "pages.parquet"

        def cancelled_export(*_args, **_kwargs):
            raise TaskCancelledError

        monkeypatch.setattr("lilbee.app.dataset.export_to_path", cancelled_export)
        result = runner.invoke(app, ["export", str(out)])

        assert result.exit_code == 1
        assert "cancelled" in result.output.lower()
        assert not out.exists()

    def test_export_empty_store_fails(self, store, tmp_path):
        result = runner.invoke(app, ["--json", "export", str(tmp_path / "pages.parquet")])
        assert result.exit_code == 1
        assert "error" in json.loads(result.output)

    def test_export_empty_store_human_output(self, store, tmp_path):
        result = runner.invoke(app, ["export", str(tmp_path / "pages.parquet")])
        assert result.exit_code == 1
        assert "Nothing to export" in result.output

    def test_export_bad_format_fails(self, store, tmp_path):
        _seed(store)
        result = runner.invoke(
            app, ["--json", "export", str(tmp_path / "pages.txt"), "--format", "csv"]
        )
        assert result.exit_code == 1
        assert "Unsupported format" in json.loads(result.output)["error"]

    def test_export_unknown_source_fails(self, store, tmp_path):
        _seed(store)
        result = runner.invoke(
            app, ["--json", "export", str(tmp_path / "p.parquet"), "--source", "missing.pdf"]
        )
        assert result.exit_code == 1
        assert "Source not found" in json.loads(result.output)["error"]

    def test_export_bracketed_path_prints_it_literally(self, store, tmp_path):
        """The output path is user-chosen and must not go through markup."""
        _seed(store)
        out = tmp_path / "note[draft].jsonl"
        result = runner.invoke(app, ["export", str(out)])
        assert result.exit_code == 0, result.output
        assert f"to {out}" in result.output

    def test_export_unknown_bracketed_source_prints_it_literally(self, store, tmp_path):
        _seed(store)
        result = runner.invoke(
            app, ["export", str(tmp_path / "p.parquet"), "--source", "mis[sing].pdf"]
        )
        assert result.exit_code == 1
        assert "Error: Source not found: mis[sing].pdf" in result.output


class TestImportCommand:
    def test_round_trip(self, store, tmp_path, data_root):
        _seed(store)
        out = tmp_path / "pages.parquet"
        assert runner.invoke(app, ["export", str(out)]).exit_code == 0

        # Fresh store for the import target.
        target = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb2"}))
        svc_mod.set_services(make_mock_services(store=target, embedder=_StubEmbedder()))

        result = runner.invoke(app, ["--json", "import", str(out)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["command"] == "import"
        assert payload["sources"] == ["doc.pdf"]
        assert payload["pages"] == 2
        assert payload["chunks"] > 0
        assert {r["page"] for r in target.get_page_texts("doc.pdf")} == {1, 2}
        # The import marked its sync on the data root of this test.
        assert (data_root / "sync.lock").exists()

    def test_round_trip_human_output(self, store, tmp_path):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        assert runner.invoke(app, ["export", str(out)]).exit_code == 0

        target = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb2"}))
        svc_mod.set_services(make_mock_services(store=target, embedder=_StubEmbedder()))

        result = runner.invoke(app, ["import", str(out)])
        assert result.exit_code == 0, result.output
        assert "Imported 1 source(s)" in result.output

    def test_one_ctrl_c_ends_an_import_that_waits_for_a_reset(self, store, tmp_path, data_root):
        _seed(store)
        out = tmp_path / "pages.jsonl"
        assert runner.invoke(app, ["export", str(out)]).exit_code == 0
        target = Store(cfg.model_copy(update={"lancedb_dir": tmp_path / "lancedb2"}))
        svc_mod.set_services(make_mock_services(store=target, embedder=_StubEmbedder()))
        assert threading.current_thread() is threading.main_thread()
        assert active_config().data_root == data_root
        interrupts: list[str] = []

        @contextmanager
        def _ctrl_c_once(_lock, _timeout):
            if not interrupts:
                interrupts.append("ctrl-c")
                _thread.interrupt_main()
            yield

        # The reset holds until this block ends; its own limit only ends a test that hangs.
        with (
            reset_held_on_a_thread(data_root, ends_after_s=_HANG_LIMIT_S) as resetter,
            read_takes_hooked(_ctrl_c_once),
        ):
            result = runner.invoke(app, ["import", str(out)])
            reset_still_held = resetter.is_alive()
            with (
                pytest.raises(ResetRefusedError, match="is running on this library"),
                _no_sync_running(data_root),
            ):
                pytest.fail("the reset ended before the import did")

        assert interrupts == ["ctrl-c"]
        assert reset_still_held, "the import ended only when the reset did"
        assert result.exit_code != 0, result.output
        assert target.get_page_texts("doc.pdf") == []
        with _no_sync_running(data_root):
            pass

    def test_import_embedder_mismatch_fails(self, store, tmp_path):
        out = tmp_path / "pages.jsonl"
        out.write_text('{"source":"doc.pdf","page":1,"text":"body","content_type":"pdf"}\n')
        # Seed the store under one embedder, then drift the model so import is refused.
        store.add_chunks(
            [
                {
                    "source": "seed.pdf",
                    "content_type": "pdf",
                    "chunk_type": "raw",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "seed",
                    "chunk_index": 0,
                    "vector": [0.1] * cfg.embedding_dim,
                }
            ]
        )
        store._config.embedding_model = "ollama/other-model:v1"
        result = runner.invoke(app, ["--json", "import", str(out)])
        assert result.exit_code == 1
        assert "error" in json.loads(result.output)

    def test_import_missing_file_fails(self, store, tmp_path):
        result = runner.invoke(app, ["--json", "import", str(tmp_path / "nope.parquet")])
        assert result.exit_code == 1
        assert "Dataset not found" in json.loads(result.output)["error"]

    def test_import_empty_dataset_fails(self, store, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["--json", "import", str(empty)])
        assert result.exit_code == 1
        assert "no pages" in json.loads(result.output)["error"]

    def test_an_import_that_cannot_open_its_sync_lock_exits_1_and_imports_nothing(
        self, store, tmp_path, data_root
    ):
        out = tmp_path / "pages.jsonl"
        out.write_text(_ONE_PAGE, encoding="utf-8")
        refused = OSError(errno.EMFILE, "Too many open files")
        with mock.patch("lilbee.runtime.lock.AsyncReadWriteLock", side_effect=refused):
            result = runner.invoke(app, ["import", str(out)])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        output = " ".join(result.output.split())
        assert output.startswith("Error: Cannot change the library:")
        assert "cannot be locked ([Errno 24] Too many open files)." in output
        assert "Close some programs or raise the limit of open files, then try again." in output
        assert "Traceback" not in result.output
        assert store.get_page_texts("doc.pdf") == []

    def test_an_import_that_cannot_open_its_sync_lock_reports_it_as_json(
        self, store, tmp_path, data_root
    ):
        out = tmp_path / "pages.jsonl"
        out.write_text(_ONE_PAGE, encoding="utf-8")
        refused = OSError(errno.EMFILE, "Too many open files")
        with mock.patch("lilbee.runtime.lock.AsyncReadWriteLock", side_effect=refused):
            result = runner.invoke(app, ["--json", "import", str(out)])

        assert result.exit_code == 1
        assert json.loads(result.output) == {
            "error": (
                f"Cannot change the library: {data_root / 'sync.lock'} cannot be locked "
                "([Errno 24] Too many open files). "
                "Close some programs or raise the limit of open files, then try again."
            )
        }
        assert store.get_page_texts("doc.pdf") == []
