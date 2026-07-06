"""Tests for the ``lilbee memory`` CLI command group."""

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.app.memory import MEMORY_DISABLED_HINT
from lilbee.cli.app import app
from lilbee.core.config import cfg
from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemoryRow, MemorySource
from tests.conftest import make_mock_services

runner = CliRunner()


def _row(text: str, *, kind: MemoryKind = MemoryKind.FACT, shared: bool = False) -> MemoryRow:
    return MemoryRow(
        id="abcdef0123456789",
        owner=LOCAL_OWNER,
        shared=shared,
        kind=kind,
        source=MemorySource.MANUAL,
        text=text,
        vector=[0.1],
        created_at="t",
        updated_at="t",
    )


@pytest.fixture(autouse=True)
def mock_svc():
    store = MagicMock()
    store.add_memory.return_value = "id123"
    store.get_memories.return_value = []
    store.search_memories.return_value = []
    store.delete_memory.return_value = True
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    services = make_mock_services(store=store, embedder=embedder)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.json_mode = False
    cfg.memory_enabled = False
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestDisabled:
    def test_add_disabled_shows_hint(self):
        result = runner.invoke(app, ["memory", "add", "hi"])
        assert result.exit_code == 0
        assert "Memory is off." in result.output  # Rich may wrap the full line

    def test_add_disabled_json(self):
        result = runner.invoke(app, ["--json", "memory", "add", "hi"])
        assert json.loads(result.output)["error"] == MEMORY_DISABLED_HINT

    def test_list_disabled_shows_hint(self, mock_svc):
        result = runner.invoke(app, ["memory", "list"])
        assert "Memory is off." in result.output
        mock_svc.store.get_memories.assert_not_called()

    def test_recall_disabled_shows_hint(self, mock_svc):
        result = runner.invoke(app, ["memory", "recall", "q"])
        assert "Memory is off." in result.output
        mock_svc.store.search_memories.assert_not_called()

    def test_remove_disabled_shows_hint(self, mock_svc):
        result = runner.invoke(app, ["memory", "remove", "abc"])
        assert "Memory is off." in result.output
        mock_svc.store.delete_memory.assert_not_called()


class TestAdd:
    def test_add_fact(self, mock_svc):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["--json", "memory", "add", "uses rust"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload == {"id": "id123", "kind": "fact"}
        mock_svc.store.add_memory.assert_called_once()

    def test_add_preference_shared(self, mock_svc):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["--json", "memory", "add", "be terse", "-p", "--shared"])
        assert json.loads(result.output)["kind"] == "preference"
        record = mock_svc.store.add_memory.call_args.args[0]
        assert record.kind is MemoryKind.PREFERENCE
        assert record.shared is True

    def test_add_console(self):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["memory", "add", "x"])
        assert "Remembered (fact)." in result.output


class TestList:
    def test_empty(self):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["memory", "list"])
        assert "No memories stored." in result.output

    def test_table(self, mock_svc):
        cfg.memory_enabled = True
        mock_svc.store.get_memories.return_value = [_row("uses rust", shared=True)]
        result = runner.invoke(app, ["memory", "list"])
        assert "uses rust" in result.output

    def test_json(self, mock_svc):
        cfg.memory_enabled = True
        mock_svc.store.get_memories.return_value = [_row("uses rust")]
        result = runner.invoke(app, ["--json", "memory", "list"])
        payload = json.loads(result.output)
        assert payload["memories"][0]["text"] == "uses rust"


class TestRecall:
    def test_empty(self):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["memory", "recall", "q"])
        assert "No relevant memories." in result.output

    def test_results(self, mock_svc):
        cfg.memory_enabled = True
        mock_svc.store.search_memories.return_value = [_row("uses rust")]
        result = runner.invoke(app, ["memory", "recall", "q"])
        assert "- uses rust" in result.output

    def test_json(self, mock_svc):
        cfg.memory_enabled = True
        mock_svc.store.search_memories.return_value = [_row("uses rust")]
        result = runner.invoke(app, ["--json", "memory", "recall", "q"])
        assert json.loads(result.output)["memories"][0]["text"] == "uses rust"


class TestRemove:
    def test_remove(self, mock_svc):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["memory", "remove", "abc"])
        assert "Removed abc." in result.output
        mock_svc.store.delete_memory.assert_called_once_with("abc", owner=LOCAL_OWNER)

    def test_remove_json(self):
        cfg.memory_enabled = True
        result = runner.invoke(app, ["--json", "memory", "remove", "abc"])
        assert json.loads(result.output) == {"id": "abc", "deleted": True}

    def test_remove_unknown_id_reports_not_found(self, mock_svc):
        cfg.memory_enabled = True
        mock_svc.store.delete_memory.return_value = False
        result = runner.invoke(app, ["memory", "remove", "ghost"])
        assert "No memory ghost found." in result.output
        assert result.exit_code == 1  # not-found exits non-zero, like `model remove`

    def test_remove_unknown_id_json_exits_nonzero(self, mock_svc):
        cfg.memory_enabled = True
        mock_svc.store.delete_memory.return_value = False
        result = runner.invoke(app, ["--json", "memory", "remove", "ghost"])
        assert json.loads(result.output) == {"id": "ghost", "deleted": False}
        assert result.exit_code == 1
