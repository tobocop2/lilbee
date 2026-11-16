"""What a successful ``lilbee use-embedder`` prints, returns and saves.

Nothing here patches the adoption itself: the command runs the real settings
boundary, so each test reads the saved setting back from the config and from
``config.toml`` in the data folder. Only the services container is a stand-in,
and its store answers the question "does the index still match this embedder".
"""

from __future__ import annotations

import io
import json
import tomllib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.cli import app
from lilbee.cli.commands import search_chat
from lilbee.cli.commands._shared import REBUILD_HINT
from lilbee.core.config import cfg
from lilbee.data.store import IndexMismatch
from lilbee.runtime.console import PlainConsole
from tests.conftest import make_mock_services

runner = CliRunner()

_REF = "orgA/repoA/built.gguf"
_OTHER_REF = "orgB/repoB/other.gguf"
# Longer than the 80 columns a captured console has, so a wrapped print splits it.
_LONG_REF = "organisation-with-a-long-name/repository-with-a-long-name-GGUF/model-Q4_K_M.gguf"

_MISMATCH = IndexMismatch(
    persisted_model=_OTHER_REF,
    persisted_dim=768,
    current_model=_REF,
    current_dim=384,
    adoptable=False,
    message="This index needs a rebuild.",
)

_CYAN = "\x1b[36m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"


def _flat(output: str) -> str:
    return " ".join(output.split())


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the config at a scratch folder and restore every field after the test."""
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.json_mode = False
    cfg.embedding_model = _OTHER_REF
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _install_services(*, has_chunks: bool = False, mismatch: IndexMismatch | None = None) -> None:
    """Install a container whose model is on disk and whose store reports *mismatch*."""
    store = MagicMock()
    store.has_chunks.return_value = has_chunks
    store.index_mismatch.return_value = mismatch
    manager = MagicMock()
    manager.is_installed.return_value = True
    svc_mod.set_services(make_mock_services(store=store, model_manager=manager))


@pytest.fixture(autouse=True)
def _services() -> Iterator[None]:
    _install_services()
    yield
    svc_mod.set_services(None)


def _saved_embedder(folder: Path) -> str:
    return tomllib.loads((folder / "config.toml").read_text(encoding="utf-8"))["embedding_model"]


class TestSuccessText:
    def test_the_first_line_names_the_adopted_model_and_ends_with_a_period(self, tmp_path):
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert result.output.splitlines()[0] == f"Now embedding with {_REF}."

    def test_a_ref_longer_than_the_console_prints_on_one_line(self, tmp_path):
        result = runner.invoke(app, ["use-embedder", _LONG_REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [f"Now embedding with {_LONG_REF}."]

    def test_no_rebuild_hint_prints_when_the_index_matches_the_new_embedder(self, tmp_path):
        _install_services(has_chunks=True, mismatch=None)
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [f"Now embedding with {_REF}."]

    def test_the_rebuild_hint_follows_the_success_line_when_the_index_does_not_match(
        self, tmp_path
    ):
        _install_services(has_chunks=True, mismatch=_MISMATCH)
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [f"Now embedding with {_REF}.", REBUILD_HINT]

    def test_the_rebuild_hint_stays_on_one_line_at_a_narrow_width(self, tmp_path, monkeypatch):
        _install_services(has_chunks=True, mismatch=_MISMATCH)
        monkeypatch.setattr(search_chat, "console", PlainConsole(width=20))
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert REBUILD_HINT in result.output.splitlines()


class TestSuccessColours:
    """The model name is cyan and the rebuild hint is yellow on a terminal."""

    @pytest.fixture
    def terminal(self, monkeypatch) -> io.StringIO:
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("NO_COLOR", raising=False)
        buffer = io.StringIO()
        monkeypatch.setattr(
            search_chat, "console", PlainConsole(file=buffer, force_terminal=True, width=200)
        )
        return buffer

    def test_the_model_name_is_cyan_and_the_rest_of_the_line_is_plain(self, tmp_path, terminal):
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.exit_code
        line = terminal.getvalue().splitlines()[0]
        assert f"Now embedding with {_CYAN}{_REF}\x1b[0m." in line
        assert _YELLOW not in terminal.getvalue()

    def test_the_rebuild_hint_is_yellow(self, tmp_path, terminal):
        _install_services(has_chunks=True, mismatch=_MISMATCH)
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.exit_code
        assert f"{_YELLOW}{REBUILD_HINT}\x1b[0m" in terminal.getvalue().splitlines()[1]


class TestSuccessJson:
    def test_a_new_embedder_reports_adopted_with_no_rebuild(self, tmp_path):
        result = runner.invoke(app, ["--json", "use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "command": "use-embedder",
            "model": _REF,
            "status": "adopted",
            "reindex_required": False,
        }

    def test_the_embedder_already_in_use_reports_already_active(self, tmp_path):
        cfg.embedding_model = _REF
        result = runner.invoke(app, ["--json", "use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["status"] == "already_active"
        assert payload["model"] == _REF

    def test_a_mismatched_index_reports_reindex_required_and_prints_no_hint(self, tmp_path):
        _install_services(has_chunks=True, mismatch=_MISMATCH)
        result = runner.invoke(app, ["--json", "use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["reindex_required"] is True
        assert REBUILD_HINT not in result.output
        assert "Now embedding" not in result.output


class TestSavedSetting:
    def test_the_embedder_lands_in_the_config_and_in_the_data_folders_toml(self, tmp_path):
        folder = tmp_path / "chosen"
        folder.mkdir()
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(folder)])
        assert result.exit_code == 0, result.output
        assert cfg.embedding_model == _REF
        assert _saved_embedder(folder) == _REF
        assert not (tmp_path / "config.toml").exists()

    def test_data_dir_picks_the_folder_that_the_command_works_in(self, tmp_path):
        folder = tmp_path / "chosen"
        folder.mkdir()
        result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(folder)])
        assert result.exit_code == 0, result.output
        assert cfg.data_root == folder

    def test_global_picks_the_global_folder_and_saves_the_embedder_there(
        self, tmp_path, monkeypatch
    ):
        global_root = tmp_path / "global"
        global_root.mkdir()
        monkeypatch.setattr("lilbee.core.system.default_data_dir", lambda: global_root)
        result = runner.invoke(app, ["use-embedder", _REF, "--global"])
        assert result.exit_code == 0, result.output
        assert cfg.data_root == global_root
        assert _saved_embedder(global_root) == _REF

    def test_global_and_data_dir_together_are_refused_before_anything_is_saved(self, tmp_path):
        folder = tmp_path / "chosen"
        folder.mkdir()
        result = runner.invoke(app, ["use-embedder", _REF, "--global", "--data-dir", str(folder)])
        assert result.exit_code != 0
        assert "Cannot use --global with --data-dir" in _flat(result.output)
        assert not (folder / "config.toml").exists()
        assert cfg.embedding_model == _OTHER_REF


class TestFailureLine:
    """The line a refused adoption prints starts with a red ``Error: `` prefix."""

    def test_the_line_reads_error_prefix_then_the_reason_and_the_exit_code_is_one(self, tmp_path):
        with patch("lilbee.app.models.adopt_embedder", side_effect=RuntimeError("no such model")):
            result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert result.output.splitlines() == ["Error: no such model"]

    def test_the_prefix_is_red_and_the_reason_is_plain(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("NO_COLOR", raising=False)
        buffer = io.StringIO()
        monkeypatch.setattr(
            search_chat, "console", PlainConsole(file=buffer, force_terminal=True, width=200)
        )
        with patch("lilbee.app.models.adopt_embedder", side_effect=RuntimeError("no such model")):
            result = runner.invoke(app, ["use-embedder", _REF, "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert buffer.getvalue().splitlines() == [f"{_RED}Error: \x1b[0mno such model"]
