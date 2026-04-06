"""Tests for the CLI interface using typer's test runner."""

import json
import logging
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

import lilbee.services as svc_mod
from lilbee.cli import (
    app,
    clean_result,
    get_version,
)
from lilbee.config import cfg
from lilbee.ingest import SyncResult
from lilbee.models import list_installed_models
from lilbee.store import SearchChunk

runner = CliRunner()

_SYNC_NOOP = SyncResult()


def _mock_stream(*texts: str):
    from lilbee.reasoning import StreamToken

    return iter([StreamToken(content=t, is_reasoning=False) for t in texts])


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """CLI tests never need real model validation or chat model checks."""
    with mock.patch("lilbee.models.ensure_chat_model"):
        yield


@pytest.fixture(autouse=True)
def mock_svc():
    """Provide a mock Services container for all CLI tests."""
    from tests.conftest import make_mock_services

    searcher = MagicMock()
    searcher.search.return_value = []
    searcher.ask_stream.return_value = _mock_stream("")
    store = MagicMock()
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.add_chunks.return_value = 0
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    embedder.embed_batch.return_value = []
    services = make_mock_services(searcher=searcher, store=store, embedder=embedder)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Redirect config paths for all CLI tests."""
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    monkeypatch.delenv("LILBEE_LOG_LEVEL", raising=False)
    snapshot = cfg.model_copy()
    root = logging.getLogger()
    old_level = root.level
    old_handlers = root.handlers[:]

    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.json_mode = False
    cfg.concept_graph = False

    yield tmp_path

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
    root.setLevel(old_level)
    root.handlers[:] = old_handlers


class TestStatus:
    def test_empty_status(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No documents indexed" in result.output

    def test_status_shows_paths(self):
        result = runner.invoke(app, ["status"])
        assert "Documents:" in result.output
        assert "Database:" in result.output

    def test_status_shows_models(self):
        result = runner.invoke(app, ["status"])
        assert "Chat model:" in result.output
        assert "Embeddings:" in result.output

    def test_status_shows_vision_model_when_set(self):
        cfg.vision_model = "test-vision:latest"
        result = runner.invoke(app, ["status"])
        assert "Vision OCR:" in result.output
        assert "test-vision:latest" in result.output

    def test_status_hides_vision_model_when_empty(self):
        cfg.vision_model = ""
        result = runner.invoke(app, ["status"])
        assert "Vision OCR:" not in result.output

    def test_status_with_indexed_docs(self, isolated_env, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {
                "filename": "test.pdf",
                "file_hash": "abc123",
                "chunk_count": 10,
                "ingested_at": "2026-01-01T00:00:00",
            }
        ]
        result = runner.invoke(app, ["status"])
        assert "test.pdf" in result.output
        assert "10" in result.output


class TestSync:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_sync_empty(self, mock_sync):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 0" in result.output

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    def test_sync_with_file(self, mock_sync, isolated_env):

        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 1" in result.output

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(failed=["bad.txt"]),
    )
    def test_sync_shows_failed(self, mock_sync):
        result = runner.invoke(app, ["sync"])
        assert "Failed: 1" in result.output
        assert "bad.txt" in result.output


class TestRebuild:
    def test_rebuild_empty(self):
        result = runner.invoke(app, ["rebuild"])
        assert result.exit_code == 0
        assert "Rebuilt:" in result.output


class TestAdd:
    def test_add_single_file(self, isolated_env, tmp_path):
        """Adding a single file copies it and ingests it."""
        src_file = tmp_path / "source" / "manual.txt"
        src_file.parent.mkdir()
        src_file.write_text("Engine oil capacity is 5 quarts.")

        result = runner.invoke(app, ["add", str(src_file)])
        assert result.exit_code == 0
        assert "Copied 1" in result.output
        assert (cfg.documents_dir / "manual.txt").exists()

    def test_add_directory(self, isolated_env, tmp_path):
        """Adding a directory recursively copies it."""
        src_dir = tmp_path / "source" / "docs"
        src_dir.mkdir(parents=True)
        (src_dir / "file1.txt").write_text("Content 1")
        (src_dir / "file2.txt").write_text("Content 2")

        result = runner.invoke(app, ["add", str(src_dir)])
        assert result.exit_code == 0
        assert (cfg.documents_dir / "docs" / "file1.txt").exists()
        assert (cfg.documents_dir / "docs" / "file2.txt").exists()

    def test_add_multiple_paths(self, isolated_env, tmp_path):
        """Adding multiple paths works."""
        f1 = tmp_path / "source" / "a.txt"
        f2 = tmp_path / "source" / "b.txt"
        f1.parent.mkdir()
        f1.write_text("File A")
        f2.write_text("File B")

        result = runner.invoke(app, ["add", str(f1), str(f2)])
        assert result.exit_code == 0
        assert "Copied 2" in result.output

    def test_add_nonexistent_fails(self):
        """Adding a nonexistent path fails."""
        result = runner.invoke(app, ["add", "/tmp/nonexistent_file_xyz.txt"])
        assert result.exit_code != 0

    def test_add_overwrites_existing_dir(self, isolated_env, tmp_path):
        """Re-adding a directory with --force updates content."""

        src_dir = tmp_path / "source" / "docs"
        src_dir.mkdir(parents=True)
        (src_dir / "file1.txt").write_text("Version 1")

        runner.invoke(app, ["add", "--force", str(src_dir)])

        # Update content and re-add with --force
        (src_dir / "file1.txt").write_text("Version 2")
        result = runner.invoke(app, ["add", "--force", str(src_dir)])
        assert result.exit_code == 0
        assert (cfg.documents_dir / "docs" / "file1.txt").read_text() == "Version 2"

    def test_add_warns_on_existing(self, isolated_env, tmp_path):
        """Adding a file that already exists warns without --force."""
        src_file = tmp_path / "source" / "manual.txt"
        src_file.parent.mkdir()
        src_file.write_text("Original content")

        runner.invoke(app, ["add", "--force", str(src_file)])

        src_file.write_text("New content")
        result = runner.invoke(app, ["add", str(src_file)])
        assert result.exit_code == 0
        assert "Warning" in result.output
        assert "already exists" in result.output


class TestAddIgnoresDirs:
    def test_add_directory_skips_git_and_node_modules(self, isolated_env, tmp_path):
        """Adding a directory filters out .git/ and node_modules/."""

        src_dir = tmp_path / "source" / "project"
        src_dir.mkdir(parents=True)
        (src_dir / "readme.txt").write_text("Real content")
        (src_dir / ".git").mkdir()
        (src_dir / ".git" / "config").write_text("git stuff")
        (src_dir / "node_modules").mkdir()
        (src_dir / "node_modules" / "pkg.txt").write_text("npm junk")
        (src_dir / "__pycache__").mkdir()
        (src_dir / "__pycache__" / "mod.pyc").write_bytes(b"\x00")

        result = runner.invoke(app, ["add", str(src_dir)])
        assert result.exit_code == 0

        dest = cfg.documents_dir / "project"
        assert (dest / "readme.txt").exists()
        assert not (dest / ".git").exists()
        assert not (dest / "node_modules").exists()
        assert not (dest / "__pycache__").exists()


class TestAsk:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_prints_response(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_stream.return_value = _mock_stream("Hello", " world")
        result = runner.invoke(app, ["ask", "test question"])
        assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_with_model_flag(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_stream.return_value = _mock_stream("answer")
        result = runner.invoke(app, ["ask", "question", "--model", "llama3"])
        assert result.exit_code == 0


class TestDataDirFlag:
    def test_status_with_data_dir(self, tmp_path):
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "documents").mkdir()
        result = runner.invoke(app, ["status", "--data-dir", str(custom)])
        assert result.exit_code == 0

    def test_sync_with_data_dir(self, tmp_path):
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "documents").mkdir()
        result = runner.invoke(app, ["sync", "--data-dir", str(custom)])
        assert result.exit_code == 0


class TestAutoSync:
    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["new.pdf"]),
    )
    def test_auto_sync_prints_summary(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_stream.return_value = _mock_stream("answer")
        result = runner.invoke(app, ["ask", "test"])
        assert result.exit_code == 0
        assert "Synced:" in result.output

    def test_auto_sync_background(self) -> None:
        from rich.console import Console

        from lilbee.cli.helpers import auto_sync

        con = Console()
        with mock.patch("lilbee.cli.sync.run_sync_background") as mock_bg:
            auto_sync(con, background=True)
            mock_bg.assert_called_once_with(con)


class TestAddPathsBackground:
    def test_add_paths_background_mode(self, isolated_env, tmp_path) -> None:
        from rich.console import Console

        from lilbee.cli.helpers import add_paths

        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        con = Console()
        with (
            mock.patch("lilbee.cli.sync.run_sync_background") as mock_bg,
            mock.patch("lilbee.cli.helpers.copy_paths", return_value=[src]),
        ):
            add_paths([src], con, background=True)
            mock_bg.assert_called_once()

    def test_add_paths_chat_mode_prints(self, isolated_env, tmp_path, capsys) -> None:
        from rich.console import Console

        from lilbee.cli.helpers import add_paths

        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        con = Console()
        with (
            mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP),
            mock.patch("lilbee.cli.helpers.copy_paths", return_value=[src]),
        ):
            add_paths([src], con, chat_mode=True)
            captured = capsys.readouterr()
            assert "Copied 1 path(s)" in captured.out


class TestChat:
    def test_chat_non_tty_exits_with_error(self) -> None:
        """CliRunner is non-TTY, so chat should exit with error."""
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 1
        assert "terminal" in result.output.lower()


class TestApplyOverrides:
    def test_data_dir_override(self, tmp_path):
        from lilbee.cli import apply_overrides

        apply_overrides(data_dir=tmp_path)
        assert tmp_path / "documents" == cfg.documents_dir

    def test_model_override(self):
        from lilbee.cli import apply_overrides

        apply_overrides(model="phi3")
        assert cfg.chat_model == "phi3:latest"

    def test_none_values_are_noop(self):
        from lilbee.cli import apply_overrides

        original_model = cfg.chat_model
        apply_overrides(data_dir=None, model=None)
        assert original_model == cfg.chat_model

    def test_use_global_resets_to_platform_default(self):
        from lilbee.cli import apply_overrides
        from lilbee.platform import default_data_dir

        apply_overrides(use_global=True)
        expected = default_data_dir()
        assert cfg.data_root == expected
        assert cfg.documents_dir == expected / "documents"
        assert cfg.data_dir == expected / "data"
        assert cfg.lancedb_dir == expected / "data" / "lancedb"

    def test_use_global_with_data_dir_raises(self, tmp_path):
        import typer

        from lilbee.cli import apply_overrides

        with pytest.raises(typer.BadParameter, match="Cannot use --global with --data-dir"):
            apply_overrides(data_dir=tmp_path, use_global=True)

    def test_lilbee_data_env_overrides_local_root(self, tmp_path, monkeypatch):
        """LILBEE_DATA env var takes precedence over .lilbee/ walk-up."""
        from lilbee.cli import apply_overrides

        env_dir = tmp_path / "env-data"
        env_dir.mkdir()
        monkeypatch.setenv("LILBEE_DATA", str(env_dir))
        apply_overrides()
        assert cfg.data_root == env_dir
        assert cfg.documents_dir == env_dir / "documents"

    def test_lilbee_data_env_ignored_when_data_dir_passed(self, tmp_path, monkeypatch):
        """Explicit --data-dir takes precedence over LILBEE_DATA."""
        from lilbee.cli import apply_overrides

        env_dir = tmp_path / "env-data"
        env_dir.mkdir()
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        monkeypatch.setenv("LILBEE_DATA", str(env_dir))
        apply_overrides(data_dir=explicit_dir)
        assert cfg.data_root == explicit_dir

    def test_lilbee_data_env_ignored_when_global(self, monkeypatch):
        """--global takes precedence over LILBEE_DATA."""
        from lilbee.cli import apply_overrides
        from lilbee.platform import default_data_dir

        monkeypatch.setenv("LILBEE_DATA", "/tmp/should-be-ignored")
        apply_overrides(use_global=True)
        assert cfg.data_root == default_data_dir()

    def test_generation_option_overrides(self):
        from lilbee.cli import apply_overrides

        apply_overrides(
            temperature=0.3,
            top_p=0.95,
            top_k_sampling=40,
            repeat_penalty=1.1,
            num_ctx=4096,
            seed=42,
        )
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.95
        assert cfg.top_k_sampling == 40
        assert cfg.repeat_penalty == 1.1
        assert cfg.num_ctx == 4096
        assert cfg.seed == 42

    def test_generation_option_none_is_noop(self):
        from lilbee.cli import apply_overrides

        cfg.temperature = 0.7
        apply_overrides(temperature=None)
        assert cfg.temperature == 0.7
        cfg.temperature = None


class TestGlobalFlag:
    """Tests for the --global / -g CLI flag."""

    def test_global_flag_on_status(self):
        from lilbee.platform import default_data_dir

        result = runner.invoke(app, ["--json", "status", "--global"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        expected = str(default_data_dir() / "documents")
        assert data["config"]["documents_dir"] == expected

    def test_global_short_flag_on_status(self):
        from lilbee.platform import default_data_dir

        result = runner.invoke(app, ["--json", "status", "-g"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        expected = str(default_data_dir() / "documents")
        assert data["config"]["documents_dir"] == expected

    def test_global_with_data_dir_errors(self, tmp_path):
        result = runner.invoke(app, ["status", "--global", "--data-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_help_shows_global_flag(self):
        result = runner.invoke(app, ["status", "--help"])
        # Rich wraps "--global" with ANSI codes in CI, so match without the dashes
        assert "global" in result.output


class TestMainModule:
    def test_python_m_lilbee_runs(self):
        """Ensure `python -m lilbee` invokes the CLI app."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "lilbee" in result.output


# ---------------------------------------------------------------------------
# Slash-command tests
# ---------------------------------------------------------------------------


class TestDefaultRequiresTerminal:
    """Invoking `lilbee` with no subcommand requires a TTY."""

    def test_non_tty_exits_with_error(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 1

    def test_tty_launches_tui(self) -> None:
        """When stdin/stdout are TTYs, the default command launches TUI."""
        from lilbee.cli.tui import run_tui as original_run_tui

        called = []

        def fake_run_tui(**kwargs: object) -> None:
            called.append(kwargs)

        import lilbee.cli.tui

        lilbee.cli.tui.run_tui = fake_run_tui  # type: ignore[assignment]
        try:
            # The runner's isatty check happens on sys.stdin, so we need to
            # make isatty return True on whatever stream the code actually reads.
            import sys

            orig_sin = sys.stdin.isatty
            orig_sout = sys.stdout.isatty
            sys.stdin.isatty = lambda: True  # type: ignore[method-assign]
            sys.stdout.isatty = lambda: True  # type: ignore[method-assign]
            try:
                from lilbee.cli.app import _default

                ctx = mock.MagicMock()
                ctx.invoked_subcommand = None
                _default(
                    ctx,
                    data_dir=None,
                    model=None,
                    json_output=False,
                    use_global=False,
                    log_level=None,
                    show_version=False,
                )
            finally:
                sys.stdin.isatty = orig_sin  # type: ignore[method-assign]
                sys.stdout.isatty = orig_sout  # type: ignore[method-assign]
        finally:
            lilbee.cli.tui.run_tui = original_run_tui
        assert called == [{"auto_sync": True}]


class TestChatLaunchesTui:
    """The chat subcommand launches TUI when on a TTY."""

    def test_chat_tty_launches_tui(self) -> None:
        from lilbee.cli.tui import run_tui as original_run_tui

        called = []

        def fake_run_tui(**kwargs: object) -> None:
            called.append(kwargs)

        import lilbee.cli.tui

        lilbee.cli.tui.run_tui = fake_run_tui  # type: ignore[assignment]
        try:
            with mock.patch("sys.stdin") as mock_in, mock.patch("sys.stdout") as mock_out:
                mock_in.isatty.return_value = True
                mock_out.isatty.return_value = True
                from lilbee.cli.commands import chat

                # Call with minimal defaults
                chat(
                    data_dir=None,
                    model=None,
                    use_global=False,
                    temperature=None,
                    top_p=None,
                    top_k_sampling=None,
                    repeat_penalty=None,
                    num_ctx=None,
                    seed=None,
                )
        finally:
            lilbee.cli.tui.run_tui = original_run_tui
        assert called == [{"auto_sync": True}]


# ---------------------------------------------------------------------------
# Completer tests
# ---------------------------------------------------------------------------


class TestListInstalledModels:
    """Test list_installed_models helper."""

    def test_returns_model_names_with_tags(self, mock_svc):
        mock_svc.provider.list_models.return_value = ["llama3:latest"]
        assert list_installed_models() == ["llama3:latest"]

    def test_returns_empty_on_error(self, mock_svc):
        mock_svc.provider.list_models.side_effect = ConnectionError("not running")
        assert list_installed_models() == []

    def test_excludes_embedding_model(self, mock_svc):
        cfg.embedding_model = "nomic-embed-text"
        mock_svc.provider.list_models.return_value = ["llama3:latest", "nomic-embed-text:latest"]
        result = list_installed_models()
        assert result == ["llama3:latest"]
        assert "nomic-embed-text:latest" not in result

    def test_exclude_vision_filters_vision_catalog(self, mock_svc):
        mock_svc.provider.list_models.return_value = [
            "llama3:latest",
            "LightOnOCR-2",
        ]
        result = list_installed_models(exclude_vision=True)
        assert result == ["llama3:latest"]
        assert "LightOnOCR-2" not in result


def _search_chunk(**overrides: object) -> SearchChunk:
    defaults: dict[str, object] = {
        "source": "a.pdf",
        "content_type": "pdf",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": "hi",
        "chunk_index": 0,
        "vector": [0.1, 0.2],
    }
    return SearchChunk(**(defaults | overrides))


class TestCleanResult:
    def test_strips_vector(self):
        result = clean_result(_search_chunk(distance=0.5))
        assert "vector" not in result
        assert result["source"] == "a.pdf"

    def test_has_distance(self):
        result = clean_result(_search_chunk(distance=0.42))
        assert result["distance"] == 0.42

    def test_excludes_none_scores(self):
        result = clean_result(_search_chunk(distance=0.5))
        assert "relevance_score" not in result

    def test_has_relevance_score(self):
        result = clean_result(_search_chunk(relevance_score=0.85))
        assert result["relevance_score"] == 0.85
        assert "vector" not in result
        assert "distance" not in result


class TestJsonFlag:
    def test_json_no_subcommand_returns_error(self):
        result = runner.invoke(app, ["--json"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data
        assert "terminal" in data["error"].lower()

    def test_short_j_flag_works(self):
        result = runner.invoke(app, ["-j"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data


# ---------------------------------------------------------------------------
# Search command tests (Task 2)
# ---------------------------------------------------------------------------

_MOCK_SEARCH_RESULTS = [
    SearchChunk(
        source="manual.pdf",
        content_type="pdf",
        page_start=5,
        page_end=5,
        line_start=0,
        line_end=0,
        chunk="The engine oil capacity is 5 quarts.",
        chunk_index=0,
        distance=0.25,
        vector=[0.1] * 768,
    ),
]


class TestSearch:
    def test_search_json_with_results(self, mock_svc):
        mock_svc.searcher.search.return_value = _MOCK_SEARCH_RESULTS
        result = runner.invoke(app, ["--json", "search", "engine oil"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "search"
        assert data["query"] == "engine oil"
        assert len(data["results"]) == 1
        assert "vector" not in data["results"][0]
        assert "distance" in data["results"][0]

    def test_search_json_empty_results(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        result = runner.invoke(app, ["--json", "search", "nothing"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["results"] == []

    def test_search_human_output(self, mock_svc):
        mock_svc.searcher.search.return_value = _MOCK_SEARCH_RESULTS
        result = runner.invoke(app, ["search", "engine oil"])
        assert result.exit_code == 0
        assert "manual.pdf" in result.output

    def test_search_human_truncates_long_chunks(self, mock_svc):
        mock_svc.searcher.search.return_value = [
            _MOCK_SEARCH_RESULTS[0].model_copy(update={"chunk": "x" * 100})
        ]
        result = runner.invoke(app, ["search", "test"])
        assert result.exit_code == 0
        assert result.output.count("x") < 100

    def test_search_human_no_results(self, mock_svc):
        mock_svc.searcher.search.return_value = []
        result = runner.invoke(app, ["search", "nothing"])
        assert result.exit_code == 0
        assert "No results found" in result.output

    def test_search_human_hybrid_shows_score(self, mock_svc):
        mock_svc.searcher.search.return_value = [
            _MOCK_SEARCH_RESULTS[0].model_copy(update={"relevance_score": 0.85, "distance": None})
        ]
        result = runner.invoke(app, ["search", "engine oil"])
        assert result.exit_code == 0
        assert "Score" in result.output
        assert "0.85" in result.output

    def test_search_json_hybrid_has_relevance_score(self, mock_svc):
        mock_svc.searcher.search.return_value = [
            _MOCK_SEARCH_RESULTS[0].model_copy(update={"relevance_score": 0.85, "distance": None})
        ]
        result = runner.invoke(app, ["--json", "search", "engine oil"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert "relevance_score" in data["results"][0]
        assert "distance" not in data["results"][0]


# ---------------------------------------------------------------------------
# JSON status tests (Task 3)
# ---------------------------------------------------------------------------


class TestVersionFlag:
    """Test --version / -V CLI flag."""

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "lilbee" in result.output
        assert get_version() in result.output

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert get_version() in result.output


class TestRemove:
    """Test remove command."""

    def test_remove_existing_source(self, isolated_env, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=["test.pdf"], not_found=[]
        )
        result = runner.invoke(app, ["remove", "test.pdf"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "test.pdf" in result.output

    def test_remove_nonexistent_source(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[], not_found=["nope.pdf"]
        )
        result = runner.invoke(app, ["remove", "nope.pdf"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_remove_multiple_sources(self, isolated_env, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=["a.pdf", "b.pdf"], not_found=[]
        )
        result = runner.invoke(app, ["remove", "a.pdf", "b.pdf"])
        assert result.exit_code == 0
        assert "a.pdf" in result.output
        assert "b.pdf" in result.output

    def test_remove_mixed_existing_and_not(self, isolated_env, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=["a.pdf"], not_found=["nope.pdf"]
        )
        result = runner.invoke(app, ["remove", "a.pdf", "nope.pdf"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "Not found" in result.output

    def test_remove_with_delete_flag(self, isolated_env, mock_svc):
        from lilbee.store import RemoveResult

        doc = cfg.documents_dir / "test.txt"
        doc.write_text("content")
        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=["test.txt"], not_found=[]
        )
        mock_svc.store.remove_documents.side_effect = lambda names, **kw: (
            doc.unlink() or RemoveResult(removed=["test.txt"], not_found=[])
            if kw.get("delete_files")
            else RemoveResult(removed=["test.txt"], not_found=[])
        )
        result = runner.invoke(app, ["remove", "--delete", "test.txt"])
        assert result.exit_code == 0
        assert not doc.exists()

    def test_remove_json(self, isolated_env, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=["test.pdf"], not_found=[]
        )
        result = runner.invoke(app, ["--json", "remove", "test.pdf"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "remove"
        assert "test.pdf" in data["removed"]

    def test_remove_json_not_found(self, mock_svc):
        from lilbee.store import RemoveResult

        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[], not_found=["nope.pdf"]
        )
        result = runner.invoke(app, ["--json", "remove", "nope.pdf"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["removed"] == []
        assert "nope.pdf" in data["not_found"]

    def test_remove_delete_path_traversal_skips(self, isolated_env, mock_svc):
        """Path traversal in name with --delete is caught and skipped."""
        from lilbee.store import RemoveResult

        traversal_name = "../../etc/passwd"
        mock_svc.store.remove_documents.return_value = RemoveResult(
            removed=[traversal_name], not_found=[]
        )
        result = runner.invoke(app, ["remove", "--delete", traversal_name])
        assert result.exit_code == 0


class TestChunks:
    """Test chunks command."""

    def test_chunks_nonexistent_source(self, mock_svc):
        mock_svc.store.get_sources.return_value = []
        result = runner.invoke(app, ["chunks", "nope.pdf"])
        assert result.exit_code == 1
        assert "Source not found" in result.output

    def test_chunks_nonexistent_json(self, mock_svc):
        mock_svc.store.get_sources.return_value = []
        result = runner.invoke(app, ["--json", "chunks", "nope.pdf"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data

    def test_chunks_with_source(self, isolated_env, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {
                "filename": "test.txt",
                "file_hash": "abc123",
                "chunk_count": 2,
                "ingested_at": "2026-01-01T00:00:00",
            },
        ]
        mock_svc.store.get_chunks_by_source.return_value = [
            SearchChunk(
                source="test.txt",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="First chunk content",
                chunk_index=0,
                vector=[0.1] * 768,
            ),
            SearchChunk(
                source="test.txt",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="Second chunk content",
                chunk_index=1,
                vector=[0.2] * 768,
            ),
        ]
        result = runner.invoke(app, ["chunks", "test.txt"])
        assert result.exit_code == 0
        assert "2 chunks" in result.output
        assert "First chunk" in result.output

    def test_chunks_truncates_long_chunk(self, isolated_env, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {
                "filename": "long.txt",
                "file_hash": "abc123",
                "chunk_count": 1,
                "ingested_at": "2026-01-01T00:00:00",
            },
        ]
        mock_svc.store.get_chunks_by_source.return_value = [
            SearchChunk(
                source="long.txt",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="x" * 200,
                chunk_index=0,
                vector=[0.1] * 768,
            ),
        ]
        result = runner.invoke(app, ["chunks", "long.txt"])
        assert result.exit_code == 0
        assert "..." in result.output

    def test_chunks_json(self, isolated_env, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {
                "filename": "test.txt",
                "file_hash": "abc123",
                "chunk_count": 1,
                "ingested_at": "2026-01-01T00:00:00",
            },
        ]
        mock_svc.store.get_chunks_by_source.return_value = [
            SearchChunk(
                source="test.txt",
                content_type="text",
                page_start=0,
                page_end=0,
                line_start=0,
                line_end=0,
                chunk="Chunk content",
                chunk_index=0,
                vector=[0.1] * 768,
            ),
        ]
        result = runner.invoke(app, ["--json", "chunks", "test.txt"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "chunks"
        assert data["source"] == "test.txt"
        assert len(data["chunks"]) == 1
        assert "vector" not in data["chunks"][0]


class TestReset:
    """Test reset command."""

    def test_reset_deletes_everything(self, isolated_env):
        """With --yes, both dirs are cleared."""

        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.documents_dir / "doc.txt").write_text("content")
        (cfg.data_dir / "db_file").write_text("data")

        result = runner.invoke(app, ["reset", "--yes"])
        assert result.exit_code == 0
        assert "Reset complete" in result.output
        assert list(cfg.documents_dir.iterdir()) == []
        assert list(cfg.data_dir.iterdir()) == []

    def test_reset_without_yes_prompts(self, isolated_env):
        """Without --yes, prompts and aborts on 'n'."""

        (cfg.documents_dir / "doc.txt").write_text("content")

        result = runner.invoke(app, ["reset"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        # File should still exist
        assert (cfg.documents_dir / "doc.txt").exists()

    def test_reset_without_yes_confirms(self, isolated_env):
        """Without --yes, confirming with 'y' deletes everything."""

        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.documents_dir / "doc.txt").write_text("content")

        result = runner.invoke(app, ["reset"], input="y\n")
        assert result.exit_code == 0
        assert "Reset complete" in result.output
        assert list(cfg.documents_dir.iterdir()) == []

    def test_reset_json_output(self, isolated_env):
        """JSON mode returns structured output."""

        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        (cfg.documents_dir / "doc.txt").write_text("content")

        result = runner.invoke(app, ["--json", "reset", "--yes"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "reset"
        assert data["deleted_docs"] == 1

    def test_reset_json_without_yes_errors(self):
        """JSON mode without --yes returns error."""
        result = runner.invoke(app, ["--json", "reset"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data

    def test_reset_empty_dirs(self, isolated_env):
        """Reset on already-empty dirs doesn't crash."""
        result = runner.invoke(app, ["reset", "--yes"])
        assert result.exit_code == 0
        assert "Reset complete" in result.output
        assert "0 document(s)" in result.output

    def test_reset_with_subdirectories(self, isolated_env):
        """Reset removes subdirectories too."""

        sub = cfg.documents_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")

        result = runner.invoke(app, ["reset", "--yes"])
        assert result.exit_code == 0
        assert list(cfg.documents_dir.iterdir()) == []

    def test_reset_data_dir_with_subdirectories(self, isolated_env):
        """Reset removes subdirectories in data dir too."""

        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        sub = cfg.data_dir / "lancedb"
        sub.mkdir()
        (sub / "table.lance").write_text("lance data")

        result = runner.invoke(app, ["reset", "--yes"])
        assert result.exit_code == 0
        assert list(cfg.data_dir.iterdir()) == []


class TestInit:
    def test_init_creates_structure(self, tmp_path):
        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        root = tmp_path / ".lilbee"
        assert root.is_dir()
        assert (root / "documents").is_dir()
        assert (root / "data").is_dir()
        assert (root / ".gitignore").read_text() == "data/\n"
        assert "Initialized" in result.output

    def test_init_already_exists(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Already initialized" in result.output

    def test_init_json_created(self, tmp_path):
        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["--json", "init"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "init"
        assert data["created"] is True
        assert ".lilbee" in data["path"]

    def test_init_json_already_exists(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(app, ["--json", "init"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["created"] is False


class TestVersion:
    def test_version_human(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "lilbee" in result.output
        assert get_version() in result.output

    def test_version_json(self):
        result = runner.invoke(app, ["--json", "version"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "version"
        assert data["version"] == get_version()


class TestGetVersion:
    def test_returns_string(self):
        ver = get_version()
        assert isinstance(ver, str)
        assert len(ver) > 0


class TestStatusJson:
    def test_status_json_empty(self):
        result = runner.invoke(app, ["--json", "status"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "status"
        assert "config" in data
        assert data["sources"] == []
        assert data["total_chunks"] == 0

    def test_status_json_with_sources(self, isolated_env, mock_svc):
        mock_svc.store.get_sources.return_value = [
            {
                "filename": "test.pdf",
                "file_hash": "abc123hash",
                "chunk_count": 10,
                "ingested_at": "2026-01-01T00:00:00",
            }
        ]
        result = runner.invoke(app, ["--json", "status"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "test.pdf"
        assert data["total_chunks"] == 10
        assert "documents_dir" in data["config"]

    def test_status_json_includes_vision_model_when_set(self):
        cfg.vision_model = "test-vision:latest"
        result = runner.invoke(app, ["--json", "status"])
        data = json.loads(result.output.strip())
        assert data["config"]["vision_model"] == "test-vision:latest"

    def test_status_json_excludes_vision_model_when_empty(self):
        cfg.vision_model = ""
        result = runner.invoke(app, ["--json", "status"])
        data = json.loads(result.output.strip())
        assert "vision_model" not in data["config"]


# ---------------------------------------------------------------------------
# JSON sync/rebuild/add tests (Task 4)
# ---------------------------------------------------------------------------


class TestSyncJson:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_sync_json_empty(self, mock_sync):
        result = runner.invoke(app, ["--json", "sync"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "sync"
        assert data["added"] == []
        assert data["unchanged"] == 0

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["new.txt"], removed=["old.txt"], unchanged=2),
    )
    def test_sync_json_with_changes(self, mock_sync):
        result = runner.invoke(app, ["--json", "sync"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["added"] == ["new.txt"]
        assert data["removed"] == ["old.txt"]
        assert data["unchanged"] == 2


class TestRebuildJson:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_rebuild_json(self, mock_sync, isolated_env):
        result = runner.invoke(app, ["--json", "rebuild"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "rebuild"
        assert "ingested" in data


class TestAddJson:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_add_json(self, mock_sync, isolated_env, tmp_path):
        src = tmp_path / "source" / "manual.txt"
        src.parent.mkdir()
        src.write_text("Engine oil capacity is 5 quarts.")
        result = runner.invoke(app, ["--json", "add", str(src)])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "add"
        assert "manual.txt" in data["copied"]
        assert "sync" in data


# ---------------------------------------------------------------------------
# JSON ask tests (Task 5)
# ---------------------------------------------------------------------------


class TestAskJson:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_json(self, mock_sync, mock_svc):
        from lilbee.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(
            answer="5 quarts",
            sources=[
                SearchChunk(
                    source="manual.pdf",
                    content_type="pdf",
                    page_start=1,
                    page_end=1,
                    line_start=0,
                    line_end=0,
                    chunk="oil",
                    chunk_index=0,
                    distance=0.3,
                    vector=[0.1],
                )
            ],
        )
        result = runner.invoke(app, ["--json", "ask", "oil capacity?"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "ask"
        assert data["question"] == "oil capacity?"
        assert data["answer"] == "5 quarts"
        assert len(data["sources"]) == 1
        assert "vector" not in data["sources"][0]
        assert "distance" in data["sources"][0]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_json_no_results(self, mock_sync, mock_svc):
        from lilbee.query import AskResult

        mock_svc.searcher.ask_raw.return_value = AskResult(
            answer="No relevant documents found.", sources=[]
        )
        result = runner.invoke(app, ["--json", "ask", "anything"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["sources"] == []
        assert "No relevant" in data["answer"]


class TestAskModelNotFound:
    """CLI should show a friendly error when the model doesn't exist."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_model_not_found_human(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_stream.side_effect = RuntimeError("Model 'bad' not found")
        result = runner.invoke(app, ["ask", "hello"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_model_not_found_json(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_raw.side_effect = RuntimeError("Model 'bad' not found")
        result = runner.invoke(app, ["--json", "ask", "hello"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "not found" in data["error"]


class TestBackendUnavailable:
    """CLI commands should show friendly errors when the backend is unreachable."""

    _ERR = RuntimeError("Connection refused")

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_sync_backend_unavailable(self, mock_sync):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_sync_backend_unavailable_json(self, mock_sync):
        result = runner.invoke(app, ["--json", "sync"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "Connection refused" in data["error"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_rebuild_backend_unavailable(self, mock_sync):
        result = runner.invoke(app, ["rebuild"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_rebuild_backend_unavailable_json(self, mock_sync):
        result = runner.invoke(app, ["--json", "rebuild"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "Connection refused" in data["error"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_add_backend_unavailable(self, mock_sync, isolated_env, tmp_path):
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        result = runner.invoke(app, ["add", str(src)])
        assert result.exit_code == 1
        assert "Connection refused" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_add_backend_unavailable_json(self, mock_sync, isolated_env, tmp_path):
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        result = runner.invoke(app, ["--json", "add", str(src)])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "Connection refused" in data["error"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_auto_sync_backend_unavailable(self, mock_sync):
        result = runner.invoke(app, ["ask", "hello"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output


class TestEnsureChatModelWiring:
    """Verify that ask and chat call ensure_chat_model before running."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_calls_ensure_chat_model(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_stream.return_value = _mock_stream("answer")
        with mock.patch("lilbee.models.ensure_chat_model") as mock_ensure:
            runner.invoke(app, ["ask", "test"])
            mock_ensure.assert_called_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_calls_validate_model(self, mock_sync, mock_svc):
        mock_svc.searcher.ask_stream.return_value = _mock_stream("answer")
        runner.invoke(app, ["ask", "test"])
        mock_svc.embedder.validate_model.assert_called_once()


# ---------------------------------------------------------------------------
# Vision model setup tests
# ---------------------------------------------------------------------------


class TestEnsureVisionModel:
    """Tests for _ensure_vision_model and helpers."""

    def test_returns_early_if_vision_configured(self) -> None:
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = "llava:7b"
        with mock.patch("lilbee.cli.commands._validate_configured_vision") as mock_val:
            _ensure_vision_model()
            mock_val.assert_called_once()

    def test_restores_saved_model_from_settings(self) -> None:
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = ""
        with (
            mock.patch("lilbee.cli.commands.settings.get", return_value="saved-model"),
            mock.patch("lilbee.cli.commands._validate_configured_vision") as mock_val,
        ):
            _ensure_vision_model()
            assert cfg.vision_model == "saved-model"
            mock_val.assert_called_once()

    def test_backend_unreachable_disables_vision(self) -> None:
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = ""
        with (
            mock.patch("lilbee.cli.commands.settings.get", return_value=""),
            mock.patch("lilbee.models.list_installed_models", side_effect=RuntimeError("fail")),
        ):
            _ensure_vision_model()
            assert cfg.vision_model == ""

    def test_tty_calls_interactive_picker(self) -> None:
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = ""
        with (
            mock.patch("lilbee.cli.commands.settings.get", return_value=""),
            mock.patch("lilbee.models.list_installed_models", return_value=["m1"]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch("lilbee.cli.commands._pick_vision_interactive") as mock_pick,
        ):
            mock_stdin.isatty.return_value = True
            _ensure_vision_model()
            mock_pick.assert_called_once()

    def test_non_tty_calls_auto_picker(self) -> None:
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = ""
        with (
            mock.patch("lilbee.cli.commands.settings.get", return_value=""),
            mock.patch("lilbee.models.list_installed_models", return_value=["m1"]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch("lilbee.cli.commands._pick_vision_auto") as mock_pick,
        ):
            mock_stdin.isatty.return_value = False
            _ensure_vision_model()
            mock_pick.assert_called_once()


class TestValidateConfiguredVision:
    def test_already_installed_noop(self) -> None:
        from lilbee.cli.commands import _validate_configured_vision

        cfg.vision_model = "llava:7b"
        with mock.patch("lilbee.models.list_installed_models", return_value=["llava:7b"]):
            _validate_configured_vision()
            assert cfg.vision_model == "llava:7b"

    def test_not_installed_pulls(self) -> None:
        from lilbee.cli.commands import _validate_configured_vision

        cfg.vision_model = "llava:7b"
        with (
            mock.patch("lilbee.models.list_installed_models", return_value=[]),
            mock.patch("lilbee.cli.commands._try_pull", return_value=True) as mock_pull,
        ):
            _validate_configured_vision()
            mock_pull.assert_called_once_with("llava:7b")

    def test_pull_fails_clears_vision(self) -> None:
        from lilbee.cli.commands import _validate_configured_vision

        cfg.vision_model = "llava:7b"
        with (
            mock.patch("lilbee.models.list_installed_models", return_value=[]),
            mock.patch("lilbee.cli.commands._try_pull", return_value=False),
        ):
            _validate_configured_vision()
            assert cfg.vision_model == ""

    def test_backend_unreachable_keeps_config(self) -> None:
        from lilbee.cli.commands import _validate_configured_vision

        cfg.vision_model = "llava:7b"
        with mock.patch("lilbee.models.list_installed_models", side_effect=RuntimeError):
            _validate_configured_vision()
            assert cfg.vision_model == "llava:7b"


class TestPickVisionInteractive:
    @pytest.fixture(autouse=True)
    def _patch_vision_deps(self):
        from lilbee.models import VISION_CATALOG

        with (
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("lilbee.models.display_vision_picker", return_value=VISION_CATALOG[0]),
        ):
            yield

    def test_default_choice(self) -> None:
        from lilbee.cli.commands import _pick_vision_interactive

        with (
            mock.patch("builtins.input", return_value=""),
            mock.patch("lilbee.cli.commands._pull_and_save_vision") as mock_save,
        ):
            _pick_vision_interactive(set())
            mock_save.assert_called_once()

    def test_eof_cancels(self) -> None:
        from lilbee.cli.commands import _pick_vision_interactive

        with mock.patch("builtins.input", side_effect=EOFError):
            _pick_vision_interactive(set())

    def test_invalid_input(self) -> None:
        from lilbee.cli.commands import _pick_vision_interactive

        with mock.patch("builtins.input", return_value="abc"):
            _pick_vision_interactive(set())

    def test_out_of_range(self) -> None:
        from lilbee.cli.commands import _pick_vision_interactive

        with mock.patch("builtins.input", return_value="999"):
            _pick_vision_interactive(set())

    def test_valid_numeric_choice(self) -> None:
        from lilbee.cli.commands import _pick_vision_interactive

        with (
            mock.patch("builtins.input", return_value="1"),
            mock.patch("lilbee.cli.commands._pull_and_save_vision") as mock_save,
        ):
            _pick_vision_interactive(set())
            mock_save.assert_called_once()


class TestPickVisionAuto:
    def test_auto_selects_and_pulls(self) -> None:
        from lilbee.cli.commands import _pick_vision_auto

        with (
            mock.patch("lilbee.models.pick_default_vision_model") as mock_pick,
            mock.patch("lilbee.cli.commands._pull_and_save_vision") as mock_save,
        ):
            mock_pick.return_value = mock.MagicMock(name="llava:7b")
            _pick_vision_auto(set())
            mock_save.assert_called_once()


class TestTryPull:
    def test_success(self) -> None:
        from lilbee.cli.commands import _try_pull

        with mock.patch("lilbee.models.pull_with_progress"):
            assert _try_pull("model") is True

    def test_failure(self) -> None:
        from lilbee.cli.commands import _try_pull

        with mock.patch("lilbee.models.pull_with_progress", side_effect=RuntimeError("fail")):
            assert _try_pull("model") is False


class TestPullAndSaveVision:
    def test_already_installed(self) -> None:
        from lilbee.cli.commands import _pull_and_save_vision

        cfg.vision_model = ""
        _pull_and_save_vision("llava:7b", {"llava:7b"})
        assert cfg.vision_model == "llava:7b"

    def test_pull_needed_succeeds(self) -> None:
        from lilbee.cli.commands import _pull_and_save_vision

        cfg.vision_model = ""
        with mock.patch("lilbee.cli.commands._try_pull", return_value=True):
            _pull_and_save_vision("llava:7b", set())
        assert cfg.vision_model == "llava:7b"

    def test_pull_fails(self) -> None:
        from lilbee.cli.commands import _pull_and_save_vision

        cfg.vision_model = ""
        with mock.patch("lilbee.cli.commands._try_pull", return_value=False):
            _pull_and_save_vision("llava:7b", set())
        assert cfg.vision_model == ""


# ---------------------------------------------------------------------------
# --vision flag tests
# ---------------------------------------------------------------------------


class TestVisionTimeout:
    """Tests for --vision-timeout flag on sync, add, rebuild."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_vision_timeout_on_sync(self, mock_sync):
        """--vision-timeout sets cfg.vision_timeout for sync."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["sync", "--vision", "--vision-timeout=60"])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 60.0

    def test_vision_timeout_on_add(self, isolated_env, tmp_path, mock_svc):
        """--vision-timeout sets cfg.vision_timeout for add."""
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["add", "--vision", "--vision-timeout=90", str(src)])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 90.0

    def test_vision_timeout_on_rebuild(self, mock_svc):
        """--vision-timeout sets cfg.vision_timeout for rebuild."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["rebuild", "--vision", "--vision-timeout=120"])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 120.0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_no_vision_timeout_leaves_default(self, mock_sync):
        """Without --vision-timeout, cfg.vision_timeout stays at default."""
        cfg.vision_timeout = 120.0
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["sync", "--vision"])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 120.0


class TestLogLevel:
    """Tests for --log-level flag and LILBEE_LOG_LEVEL configuration."""

    def test_log_level_flag_debug(self):
        """--log-level=DEBUG sets root logger to DEBUG."""
        result = runner.invoke(app, ["--log-level=DEBUG", "status"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.DEBUG

    def test_log_level_flag_info(self):
        """--log-level=INFO sets root logger to INFO."""
        result = runner.invoke(app, ["--log-level=INFO", "status"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.INFO

    def test_log_level_env_var(self, monkeypatch):
        """LILBEE_LOG_LEVEL env var controls log level."""
        monkeypatch.setenv("LILBEE_LOG_LEVEL", "INFO")
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.INFO

    def test_flag_overrides_env_var(self, monkeypatch):
        """--log-level overrides LILBEE_LOG_LEVEL."""
        monkeypatch.setenv("LILBEE_LOG_LEVEL", "WARNING")
        result = runner.invoke(app, ["--log-level=DEBUG", "status"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_log_level_defaults_to_warning(self, monkeypatch):
        """Invalid LILBEE_LOG_LEVEL falls back to WARNING."""
        monkeypatch.setenv("LILBEE_LOG_LEVEL", "BOGUS")
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert logging.getLogger().level == logging.WARNING


class TestIngestShutdownError:
    def test_process_one_converts_shutdown_error(self):
        """RuntimeError from executor shutdown is converted to CancelledError."""
        import asyncio

        from lilbee.ingest import ingest_batch

        shutdown_err = RuntimeError("cannot schedule new futures after shutdown")

        async def _run():
            added = ["test.txt"]
            updated: list[str] = []
            failed: list[str] = []
            with (
                mock.patch("lilbee.ingest._ingest_file", side_effect=shutdown_err),
                pytest.raises(asyncio.CancelledError),
            ):
                await ingest_batch(
                    [("test.txt", __import__("pathlib").Path("test.txt"), "text", "abc123", False)],
                    added,
                    updated,
                    failed,
                    quiet=True,
                )

        asyncio.run(_run())


class TestAddWithUrls:
    """Tests for URL crawling through the add CLI command."""

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.cli.commands._crawl_urls_blocking", return_value=[])
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_add_url_triggers_crawl(self, mock_sync, mock_crawl, mock_avail):
        """Adding a URL calls the crawler instead of copying files."""
        result = runner.invoke(app, ["add", "https://example.com"])
        assert result.exit_code == 0
        mock_crawl.assert_called_once()
        args = mock_crawl.call_args
        assert args[0][0] == ["https://example.com"]

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.cli.commands._crawl_urls_blocking", return_value=[])
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_add_url_with_crawl_flag(self, mock_sync, mock_crawl, mock_avail):
        """--crawl flag is passed through to the crawler."""
        result = runner.invoke(app, ["add", "--crawl", "https://example.com"])
        assert result.exit_code == 0
        assert mock_crawl.call_args[1]["crawl"] is True

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.cli.commands._crawl_urls_blocking", return_value=[])
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_add_url_with_depth(self, mock_sync, mock_crawl, mock_avail):
        """--depth is passed through to the crawler."""
        result = runner.invoke(app, ["add", "--crawl", "--depth", "3", "https://example.com"])
        assert result.exit_code == 0
        assert mock_crawl.call_args[1]["depth"] == 3

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.cli.commands._crawl_urls_blocking", return_value=[])
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_add_url_with_max_pages(self, mock_sync, mock_crawl, mock_avail):
        """--max-pages is passed through to the crawler."""
        result = runner.invoke(app, ["add", "--crawl", "--max-pages", "10", "https://example.com"])
        assert result.exit_code == 0
        assert mock_crawl.call_args[1]["max_pages"] == 10

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.cli.commands._crawl_urls_blocking")
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_add_url_json_mode(self, mock_sync, mock_crawl, mock_avail, isolated_env):
        """URL add in JSON mode returns structured output."""
        from pathlib import Path

        mock_crawl.return_value = [Path("/tmp/a.md")]
        result = runner.invoke(app, ["--json", "add", "https://example.com"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "add"
        assert data["crawled"] == 1

    @mock.patch("lilbee.crawler.crawler_available", return_value=True)
    @mock.patch("lilbee.cli.commands._crawl_urls_blocking", return_value=[])
    def test_add_mixed_urls_and_files(
        self, mock_crawl, mock_avail, isolated_env, tmp_path, mock_svc
    ):
        """Mixing URLs and file paths in one add command."""
        src = tmp_path / "source" / "doc.txt"
        src.parent.mkdir()
        src.write_text("file content")
        result = runner.invoke(app, ["add", str(src), "https://example.com"])
        assert result.exit_code == 0
        mock_crawl.assert_called_once()

    def test_add_url_without_crawler_installed(self):
        """Adding a URL when crawl4ai is not installed shows install message."""
        with mock.patch("lilbee.crawler.crawler_available", return_value=False):
            result = runner.invoke(app, ["add", "https://example.com"])
            assert result.exit_code == 1
            assert "pip install" in result.output.lower()

    def test_add_nonexistent_path_fails(self):
        """Adding a nonexistent file path fails with error."""
        result = runner.invoke(app, ["add", "/tmp/nonexistent_crawl_test_xyz.txt"])
        assert result.exit_code != 0

    def test_add_nonexistent_path_json_fails(self):
        """Adding a nonexistent file path in JSON mode returns error."""
        result = runner.invoke(app, ["--json", "add", "/tmp/nonexistent_crawl_test_xyz.txt"])
        assert result.exit_code != 0


class TestIsUrl:
    def test_http(self):
        from lilbee.crawler import is_url

        assert is_url("http://example.com")

    def test_https(self):
        from lilbee.crawler import is_url

        assert is_url("https://example.com")

    def test_not_url(self):
        from lilbee.crawler import is_url

        assert not is_url("/tmp/file.txt")

    def test_ftp_not_url(self):
        from lilbee.crawler import is_url

        assert not is_url("ftp://example.com")


class TestPartitionInputs:
    def test_separates_urls_and_paths(self):
        from lilbee.cli.commands import _partition_inputs

        paths, urls = _partition_inputs(["/tmp/a.txt", "https://example.com", "/tmp/b.txt"])
        assert len(paths) == 2
        assert urls == ["https://example.com"]

    def test_all_urls(self):
        from lilbee.cli.commands import _partition_inputs

        paths, urls = _partition_inputs(["https://a.com", "http://b.com"])
        assert len(paths) == 0
        assert len(urls) == 2

    def test_all_paths(self):
        from lilbee.cli.commands import _partition_inputs

        paths, urls = _partition_inputs(["/a.txt", "/b.txt"])
        assert len(paths) == 2
        assert len(urls) == 0


class TestCrawlUrlsBlocking:
    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    def test_single_url(self, mock_crawl, isolated_env):
        from pathlib import Path

        from lilbee.cli.commands import _crawl_urls_blocking

        async def _fake_crawl(url, **kwargs):
            # Call the on_progress callback to cover the closure body
            from lilbee.progress import CrawlPageEvent, EventType

            cb = kwargs.get("on_progress")
            if cb:
                cb(EventType.CRAWL_PAGE, CrawlPageEvent(current=1, total=1, url=url))
            return [Path("/tmp/page.md")]

        mock_crawl.side_effect = _fake_crawl
        result = _crawl_urls_blocking(
            ["https://example.com"], crawl=False, depth=None, max_pages=None
        )
        assert len(result) == 1
        mock_crawl.assert_called_once()

    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    def test_with_crawl_flag(self, mock_crawl, isolated_env):
        from lilbee.cli.commands import _crawl_urls_blocking

        mock_crawl.return_value = []
        _crawl_urls_blocking(["https://example.com"], crawl=True, depth=None, max_pages=None)
        call_kwargs = mock_crawl.call_args[1]
        assert call_kwargs["depth"] == cfg.crawl_max_depth

    @mock.patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock)
    def test_with_explicit_depth(self, mock_crawl, isolated_env):
        from lilbee.cli.commands import _crawl_urls_blocking

        mock_crawl.return_value = []
        _crawl_urls_blocking(["https://example.com"], crawl=True, depth=5, max_pages=20)
        call_kwargs = mock_crawl.call_args[1]
        assert call_kwargs["depth"] == 5
        assert call_kwargs["max_pages"] == 20


class TestTopicsCommand:
    def test_not_installed_shows_error(self):
        with mock.patch("lilbee.concepts.concepts_available", return_value=False):
            result = runner.invoke(app, ["topics"])
            assert result.exit_code == 1
            assert "pip install" in result.output.lower()

    def test_not_installed_json_mode(self):
        with mock.patch("lilbee.concepts.concepts_available", return_value=False):
            result = runner.invoke(app, ["--json", "topics"])
            assert result.exit_code == 1
            output = json.loads(result.output)
            assert "error" in output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_disabled_shows_error(self, _mock_avail):
        cfg.concept_graph = False
        result = runner.invoke(app, ["topics"])
        assert result.exit_code == 1
        assert "disabled" in result.output.lower()

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_disabled_json_mode(self, _mock_avail):
        cfg.concept_graph = False
        result = runner.invoke(app, ["--json", "topics"])
        assert result.exit_code == 1
        output = json.loads(result.output)
        assert "error" in output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_overview_shows_communities(self, _mock_avail, mock_svc):
        from lilbee.concepts import Community

        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.top_communities.return_value = [
            Community(cluster_id=0, size=3, concepts=["python", "django", "flask"]),
        ]
        result = runner.invoke(app, ["topics"])
        assert result.exit_code == 0
        assert "python" in result.output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_overview_json_mode(self, _mock_avail, mock_svc):
        from lilbee.concepts import Community

        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.top_communities.return_value = [
            Community(cluster_id=0, size=2, concepts=["ml", "ai"]),
        ]
        result = runner.invoke(app, ["--json", "topics"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["command"] == "topics"
        assert len(output["communities"]) == 1

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_query_shows_related_concepts(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = ["python"]
        mock_svc.concepts.expand_query.return_value = ["django", "flask"]
        result = runner.invoke(app, ["topics", "python"])
        assert result.exit_code == 0
        assert "django" in result.output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_query_json_mode(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = ["python"]
        mock_svc.concepts.expand_query.return_value = ["django"]
        result = runner.invoke(app, ["--json", "topics", "python"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "python" in output["concepts"]
        assert "django" in output["concepts"]

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_no_communities(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.top_communities.return_value = []
        result = runner.invoke(app, ["topics"])
        assert result.exit_code == 0
        assert "No concept communities" in result.output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_query_no_concepts(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.extract_concepts.return_value = []
        mock_svc.concepts.expand_query.return_value = []
        result = runner.invoke(app, ["topics", "???"])
        assert result.exit_code == 0
        assert "No concepts found" in result.output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_graph_none_shows_error(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = False
        result = runner.invoke(app, ["topics"])
        assert result.exit_code == 1
        assert "not available" in result.output.lower()

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_graph_none_json_mode(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = False
        result = runner.invoke(app, ["--json", "topics"])
        assert result.exit_code == 1
        output = json.loads(result.output)
        assert "error" in output

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_top_k_option(self, _mock_avail, mock_svc):
        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        mock_svc.concepts.top_communities.return_value = []
        runner.invoke(app, ["topics", "--top-k", "5"])
        mock_svc.concepts.top_communities.assert_called_once_with(k=5)

    @mock.patch("lilbee.concepts.concepts_available", return_value=True)
    def test_large_community_shows_more_count(self, _mock_avail, mock_svc):
        from lilbee.concepts import Community

        cfg.concept_graph = True
        mock_svc.concepts.get_graph.return_value = True
        many_concepts = [f"concept_{i}" for i in range(8)]
        mock_svc.concepts.top_communities.return_value = [
            Community(cluster_id=0, size=8, concepts=many_concepts),
        ]
        result = runner.invoke(app, ["topics"])
        assert result.exit_code == 0
        assert "concept_0" in result.output
        assert "more)" in result.output


class TestWikiLint:
    def test_lint_all_no_issues(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["wiki", "lint"])
        assert result.exit_code == 0
        assert "No issues found" in result.output

    def test_lint_all_with_issues(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        wiki_dir = isolated_env / "wiki" / "summaries"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "doc.md").write_text("Unmarked claim.\n")
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["wiki", "lint"])
        assert result.exit_code == 0
        assert "Unmarked" in result.output

    def test_lint_single_page(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        wiki_dir = isolated_env / "wiki" / "summaries"
        wiki_dir.mkdir(parents=True)
        (wiki_dir / "doc.md").write_text(
            "> Cited.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["wiki", "lint", "wiki/summaries/doc.md"])
        assert result.exit_code == 0

    def test_lint_json_output(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        cfg.json_mode = True
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["--json", "wiki", "lint"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["command"] == "wiki_lint"
        assert "total" in data


class TestWikiCitations:
    def test_citations_empty(self, mock_svc):
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["wiki", "citations", "wiki/summaries/doc.md"])
        assert result.exit_code == 0
        assert "No citations found" in result.output

    def test_citations_with_records(self, mock_svc):
        mock_svc.store.get_citations_for_wiki.return_value = [
            {
                "wiki_source": "wiki/summaries/doc.md",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": "doc.md",
                "source_hash": "abc",
                "page_start": 0,
                "page_end": 0,
                "line_start": 1,
                "line_end": 10,
                "excerpt": "Python supports typing.",
                "created_at": "2026-01-01",
            }
        ]
        result = runner.invoke(app, ["wiki", "citations", "wiki/summaries/doc.md"])
        assert result.exit_code == 0
        assert "src1" in result.output
        assert "doc.md" in result.output

    def test_citations_long_excerpt_truncated(self, mock_svc):
        long_excerpt = "A" * 80
        mock_svc.store.get_citations_for_wiki.return_value = [
            {
                "wiki_source": "wiki/summaries/doc.md",
                "wiki_chunk_index": 0,
                "citation_key": "src1",
                "claim_type": "fact",
                "source_filename": "doc.md",
                "source_hash": "abc",
                "page_start": 0,
                "page_end": 0,
                "line_start": 1,
                "line_end": 10,
                "excerpt": long_excerpt,
                "created_at": "2026-01-01",
            }
        ]
        result = runner.invoke(app, ["wiki", "citations", "wiki/summaries/doc.md"])
        assert result.exit_code == 0
        # Full 80-char excerpt should not appear — truncated by code or Rich
        assert long_excerpt not in result.output

    def test_citations_json_output(self, mock_svc):
        cfg.json_mode = True
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["--json", "wiki", "citations", "wiki/summaries/doc.md"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["command"] == "wiki_citations"
        assert data["total"] == 0


class TestWikiStatus:
    def test_status_no_wiki_dir(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        result = runner.invoke(app, ["wiki", "status"])
        assert result.exit_code == 0
        assert "does not exist" in result.output

    def test_status_with_pages(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        (isolated_env / "wiki" / "summaries").mkdir(parents=True)
        (isolated_env / "wiki" / "summaries" / "a.md").write_text("content")
        (isolated_env / "wiki" / "drafts").mkdir(parents=True)
        (isolated_env / "wiki" / "drafts" / "b.md").write_text("content")
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["wiki", "status"])
        assert result.exit_code == 0
        assert "1" in result.output  # summaries count

    def test_status_json_output(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        cfg.json_mode = True
        result = runner.invoke(app, ["--json", "wiki", "status"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "wiki_enabled" in data
        assert data["pages"] == 0

    def test_status_json_with_pages(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        cfg.json_mode = True
        (isolated_env / "wiki" / "summaries").mkdir(parents=True)
        (isolated_env / "wiki" / "summaries" / "a.md").write_text("content")
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["--json", "wiki", "status"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summaries"] == 1
        assert data["drafts"] == 0

    def test_status_all_clean(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        (isolated_env / "wiki" / "summaries").mkdir(parents=True)
        (isolated_env / "wiki" / "summaries" / "a.md").write_text(
            "> Cited.[^src1]\n\n"
            "---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            "[^src1]: doc.md, lines 1-5\n"
        )
        mock_svc.store.get_citations_for_wiki.return_value = []
        result = runner.invoke(app, ["wiki", "status"])
        assert result.exit_code == 0
        assert "all clean" in result.output

    def test_status_wiki_disabled(self, mock_svc, isolated_env):
        cfg.wiki = False
        cfg.wiki_dir = "wiki"
        result = runner.invoke(app, ["wiki", "status"])
        assert result.exit_code == 0


class TestWikiPrune:
    def test_prune_no_pages(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        with mock.patch("lilbee.wiki.prune.prune_wiki") as mock_prune:
            from lilbee.wiki.prune import PruneReport

            mock_prune.return_value = PruneReport()
            result = runner.invoke(app, ["wiki", "prune"])
        assert result.exit_code == 0
        assert "No pages pruned" in result.output

    def test_prune_json_output(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        cfg.json_mode = True
        with mock.patch("lilbee.wiki.prune.prune_wiki") as mock_prune:
            from lilbee.wiki.prune import PruneReport

            mock_prune.return_value = PruneReport()
            result = runner.invoke(app, ["--json", "wiki", "prune"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["command"] == "wiki_prune"
        assert data["archived"] == 0

    def test_prune_with_records(self, mock_svc, isolated_env):
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        with mock.patch("lilbee.wiki.prune.prune_wiki") as mock_prune:
            from lilbee.wiki.prune import PruneAction, PruneRecord, PruneReport

            report = PruneReport()
            report.records = [
                PruneRecord(
                    wiki_source="wiki/summaries/old.md",
                    action=PruneAction.ARCHIVED,
                    reason="all sources deleted",
                ),
            ]
            mock_prune.return_value = report
            result = runner.invoke(app, ["wiki", "prune"])
        assert result.exit_code == 0
        assert "old.md" in result.output


class TestCrawlProgressCallback:
    def test_crawl_page_event(self):
        """Crawl progress callback handles CrawlPageEvent."""
        from lilbee.progress import CrawlPageEvent, EventType

        event = CrawlPageEvent(url="https://example.com", current=3, total=10)
        # The callback in commands.py checks isinstance(data, CrawlPageEvent)
        assert event.current == 3
        assert event.total == 10

    def test_crawl_callback_wrong_type_raises(self):
        """Crawl progress callback raises TypeError for non-CrawlPageEvent."""
        from lilbee.progress import EventType, FileStartEvent

        # Simulate what _make_callback does
        from unittest.mock import MagicMock

        progress = MagicMock()

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_PAGE:
                if not isinstance(data, CrawlPageEvent):
                    raise TypeError(f"Expected CrawlPageEvent, got {type(data).__name__}")

        from lilbee.progress import CrawlPageEvent

        with pytest.raises(TypeError, match="Expected CrawlPageEvent"):
            on_progress(EventType.CRAWL_PAGE, FileStartEvent(file="x", total_files=1, current_file=1))


class TestLoginCommand:
    def test_login_already_logged_in_decline(self):
        """Login when already logged in and user declines."""
        with (
            mock.patch("huggingface_hub.login"),
            mock.patch("huggingface_hub.get_token", return_value="existing-token"),
            mock.patch("webbrowser.open"),
        ):
            result = runner.invoke(app, ["login"], input="n\n")
            assert result.exit_code == 0
            assert "Already logged in" in result.output

    def test_login_fresh(self):
        """Login with fresh token."""
        with (
            mock.patch("huggingface_hub.login") as mock_hf_login,
            mock.patch("huggingface_hub.get_token", return_value=None),
            mock.patch("webbrowser.open"),
        ):
            result = runner.invoke(app, ["login"], input="hf_test_token_123\n")
            assert result.exit_code == 0
            assert "Logged in" in result.output
            mock_hf_login.assert_called_once()

    def test_login_empty_token(self):
        """Login with empty token exits with error."""
        with (
            mock.patch("huggingface_hub.login"),
            mock.patch("huggingface_hub.get_token", return_value=None),
            mock.patch("webbrowser.open"),
        ):
            # typer.prompt with hide_input requires a non-empty value;
            # supply a whitespace-only token to trigger the "No token" error path
            result = runner.invoke(app, ["login"], input="   \n")
            assert result.exit_code == 1
            assert "No token" in result.output


class TestSyncProgressPrinter:
    def test_file_start_event(self):
        """_sync_progress_printer handles FILE_START event."""
        from lilbee.cli.sync import _sync_progress_printer
        from lilbee.progress import EventType, FileStartEvent

        con = MagicMock()
        cb = _sync_progress_printer(con)
        cb(EventType.FILE_START, FileStartEvent(file="doc.md", total_files=5, current_file=2))
        con.print.assert_called_once()
        assert "doc.md" in str(con.print.call_args)

    def test_done_event(self):
        """_sync_progress_printer handles DONE event with summary."""
        from lilbee.cli.sync import _sync_progress_printer
        from lilbee.progress import EventType, SyncDoneEvent

        con = MagicMock()
        cb = _sync_progress_printer(con)
        cb(EventType.DONE, SyncDoneEvent(added=1, updated=0, removed=0, failed=0, unchanged=0))
        con.print.assert_called_once()
        assert "Synced" in str(con.print.call_args)

    def test_file_start_wrong_type_raises(self):
        """_sync_progress_printer raises TypeError for wrong event type."""
        from lilbee.cli.sync import _sync_progress_printer
        from lilbee.progress import EventType, SyncDoneEvent

        con = MagicMock()
        cb = _sync_progress_printer(con)
        with pytest.raises(TypeError, match="Expected FileStartEvent"):
            cb(EventType.FILE_START, SyncDoneEvent(added=0, updated=0, removed=0, failed=0, unchanged=0))

    def test_done_wrong_type_raises(self):
        """_sync_progress_printer raises TypeError for wrong data type on DONE."""
        from lilbee.cli.sync import _sync_progress_printer
        from lilbee.progress import EventType, FileStartEvent

        con = MagicMock()
        cb = _sync_progress_printer(con)
        with pytest.raises(TypeError, match="Expected SyncDoneEvent"):
            cb(EventType.DONE, FileStartEvent(file="x", total_files=1, current_file=1))


class TestChatSyncCallback:
    def test_file_start_updates_status(self):
        """Background sync callback updates status on FILE_START."""
        from lilbee.cli.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType, FileStartEvent

        status = SyncStatus()
        cb = _chat_sync_callback(status)
        cb(EventType.FILE_START, FileStartEvent(file="test.md", total_files=3, current_file=1))
        assert "test.md" in status.text

    def test_extract_updates_status(self):
        """Background sync callback updates status on EXTRACT."""
        from lilbee.cli.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType, ExtractEvent

        status = SyncStatus()
        cb = _chat_sync_callback(status)
        cb(EventType.EXTRACT, ExtractEvent(file="scan.pdf", page=2, total_pages=5))
        assert "Vision OCR" in status.text
        assert "scan.pdf" in status.text

    def test_done_clears_status(self):
        """Background sync callback clears status on DONE."""
        from lilbee.cli.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType, SyncDoneEvent

        status = SyncStatus()
        status.text = "something"
        cb = _chat_sync_callback(status)
        with mock.patch("builtins.print"):
            cb(EventType.DONE, SyncDoneEvent(added=2, updated=0, removed=0, failed=0, unchanged=0))
        assert status.text == ""

    def test_file_start_wrong_type_raises(self):
        from lilbee.cli.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType, SyncDoneEvent

        status = SyncStatus()
        cb = _chat_sync_callback(status)
        with pytest.raises(TypeError, match="Expected FileStartEvent"):
            cb(EventType.FILE_START, SyncDoneEvent(added=0, updated=0, removed=0, failed=0, unchanged=0))

    def test_extract_wrong_type_raises(self):
        from lilbee.cli.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType, FileStartEvent

        status = SyncStatus()
        cb = _chat_sync_callback(status)
        with pytest.raises(TypeError, match="Expected ExtractEvent"):
            cb(EventType.EXTRACT, FileStartEvent(file="x", total_files=1, current_file=1))

    def test_done_wrong_type_raises(self):
        from lilbee.cli.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType, FileStartEvent

        status = SyncStatus()
        cb = _chat_sync_callback(status)
        with pytest.raises(TypeError, match="Expected SyncDoneEvent"):
            cb(EventType.DONE, FileStartEvent(file="x", total_files=1, current_file=1))


class TestSyncResultToJson:
    def test_non_sync_result_raises(self):
        """sync_result_to_json raises TypeError for non-SyncResult input."""
        from lilbee.cli.helpers import sync_result_to_json

        with pytest.raises(TypeError, match="Expected SyncResult"):
            sync_result_to_json("not a SyncResult")
