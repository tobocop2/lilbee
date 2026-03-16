"""Tests for the CLI interface using typer's test runner."""

import json
import logging
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from lilbee.cli import (
    QuitChat,
    app,
    clean_result,
    console,
    get_version,
    list_ollama_models,
    make_completer,
)
from lilbee.config import cfg
from lilbee.ingest import SyncResult

runner = CliRunner()

_SYNC_NOOP = SyncResult()


@pytest.fixture(autouse=True)
def _skip_model_validation():
    """CLI tests never need real Ollama model validation or chat model checks."""
    with (
        mock.patch("lilbee.embedder.validate_model"),
        mock.patch("lilbee.models.ensure_chat_model"),
    ):
        yield


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
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.json_mode = False

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

    def test_status_with_indexed_docs(self, isolated_env):
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123", 10)
        result = runner.invoke(app, ["status"])
        assert "test.pdf" in result.output
        assert "10" in result.output


class TestSync:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_sync_empty(self, _sync):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 0" in result.output

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(added=["test.txt"]),
    )
    def test_sync_with_file(self, _sync, isolated_env):

        (cfg.documents_dir / "test.txt").write_text("Hello world content.")
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 1" in result.output

    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(failed=["bad.txt"]),
    )
    def test_sync_shows_failed(self, _sync):
        result = runner.invoke(app, ["sync"])
        assert "Failed: 1" in result.output
        assert "bad.txt" in result.output


class TestRebuild:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_rebuild_empty(self, _e, _eb):
        result = runner.invoke(app, ["rebuild"])
        assert result.exit_code == 0
        assert "Rebuilt:" in result.output


class TestAdd:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_single_file(self, _e, _eb, isolated_env, tmp_path):
        """Adding a single file copies it and ingests it."""
        src_file = tmp_path / "source" / "manual.txt"
        src_file.parent.mkdir()
        src_file.write_text("Engine oil capacity is 5 quarts.")

        result = runner.invoke(app, ["add", str(src_file)])
        assert result.exit_code == 0
        assert "Copied 1" in result.output
        assert (cfg.documents_dir / "manual.txt").exists()

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_directory(self, _e, _eb, isolated_env, tmp_path):
        """Adding a directory recursively copies it."""
        src_dir = tmp_path / "source" / "docs"
        src_dir.mkdir(parents=True)
        (src_dir / "file1.txt").write_text("Content 1")
        (src_dir / "file2.txt").write_text("Content 2")

        result = runner.invoke(app, ["add", str(src_dir)])
        assert result.exit_code == 0
        assert (cfg.documents_dir / "docs" / "file1.txt").exists()
        assert (cfg.documents_dir / "docs" / "file2.txt").exists()

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_multiple_paths(self, _e, _eb, isolated_env, tmp_path):
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

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_overwrites_existing_dir(self, _e, _eb, isolated_env, tmp_path):
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

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_warns_on_existing(self, _e, _eb, isolated_env, tmp_path):
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
    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_directory_skips_git_and_node_modules(self, _e, _eb, isolated_env, tmp_path):
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
    @mock.patch("lilbee.query.ask_stream")
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_prints_response(self, _sync, mock_stream):
        mock_stream.return_value = iter(["Hello", " world"])
        result = runner.invoke(app, ["ask", "test question"])
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream")
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_with_model_flag(self, _sync, mock_stream):
        mock_stream.return_value = iter(["answer"])
        result = runner.invoke(app, ["ask", "question", "--model", "llama3"])
        assert result.exit_code == 0


class TestDataDirFlag:
    def test_status_with_data_dir(self, tmp_path):
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "documents").mkdir()
        result = runner.invoke(app, ["status", "--data-dir", str(custom)])
        assert result.exit_code == 0

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_with_data_dir(self, _e, _eb, tmp_path):
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
    @mock.patch("lilbee.query.ask_stream", return_value=iter(["answer"]))
    def test_auto_sync_prints_summary(self, _stream, _sync):
        result = runner.invoke(app, ["ask", "test"])
        assert result.exit_code == 0
        assert "Synced:" in result.output


class TestChat:
    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello", " world"]))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_quit(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="question\n/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_slash_quit(self, _sync, _stream):
        """Bare /quit exits immediately."""
        result = runner.invoke(app, ["chat"], input="/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_empty_input_skipped(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="\n/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_eof_exits(self, _sync):
        # Empty input simulates EOF
        result = runner.invoke(app, ["chat"], input="")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream")
    @mock.patch(
        "lilbee.ingest.sync",
        new_callable=AsyncMock,
        return_value=SyncResult(),
    )
    def test_chat_passes_history(self, _sync, mock_stream):
        """Second question should include history from first exchange."""
        call_count = 0

        def fake_stream(question, history=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                assert history == []
            else:
                assert len(history) == 2
                assert history[0]["role"] == "user"
                assert history[1]["role"] == "assistant"
            return iter(["answer"])

        mock_stream.side_effect = fake_stream
        result = runner.invoke(app, ["chat"], input="first\nsecond\n/quit\n")
        assert result.exit_code == 0
        assert call_count == 2

    @mock.patch("lilbee.query.ask_stream")
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_model_not_found_recovers(self, _sync, mock_stream):
        """Model error in chat shows message and continues the loop."""

        def failing_gen(*_args, **_kwargs):
            raise RuntimeError("Model 'bad' not found")
            yield  # type: ignore[misc]  # makes this a generator

        mock_stream.side_effect = failing_gen
        result = runner.invoke(app, ["chat"], input="hello\n/quit\n")
        assert result.exit_code == 0
        assert "not found" in result.output


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


class TestDispatchSlash:
    """Test dispatch_slash in isolation."""

    def test_non_slash_returns_false(self):
        from lilbee.cli import dispatch_slash

        assert dispatch_slash("hello", console) is False

    def test_known_command_returns_true(self):
        from lilbee.cli import dispatch_slash

        assert dispatch_slash("/help", console) is True

    def test_unknown_command_prints_error(self):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli import dispatch_slash

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        result = dispatch_slash("/foobar", con)
        assert result is True
        assert "Unknown command" in buf.getvalue()

    def test_slash_help_output(self):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli import dispatch_slash

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        dispatch_slash("/help", con)
        output = buf.getvalue()
        assert "/status" in output
        assert "/add" in output
        assert "/model" in output
        assert "/vision" in output
        assert "/version" in output
        assert "/reset" in output
        assert "/help" in output
        assert "/quit" in output


class TestSlashStatus:
    """Test /status inside the chat loop."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_status_in_chat(self, _sync):
        result = runner.invoke(app, ["chat"], input="/status\n/quit\n")
        assert result.exit_code == 0
        assert "Documents:" in result.output


class TestSlashHelp:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_help_in_chat(self, _sync):
        result = runner.invoke(app, ["chat"], input="/help\n/quit\n")
        assert result.exit_code == 0
        assert "/status" in result.output


class TestSlashAdd:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_add_with_path(self, _sync, _e, _eb, isolated_env, tmp_path):
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")

        # Re-mock sync for the add_paths call (it calls sync() internally)
        _sync.return_value = SyncResult(added=["test.txt"])

        result = runner.invoke(app, ["chat"], input=f"/add {src}\n/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_add_nonexistent_path(self, _sync):
        result = runner.invoke(app, ["chat"], input="/add /tmp/nope_xyz_999\n/quit\n")
        assert result.exit_code == 0
        assert "Path not found" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_add_interactive_import_error(self, _sync):
        """When prompt_toolkit import fails, /add with no args does nothing."""
        import sys

        orig = sys.modules.get("prompt_toolkit")
        sys.modules["prompt_toolkit"] = None  # type: ignore[assignment]
        try:
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0
        finally:
            if orig is not None:
                sys.modules["prompt_toolkit"] = orig
            else:
                sys.modules.pop("prompt_toolkit", None)

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_add_interactive_eof(self, _sync):
        """When user hits Ctrl+C / EOF in prompt_toolkit, /add exits gracefully."""
        with mock.patch("prompt_toolkit.prompt", side_effect=EOFError):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_add_interactive_empty_input(self, _sync):
        """When user enters empty path in prompt_toolkit, /add does nothing."""
        with mock.patch("prompt_toolkit.prompt", return_value="   "):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_add_interactive_nonexistent(self, _sync):
        """When user enters nonexistent path in prompt_toolkit prompt."""
        with mock.patch("prompt_toolkit.prompt", return_value="/tmp/nope_xyz_999"):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0
            assert "Path not found" in result.output

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock)
    def test_slash_add_interactive_success(self, mock_sync, _e, _eb, tmp_path):
        """Full /add interactive flow with successful file."""
        src = tmp_path / "source" / "doc.txt"
        src.parent.mkdir()
        src.write_text("some content")
        mock_sync.return_value = SyncResult(added=["doc.txt"])

        with mock.patch("prompt_toolkit.prompt", return_value=str(src)):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0


class TestSlashModel:
    """Test /model slash command."""

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_model_interactive_cancel(self, _ram, _disk):
        """Empty input cancels the interactive picker without changing model."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("builtins.input", return_value=""):
            handle_slash_model("", con)
        output = buf.getvalue()
        assert "Current model:" in output

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=["qwen3:8b"])
    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_model_interactive_picks_installed(self, _ram, _disk, _models):
        """Picking an already-installed model switches without pulling."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        original = cfg.chat_model
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        try:
            with (
                mock.patch("builtins.input", return_value="4"),
                mock.patch("lilbee.settings.set_value") as mock_set,
            ):
                handle_slash_model("", con)
            assert cfg.chat_model == "qwen3:8b"
            output = buf.getvalue()
            assert "Switched to model" in output
            mock_set.assert_called_once_with(cfg.data_root, "chat_model", "qwen3:8b")
        finally:
            cfg.chat_model = original

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[])
    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    @mock.patch("lilbee.models.pull_with_progress")
    @mock.patch("lilbee.settings.set_value")
    def test_model_interactive_pulls_uninstalled(self, _save, mock_pull, _ram, _disk, _models):
        """Picking an uninstalled model triggers a pull."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        original = cfg.chat_model
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        try:
            with mock.patch("builtins.input", return_value="1"):
                handle_slash_model("", con)
            mock_pull.assert_called_once_with("qwen3:1.7b")
            output = buf.getvalue()
            assert "Switched to model" in output
        finally:
            cfg.chat_model = original

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_model_interactive_invalid_input(self, _ram, _disk):
        """Non-numeric input shows an error."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("builtins.input", return_value="abc"):
            handle_slash_model("", con)
        output = buf.getvalue()
        assert "Enter a number" in output

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_model_interactive_out_of_range(self, _ram, _disk):
        """Out-of-range number shows an error."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("builtins.input", return_value="99"):
            handle_slash_model("", con)
        output = buf.getvalue()
        assert "Enter a number" in output

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_model_interactive_eof(self, _ram, _disk):
        """EOF during input cancels gracefully."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("builtins.input", side_effect=EOFError):
            handle_slash_model("", con)
        # Should not raise

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models", return_value=["llama3:latest", "mistral:latest"]
    )
    def test_model_switches(self, _models):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        original = cfg.chat_model
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        try:
            with mock.patch("lilbee.settings.set_value") as mock_set:
                handle_slash_model("llama3", con)
                assert cfg.chat_model == "llama3:latest"
                output = buf.getvalue()
                assert "Switched to model" in output
                assert "(saved)" in output
                mock_set.assert_called_once_with(cfg.data_root, "chat_model", "llama3:latest")
        finally:
            cfg.chat_model = original

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models", return_value=["llama3:latest", "mistral:latest"]
    )
    def test_model_rejects_unknown(self, _models):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_model

        original = cfg.chat_model
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        try:
            handle_slash_model("nonexistent", con)
            assert original == cfg.chat_model
            output = buf.getvalue()
            assert "Unknown model" in output
            assert "Available:" in output
        finally:
            cfg.chat_model = original

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models", return_value=["phi3:latest", "mistral:latest"]
    )
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_model_switch_inchat_loop(self, _sync, _models):

        original = cfg.chat_model
        try:
            result = runner.invoke(app, ["chat"], input="/model phi3\n/quit\n")
            assert result.exit_code == 0
            assert "Switched to model" in result.output
        finally:
            cfg.chat_model = original


class TestSlashVision:
    """Test /vision slash command — model selection, disable, and status."""

    def test_vision_off_disables(self):
        """Test /vision off disables vision and preserves model in TOML."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        cfg.vision_model = "some-model"
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("lilbee.settings.set_value") as mock_set:
            handle_slash_vision("off", con)
        assert cfg.vision_model == ""
        mock_set.assert_called_once_with(cfg.data_root, "vision_model", "")
        output = buf.getvalue()
        assert "disabled" in output
        assert "(saved)" in output

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=["test-vision:latest"])
    def test_vision_name_sets_and_enables(self, _models):
        """Test /vision <name> sets model and enables vision."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("lilbee.settings.set_value"):
            handle_slash_vision("test-vision", con)
        assert cfg.vision_model == "test-vision:latest"
        output = buf.getvalue()
        assert "Vision model set to" in output
        assert "(saved)" in output

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models", return_value=["model-a:latest", "model-b:latest"]
    )
    def test_vision_name_rejects_unknown(self, _models):
        """Test /vision <name> rejects unknown models."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        cfg.vision_model = "original-model"
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        handle_slash_vision("nonexistent", con)
        assert cfg.vision_model == "original-model"
        output = buf.getvalue()
        assert "Unknown model" in output
        assert "Available:" in output

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[])
    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_vision_bare_shows_status_enabled(self, _ram, _disk, _models):
        """Test bare /vision shows enabled status then picker."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        cfg.vision_model = "test-model"
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with (
            mock.patch("builtins.input", return_value=""),
            mock.patch("lilbee.settings.set_value"),
        ):
            handle_slash_vision("", con)
        output = buf.getvalue()
        assert "test-model" in output

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[])
    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_vision_bare_shows_disabled(self, _ram, _disk, _models):
        """Test bare /vision shows disabled when no model set."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        cfg.vision_model = ""
        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with (
            mock.patch("builtins.input", return_value=""),
            mock.patch("lilbee.settings.set_value"),
        ):
            handle_slash_vision("", con)
        output = buf.getvalue()
        assert "disabled" in output

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models",
        return_value=["maternion/LightOnOCR-2:latest"],
    )
    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_vision_interactive_picks_installed(self, _ram, _disk, _models):
        """Picking an already-installed model switches without pulling."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with (
            mock.patch("builtins.input", return_value="1"),
            mock.patch("lilbee.settings.set_value"),
        ):
            handle_slash_vision("", con)
        assert cfg.vision_model == "maternion/LightOnOCR-2:latest"
        output = buf.getvalue()
        assert "Vision model set to" in output

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[])
    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    @mock.patch("lilbee.models.pull_with_progress")
    @mock.patch("lilbee.settings.set_value")
    def test_vision_interactive_pulls_uninstalled(self, _save, mock_pull, _ram, _disk, _models):
        """Picking an uninstalled model triggers a pull."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch("builtins.input", return_value="1"):
            handle_slash_vision("", con)
        mock_pull.assert_called_once_with("maternion/LightOnOCR-2:latest")
        output = buf.getvalue()
        assert "Vision model set to" in output

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_vision_interactive_invalid_input(self, _ram, _disk):
        """Non-numeric input shows an error."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with (
            mock.patch("builtins.input", return_value="abc"),
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
        ):
            handle_slash_vision("", con)
        output = buf.getvalue()
        assert "Enter a number" in output

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_vision_interactive_out_of_range(self, _ram, _disk):
        """Out-of-range number shows an error."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with (
            mock.patch("builtins.input", return_value="99"),
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
        ):
            handle_slash_vision("", con)
        output = buf.getvalue()
        assert "Enter a number" in output

    @mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0)
    @mock.patch("lilbee.models.get_system_ram_gb", return_value=8.0)
    def test_vision_interactive_eof(self, _ram, _disk):
        """EOF during input cancels gracefully."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_vision

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with (
            mock.patch("builtins.input", side_effect=EOFError),
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
        ):
            handle_slash_vision("", con)
        # Should not raise

    def test_vision_help_listed(self):
        """Test /vision appears in help output."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli import dispatch_slash

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        dispatch_slash("/help", con)
        output = buf.getvalue()
        assert "/vision" in output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_vision_off_inchat_loop(self, _sync):
        """Test /vision off works in the chat loop."""
        cfg.vision_model = "some-model"
        result = runner.invoke(app, ["chat"], input="/vision off\n/quit\n")
        assert result.exit_code == 0
        assert "disabled" in result.output

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models", return_value=["phi3:latest", "mistral:latest"]
    )
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_vision_name_inchat_loop(self, _sync, _models):
        """Test /vision <name> works in the chat loop."""
        result = runner.invoke(app, ["chat"], input="/vision phi3\n/quit\n")
        assert result.exit_code == 0
        assert "Vision model set to" in result.output


class TestSlashVersion:
    """Test /version slash command."""

    def test_version_shows_version(self):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_version

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        handle_slash_version("", con)
        output = buf.getvalue()
        assert "lilbee" in output
        assert get_version() in output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_version_inchat_loop(self, _sync):
        result = runner.invoke(app, ["chat"], input="/version\n/quit\n")
        assert result.exit_code == 0
        assert "lilbee" in result.output


class TestSlashUnknown:
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_unknown_slash_command(self, _sync):
        result = runner.invoke(app, ["chat"], input="/foobar\n/quit\n")
        assert result.exit_code == 0
        assert "Unknown command" in result.output


class TestDefaultInvokesChatLoop:
    """Invoking `lilbee` with no subcommand enters chat mode."""

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_default_chat(self, _sync, _stream):
        result = runner.invoke(app, [], input="question\n/quit\n")
        assert result.exit_code == 0
        assert "lilbee chat" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_default_slash_help(self, _sync):
        result = runner.invoke(app, [], input="/help\n/quit\n")
        assert result.exit_code == 0
        assert "/status" in result.output


# ---------------------------------------------------------------------------
# Completer tests
# ---------------------------------------------------------------------------


class TestLilbeeCompleter:
    """Unit tests for the chat completer."""

    def _complete(self, text: str) -> list[str]:
        from prompt_toolkit.document import Document

        doc = Document(text, len(text))
        completer = make_completer()
        return [c.text for c in completer.get_completions(doc, None)]

    def test_slash_shows_all_commands(self):
        results = self._complete("/")
        assert "/status" in results
        assert "/add" in results
        assert "/model" in results
        assert "/vision" in results
        assert "/version" in results
        assert "/reset" in results
        assert "/help" in results
        assert "/quit" in results

    def test_slash_s_narrows(self):
        results = self._complete("/s")
        assert "/status" in results
        assert "/settings" in results
        assert "/set" in results

    def test_add_path_delegates(self):
        results = self._complete("/add /")
        assert len(results) > 0

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models",
        return_value=["llama3:latest", "mistral:latest", "phi3:latest"],
    )
    def test_model_prefix_completes(self, _models):
        results = self._complete("/model ")
        assert "llama3:latest" in results
        assert "mistral:latest" in results
        assert "phi3:latest" in results

    @mock.patch(
        "lilbee.cli.chat.list_ollama_models",
        return_value=["llama3:latest", "mistral:latest"],
    )
    def test_model_prefix_filters(self, _models):
        results = self._complete("/model ll")
        assert results == ["llama3:latest"]

    @mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[])
    def test_model_prefix_no_models(self, _models):
        results = self._complete("/model ")
        assert results == []

    def test_vision_prefix_completes(self):
        results = self._complete("/vision ")
        assert results[0] == "off"
        assert "maternion/LightOnOCR-2:latest" in results

    def test_vision_prefix_filters_off(self):
        results = self._complete("/vision of")
        assert results == ["off"]

    def test_vision_prefix_filters_model(self):
        results = self._complete("/vision gl")
        assert results == ["glm-ocr:latest"]

    def test_plain_text_no_completions(self):
        results = self._complete("hello")
        assert results == []


class TestListOllamaModels:
    """Test list_ollama_models helper."""

    def test_returns_model_names_with_tags(self):
        mock_model = mock.MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [mock_model]
        with mock.patch("ollama.list", return_value=mock_response):
            assert list_ollama_models() == ["llama3:latest"]

    def test_returns_empty_on_error(self):
        with mock.patch("ollama.list", side_effect=ConnectionError("not running")):
            assert list_ollama_models() == []

    def test_excludes_embedding_model(self):
        chat = mock.MagicMock()
        chat.model = "llama3:latest"
        embed = mock.MagicMock()
        embed.model = "nomic-embed-text:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [chat, embed]
        with mock.patch("ollama.list", return_value=mock_response):
            result = list_ollama_models()
            assert result == ["llama3:latest"]
            assert "nomic-embed-text:latest" not in result

    def test_exclude_vision_filters_vision_catalog(self):
        chat = mock.MagicMock()
        chat.model = "llama3:latest"
        vision = mock.MagicMock()
        vision.model = "maternion/LightOnOCR-2:latest"
        mock_response = mock.MagicMock()
        mock_response.models = [chat, vision]
        with mock.patch("ollama.list", return_value=mock_response):
            result = list_ollama_models(exclude_vision=True)
            assert result == ["llama3:latest"]
            assert "maternion/LightOnOCR-2:latest" not in result


class TestQuitChat:
    """Test QuitChat sentinel."""

    def test_quit_chat_is_exception(self):
        assert issubclass(QuitChat, Exception)

    def test_slash_quit_raises(self):
        from lilbee.cli.chat import handle_slash_quit

        with pytest.raises(QuitChat):
            handle_slash_quit("", console)


class TestPromptSessionBranch:
    """Test that TTY branch uses PromptSession."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_tty_uses_prompt_session(self, _sync):
        mock_session = mock.MagicMock()
        mock_session.prompt.side_effect = ["/quit"]
        mock_ps_cls = mock.MagicMock(return_value=mock_session)

        with (
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch.dict(
                "sys.modules",
                {"prompt_toolkit": mock.MagicMock(PromptSession=mock_ps_cls)},
            ),
        ):
            mock_stdin.isatty.return_value = True
            from lilbee.cli import chat_loop

            chat_loop(console)
            mock_session.prompt.assert_called()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_tty_import_error_falls_back(self, _sync):
        """When prompt_toolkit import fails in TTY mode, falls back to con.input."""
        from lilbee.cli import chat_loop

        mock_con = mock.MagicMock()
        mock_con.input.side_effect = EOFError

        with (
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch.dict("sys.modules", {"prompt_toolkit": None}),
        ):
            mock_stdin.isatty.return_value = True
            chat_loop(mock_con)
            # Verify it fell back to con.input (not PromptSession)
            mock_con.input.assert_called()


# ---------------------------------------------------------------------------
# JSON output infrastructure tests (Task 1)
# ---------------------------------------------------------------------------


class TestCleanResult:
    def test_strips_vector(self):
        result = clean_result({"source": "a.pdf", "vector": [0.1, 0.2], "chunk": "hi"})
        assert "vector" not in result
        assert result["source"] == "a.pdf"

    def test_renames_distance(self):
        result = clean_result({"_distance": 0.42, "chunk": "hi"})
        assert "distance" in result
        assert "_distance" not in result
        assert result["distance"] == 0.42

    def test_passthrough_other_fields(self):
        result = clean_result({"source": "a.pdf", "chunk": "hi", "page_start": 1})
        assert result == {"source": "a.pdf", "chunk": "hi", "page_start": 1}

    def test_renames_relevance_score(self):
        result = clean_result({"_relevance_score": 0.85, "chunk": "hi", "vector": [0.1]})
        assert "relevance_score" in result
        assert "_relevance_score" not in result
        assert "vector" not in result
        assert result["relevance_score"] == 0.85

    def test_relevance_score_strips_distance(self):
        result = clean_result(
            {"_relevance_score": 0.85, "_distance": 0.3, "chunk": "hi", "vector": [0.1]}
        )
        assert "relevance_score" in result
        assert "distance" not in result
        assert "_distance" not in result


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
    {
        "source": "manual.pdf",
        "content_type": "pdf",
        "page_start": 5,
        "page_end": 5,
        "line_start": 0,
        "line_end": 0,
        "chunk": "The engine oil capacity is 5 quarts.",
        "chunk_index": 0,
        "_distance": 0.25,
        "vector": [0.1] * 768,
    },
]


class TestSearch:
    @mock.patch("lilbee.query.search_context", return_value=_MOCK_SEARCH_RESULTS)
    def test_search_json_with_results(self, _search):
        result = runner.invoke(app, ["--json", "search", "engine oil"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "search"
        assert data["query"] == "engine oil"
        assert len(data["results"]) == 1
        assert "vector" not in data["results"][0]
        assert "distance" in data["results"][0]

    @mock.patch("lilbee.query.search_context", return_value=[])
    def test_search_json_empty_results(self, _search):
        result = runner.invoke(app, ["--json", "search", "nothing"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["results"] == []

    @mock.patch("lilbee.query.search_context", return_value=_MOCK_SEARCH_RESULTS)
    def test_search_human_output(self, _search):
        result = runner.invoke(app, ["search", "engine oil"])
        assert result.exit_code == 0
        assert "manual.pdf" in result.output

    @mock.patch(
        "lilbee.query.search_context",
        return_value=[{**_MOCK_SEARCH_RESULTS[0], "chunk": "x" * 100}],
    )
    def test_search_human_truncates_long_chunks(self, _search):
        result = runner.invoke(app, ["search", "test"])
        assert result.exit_code == 0
        # Chunk is truncated (100 chars doesn't all appear, Rich truncates with …)
        assert result.output.count("x") < 100

    @mock.patch("lilbee.query.search_context", return_value=[])
    def test_search_human_no_results(self, _search):
        result = runner.invoke(app, ["search", "nothing"])
        assert result.exit_code == 0
        assert "No results found" in result.output

    @mock.patch(
        "lilbee.query.search_context",
        return_value=[{**_MOCK_SEARCH_RESULTS[0], "_relevance_score": 0.85}],
    )
    def test_search_human_hybrid_shows_score(self, _search):
        result = runner.invoke(app, ["search", "engine oil"])
        assert result.exit_code == 0
        assert "Score" in result.output
        assert "0.85" in result.output

    @mock.patch(
        "lilbee.query.search_context",
        return_value=[{**_MOCK_SEARCH_RESULTS[0], "_relevance_score": 0.85}],
    )
    def test_search_json_hybrid_has_relevance_score(self, _search):
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

    def test_remove_existing_source(self, isolated_env):
        from lilbee.store import get_sources, upsert_source

        upsert_source("test.pdf", "abc123", 10)
        assert len(get_sources()) == 1

        result = runner.invoke(app, ["remove", "test.pdf"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "test.pdf" in result.output
        assert len(get_sources()) == 0

    def test_remove_nonexistent_source(self):
        result = runner.invoke(app, ["remove", "nope.pdf"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_remove_multiple_sources(self, isolated_env):
        from lilbee.store import get_sources, upsert_source

        upsert_source("a.pdf", "hash1", 5)
        upsert_source("b.pdf", "hash2", 3)

        result = runner.invoke(app, ["remove", "a.pdf", "b.pdf"])
        assert result.exit_code == 0
        assert "a.pdf" in result.output
        assert "b.pdf" in result.output
        assert len(get_sources()) == 0

    def test_remove_mixed_existing_and_not(self, isolated_env):
        from lilbee.store import get_sources, upsert_source

        upsert_source("a.pdf", "hash1", 5)

        result = runner.invoke(app, ["remove", "a.pdf", "nope.pdf"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "Not found" in result.output
        assert len(get_sources()) == 0

    def test_remove_with_delete_flag(self, isolated_env):
        from lilbee.store import upsert_source

        doc = cfg.documents_dir / "test.txt"
        doc.write_text("content")
        upsert_source("test.txt", "abc123", 1)

        result = runner.invoke(app, ["remove", "--delete", "test.txt"])
        assert result.exit_code == 0
        assert not doc.exists()

    def test_remove_json(self, isolated_env):
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123", 10)

        result = runner.invoke(app, ["--json", "remove", "test.pdf"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "remove"
        assert "test.pdf" in data["removed"]

    def test_remove_json_not_found(self):
        result = runner.invoke(app, ["--json", "remove", "nope.pdf"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["removed"] == []
        assert "nope.pdf" in data["not_found"]


class TestChunks:
    """Test chunks command."""

    def test_chunks_nonexistent_source(self):
        result = runner.invoke(app, ["chunks", "nope.pdf"])
        assert result.exit_code == 1
        assert "Source not found" in result.output

    def test_chunks_nonexistent_json(self):
        result = runner.invoke(app, ["--json", "chunks", "nope.pdf"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data

    def test_chunks_with_source(self, isolated_env):
        from lilbee.store import add_chunks, upsert_source

        upsert_source("test.txt", "abc123", 2)
        add_chunks(
            [
                {
                    "source": "test.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "First chunk content",
                    "chunk_index": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "source": "test.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Second chunk content",
                    "chunk_index": 1,
                    "vector": [0.2] * 768,
                },
            ]
        )
        result = runner.invoke(app, ["chunks", "test.txt"])
        assert result.exit_code == 0
        assert "2 chunks" in result.output
        assert "First chunk" in result.output

    def test_chunks_truncates_long_chunk(self, isolated_env):
        from lilbee.store import add_chunks, upsert_source

        upsert_source("long.txt", "abc123", 1)
        add_chunks(
            [
                {
                    "source": "long.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "x" * 200,
                    "chunk_index": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        result = runner.invoke(app, ["chunks", "long.txt"])
        assert result.exit_code == 0
        assert "..." in result.output

    def test_chunks_json(self, isolated_env):
        from lilbee.store import add_chunks, upsert_source

        upsert_source("test.txt", "abc123", 1)
        add_chunks(
            [
                {
                    "source": "test.txt",
                    "content_type": "text",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "Chunk content",
                    "chunk_index": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
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


class TestSlashReset:
    """Test /reset inside the chat loop."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_reset_confirms(self, _sync, isolated_env):

        (cfg.documents_dir / "doc.txt").write_text("content")

        result = runner.invoke(app, ["chat"], input="/reset\nyes\n/quit\n")
        assert result.exit_code == 0
        assert "Reset complete" in result.output
        assert list(cfg.documents_dir.iterdir()) == []

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_reset_eof_aborts(self, _sync):
        """EOF during /reset confirmation aborts gracefully."""
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli.chat import handle_slash_reset

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        with mock.patch.object(con, "input", side_effect=EOFError):
            handle_slash_reset("", con)
        assert "Aborted" in buf.getvalue()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_slash_reset_aborts(self, _sync, isolated_env):

        (cfg.documents_dir / "doc.txt").write_text("content")

        result = runner.invoke(app, ["chat"], input="/reset\nno\n/quit\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert (cfg.documents_dir / "doc.txt").exists()


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

    def test_status_json_with_sources(self, isolated_env):
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123hash", 10)
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
    def test_sync_json_empty(self, _sync):
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
    def test_sync_json_with_changes(self, _sync):
        result = runner.invoke(app, ["--json", "sync"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["added"] == ["new.txt"]
        assert data["removed"] == ["old.txt"]
        assert data["unchanged"] == 2


class TestRebuildJson:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_rebuild_json(self, _e, _eb):
        result = runner.invoke(app, ["--json", "rebuild"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["command"] == "rebuild"
        assert "ingested" in data


class TestAddJson:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_json(self, _e, _eb, isolated_env, tmp_path):
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
    @mock.patch("lilbee.query.ask_raw")
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_json(self, _sync, mock_ask_raw):
        from lilbee.query import AskResult

        mock_ask_raw.return_value = AskResult(
            answer="5 quarts",
            sources=[
                {
                    "source": "manual.pdf",
                    "content_type": "pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "oil",
                    "chunk_index": 0,
                    "_distance": 0.3,
                    "vector": [0.1],
                }
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

    @mock.patch("lilbee.query.ask_raw")
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_json_no_results(self, _sync, mock_ask_raw):
        from lilbee.query import AskResult

        mock_ask_raw.return_value = AskResult(answer="No relevant documents found.", sources=[])
        result = runner.invoke(app, ["--json", "ask", "anything"])
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data["sources"] == []
        assert "No relevant" in data["answer"]


class TestAskModelNotFound:
    """CLI should show a friendly error when the model doesn't exist."""

    @mock.patch("lilbee.query.ask_stream", side_effect=RuntimeError("Model 'bad' not found"))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_model_not_found_human(self, _sync, _stream):
        result = runner.invoke(app, ["ask", "hello"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @mock.patch("lilbee.query.ask_raw", side_effect=RuntimeError("Model 'bad' not found"))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_model_not_found_json(self, _sync, _raw):
        result = runner.invoke(app, ["--json", "ask", "hello"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "not found" in data["error"]


class TestOllamaUnavailable:
    """CLI commands should show friendly errors when Ollama is unreachable."""

    _ERR = RuntimeError("Cannot connect to Ollama: Connection refused")

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_sync_ollama_unavailable(self, _sync):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 1
        assert "Cannot connect to Ollama" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_sync_ollama_unavailable_json(self, _sync):
        result = runner.invoke(app, ["--json", "sync"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "Cannot connect to Ollama" in data["error"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_rebuild_ollama_unavailable(self, _sync):
        result = runner.invoke(app, ["rebuild"])
        assert result.exit_code == 1
        assert "Cannot connect to Ollama" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_rebuild_ollama_unavailable_json(self, _sync):
        result = runner.invoke(app, ["--json", "rebuild"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "Cannot connect to Ollama" in data["error"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_add_ollama_unavailable(self, _sync, isolated_env, tmp_path):
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        result = runner.invoke(app, ["add", str(src)])
        assert result.exit_code == 1
        assert "Cannot connect to Ollama" in result.output

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_add_ollama_unavailable_json(self, _sync, isolated_env, tmp_path):
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        result = runner.invoke(app, ["--json", "add", str(src)])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "Cannot connect to Ollama" in data["error"]

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, side_effect=_ERR)
    def test_auto_sync_ollama_unavailable(self, _sync):
        result = runner.invoke(app, ["ask", "hello"])
        assert result.exit_code == 1
        assert "Cannot connect to Ollama" in result.output


class TestEnsureChatModelWiring:
    """Verify that ask and chat call ensure_chat_model before running."""

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["answer"]))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_calls_ensure_chat_model(self, _sync, _stream):
        with mock.patch("lilbee.models.ensure_chat_model") as mock_ensure:
            runner.invoke(app, ["ask", "test"])
            mock_ensure.assert_called_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_calls_ensure_chat_model(self, _sync):
        with mock.patch("lilbee.models.ensure_chat_model") as mock_ensure:
            runner.invoke(app, ["chat"], input="/quit\n")
            mock_ensure.assert_called_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_default_calls_ensure_chat_model(self, _sync):
        """Bare `lilbee` (no subcommand) also calls ensure_chat_model."""
        with mock.patch("lilbee.models.ensure_chat_model") as mock_ensure:
            runner.invoke(app, [], input="/quit\n")
            mock_ensure.assert_called_once()

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["answer"]))
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_ask_calls_validate_model(self, _sync, _stream):
        with mock.patch("lilbee.embedder.validate_model") as mock_val:
            runner.invoke(app, ["ask", "test"])
            mock_val.assert_called_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_chat_calls_validate_model(self, _sync):
        with mock.patch("lilbee.embedder.validate_model") as mock_val:
            runner.invoke(app, ["chat"], input="/quit\n")
            mock_val.assert_called_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_default_calls_validate_model(self, _sync):
        with mock.patch("lilbee.embedder.validate_model") as mock_val:
            runner.invoke(app, [], input="/quit\n")
            mock_val.assert_called_once()


# ---------------------------------------------------------------------------
# --vision flag tests
# ---------------------------------------------------------------------------


class TestEnsureVisionModel:
    """Test _ensure_vision_model helper and --vision CLI flag."""

    def test_already_configured_and_installed(self):
        """No-op when vision model is set and installed."""
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = "test-vision"
        with mock.patch("lilbee.cli.chat.list_ollama_models", return_value=["test-vision:latest"]):
            _ensure_vision_model()
        assert cfg.vision_model == "test-vision:latest"

    def test_configured_but_not_installed_pulls(self):
        """Pulls the model when configured but not installed."""
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = "test-vision"
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("lilbee.models.pull_with_progress") as mock_pull,
        ):
            _ensure_vision_model()
        mock_pull.assert_called_once_with("test-vision:latest")
        assert cfg.vision_model == "test-vision:latest"

    def test_configured_pull_fails_gracefully(self):
        """Clears vision model when configured model can't be pulled."""
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = "test-vision"
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch(
                "lilbee.models.pull_with_progress",
                side_effect=Exception("pull failed"),
            ),
        ):
            _ensure_vision_model()
        assert cfg.vision_model == ""

    def test_configured_ollama_unreachable_keeps_model(self):
        """When Ollama is unreachable, keeps configured model for downstream handling."""
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = "test-vision"
        with mock.patch(
            "lilbee.cli.chat.list_ollama_models", side_effect=Exception("conn refused")
        ):
            _ensure_vision_model()
        # Model kept — downstream will surface the error during ingestion
        assert cfg.vision_model == "test-vision:latest"

    def test_not_configured_restores_from_toml(self):
        """Restores persisted model from TOML when cfg.vision_model is empty (e.g. --vision)."""
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = ""
        with (
            mock.patch("lilbee.settings.get", return_value="saved-vision:latest"),
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=["saved-vision:latest"]),
        ):
            _ensure_vision_model()
        assert cfg.vision_model == "saved-vision:latest"

    def test_not_configured_non_interactive_auto_picks(self):
        """Auto-picks and pulls in non-interactive mode."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_model = ModelInfo("auto-vision", 1.5, 4, "test")
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch("lilbee.models.pick_default_vision_model", return_value=fake_model),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("lilbee.models.pull_with_progress") as mock_pull,
            mock.patch("lilbee.settings.set_value") as mock_set,
        ):
            mock_stdin.isatty.return_value = False
            _ensure_vision_model()
        mock_pull.assert_called_once_with("auto-vision")
        assert cfg.vision_model == "auto-vision"
        mock_set.assert_called_once_with(cfg.data_root, "vision_model", "auto-vision")

    def test_not_configured_interactive_shows_picker(self):
        """Shows picker in interactive mode."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_catalog = (
            ModelInfo("picked-vision", 1.5, 4, "test"),
            ModelInfo("other-vision", 2.0, 4, "other"),
        )
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch("lilbee.models.display_vision_picker", return_value=fake_catalog[0]),
            mock.patch("lilbee.models.VISION_CATALOG", fake_catalog),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("lilbee.models.pull_with_progress") as mock_pull,
            mock.patch("lilbee.settings.set_value") as mock_set,
            mock.patch("builtins.input", return_value=""),
        ):
            mock_stdin.isatty.return_value = True
            _ensure_vision_model()
        mock_pull.assert_called_once_with("picked-vision")
        assert cfg.vision_model == "picked-vision"
        mock_set.assert_called_once_with(cfg.data_root, "vision_model", "picked-vision")

    def test_not_configured_interactive_choice_number(self):
        """Picks model by number in interactive mode."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_catalog = (
            ModelInfo("v1", 1.0, 4, "first"),
            ModelInfo("v2", 2.0, 4, "second"),
        )
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch(
                "lilbee.models.display_vision_picker",
                return_value=fake_catalog[0],
            ),
            mock.patch("lilbee.models.VISION_CATALOG", fake_catalog),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("lilbee.models.pull_with_progress"),
            mock.patch("lilbee.settings.set_value"),
            mock.patch("builtins.input", return_value="2"),
        ):
            mock_stdin.isatty.return_value = True
            _ensure_vision_model()
        assert cfg.vision_model == "v2"

    def test_not_configured_interactive_invalid_input(self):
        """Invalid input in interactive mode returns without setting model."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_catalog = (ModelInfo("rec-vision", 1.5, 4, "recommended"),)
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch(
                "lilbee.models.display_vision_picker",
                return_value=fake_catalog[0],
            ),
            mock.patch("lilbee.models.VISION_CATALOG", fake_catalog),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("builtins.input", return_value="abc"),
        ):
            mock_stdin.isatty.return_value = True
            _ensure_vision_model()
        assert cfg.vision_model == ""

    def test_not_configured_interactive_out_of_range(self):
        """Out-of-range choice in interactive mode returns without setting model."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_catalog = (ModelInfo("rec-vision", 1.5, 4, "recommended"),)
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch(
                "lilbee.models.display_vision_picker",
                return_value=fake_catalog[0],
            ),
            mock.patch("lilbee.models.VISION_CATALOG", fake_catalog),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("builtins.input", return_value="99"),
        ):
            mock_stdin.isatty.return_value = True
            _ensure_vision_model()
        assert cfg.vision_model == ""

    def test_not_configured_interactive_eof(self):
        """EOF during picker returns without setting model."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_catalog = (ModelInfo("rec-vision", 1.5, 4, "recommended"),)
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch(
                "lilbee.models.display_vision_picker",
                return_value=fake_catalog[0],
            ),
            mock.patch("lilbee.models.VISION_CATALOG", fake_catalog),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch("builtins.input", side_effect=EOFError),
        ):
            mock_stdin.isatty.return_value = True
            _ensure_vision_model()
        assert cfg.vision_model == ""

    def test_ollama_connection_fails_gracefully(self):
        """Continues without vision when Ollama is down and no model configured."""
        from lilbee.cli.commands import _ensure_vision_model

        cfg.vision_model = ""
        with mock.patch(
            "lilbee.cli.chat.list_ollama_models", side_effect=Exception("conn refused")
        ):
            _ensure_vision_model()
        assert cfg.vision_model == ""

    def test_non_interactive_pull_fails_gracefully(self):
        """Non-interactive auto-pick continues without vision when pull fails."""
        from lilbee.cli.commands import _ensure_vision_model
        from lilbee.models import ModelInfo

        cfg.vision_model = ""
        fake_model = ModelInfo("auto-vision", 1.5, 4, "test")
        with (
            mock.patch("lilbee.cli.chat.list_ollama_models", return_value=[]),
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch("lilbee.models.pick_default_vision_model", return_value=fake_model),
            mock.patch("lilbee.models.get_system_ram_gb", return_value=16.0),
            mock.patch("lilbee.models.get_free_disk_gb", return_value=50.0),
            mock.patch(
                "lilbee.models.pull_with_progress",
                side_effect=Exception("pull failed"),
            ),
        ):
            mock_stdin.isatty.return_value = False
            _ensure_vision_model()
        assert cfg.vision_model == ""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_vision_flag_on_sync(self, _sync):
        """--vision flag is accepted by sync command."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model") as mock_ensure:
            result = runner.invoke(app, ["sync", "--vision"])
            assert result.exit_code == 0
            mock_ensure.assert_called_once()

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_vision_flag_not_passed_on_sync(self, _sync):
        """Without --vision, _ensure_vision_model is not called."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model") as mock_ensure:
            result = runner.invoke(app, ["sync"])
            assert result.exit_code == 0
            mock_ensure.assert_not_called()

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_vision_flag_on_add(self, _e, _eb, isolated_env, tmp_path):
        """--vision flag is accepted by add command."""
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        with mock.patch("lilbee.cli.commands._ensure_vision_model") as mock_ensure:
            result = runner.invoke(app, ["add", "--vision", str(src)])
            assert result.exit_code == 0
            mock_ensure.assert_called_once()

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_vision_flag_on_rebuild(self, _e, _eb):
        """--vision flag is accepted by rebuild command."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model") as mock_ensure:
            result = runner.invoke(app, ["rebuild", "--vision"])
            assert result.exit_code == 0
            mock_ensure.assert_called_once()


class TestVisionTimeout:
    """Tests for --vision-timeout flag on sync, add, rebuild."""

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_vision_timeout_on_sync(self, _sync):
        """--vision-timeout sets cfg.vision_timeout for sync."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["sync", "--vision", "--vision-timeout=60"])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 60.0

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_vision_timeout_on_add(self, _e, _eb, isolated_env, tmp_path):
        """--vision-timeout sets cfg.vision_timeout for add."""
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["add", "--vision", "--vision-timeout=90", str(src)])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 90.0

    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_vision_timeout_on_rebuild(self, _e, _eb):
        """--vision-timeout sets cfg.vision_timeout for rebuild."""
        with mock.patch("lilbee.cli.commands._ensure_vision_model"):
            result = runner.invoke(app, ["rebuild", "--vision", "--vision-timeout=120"])
        assert result.exit_code == 0
        assert cfg.vision_timeout == 120.0

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_no_vision_timeout_leaves_default(self, _sync):
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
