"""Tests for the CLI interface using typer's test runner."""

from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.cli import _make_completer, _QuitChat, app, console

runner = CliRunner()

_SYNC_NOOP = {
    "added": [],
    "updated": [],
    "removed": [],
    "unchanged": 0,
    "failed": [],
}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all CLI tests."""
    import lilbee.config as cfg
    import lilbee.store as store_mod

    orig_docs, orig_db, orig_data = cfg.DOCUMENTS_DIR, cfg.LANCEDB_DIR, cfg.DATA_DIR

    cfg.DOCUMENTS_DIR = tmp_path / "documents"
    cfg.DOCUMENTS_DIR.mkdir()
    cfg.DATA_DIR = tmp_path / "data"
    cfg.LANCEDB_DIR = tmp_path / "data" / "lancedb"
    store_mod.LANCEDB_DIR = cfg.LANCEDB_DIR

    yield tmp_path

    cfg.DOCUMENTS_DIR = orig_docs
    cfg.DATA_DIR = orig_data
    cfg.LANCEDB_DIR = orig_db
    store_mod.LANCEDB_DIR = orig_db


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

    def test_status_with_indexed_docs(self, isolated_env):
        from lilbee.store import upsert_source

        upsert_source("test.pdf", "abc123", 10)
        result = runner.invoke(app, ["status"])
        assert "test.pdf" in result.output
        assert "10" in result.output


class TestSync:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_empty(self, _e, _eb):
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 0" in result.output

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_sync_with_file(self, _e, _eb, isolated_env):
        import lilbee.config as cfg

        (cfg.DOCUMENTS_DIR / "test.txt").write_text("Hello world content.")
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Added: 1" in result.output

    @mock.patch(
        "lilbee.ingest.sync",
        return_value={
            "added": [],
            "updated": [],
            "removed": [],
            "unchanged": 0,
            "failed": ["bad.txt"],
        },
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

        import lilbee.config as cfg

        result = runner.invoke(app, ["add", str(src_file)])
        assert result.exit_code == 0
        assert "Copied 1" in result.output
        assert (cfg.DOCUMENTS_DIR / "manual.txt").exists()

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    def test_add_directory(self, _e, _eb, isolated_env, tmp_path):
        """Adding a directory recursively copies it."""
        src_dir = tmp_path / "source" / "docs"
        src_dir.mkdir(parents=True)
        (src_dir / "file1.txt").write_text("Content 1")
        (src_dir / "file2.txt").write_text("Content 2")

        import lilbee.config as cfg

        result = runner.invoke(app, ["add", str(src_dir)])
        assert result.exit_code == 0
        assert (cfg.DOCUMENTS_DIR / "docs" / "file1.txt").exists()
        assert (cfg.DOCUMENTS_DIR / "docs" / "file2.txt").exists()

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
        """Re-adding a directory updates content."""
        import lilbee.config as cfg

        src_dir = tmp_path / "source" / "docs"
        src_dir.mkdir(parents=True)
        (src_dir / "file1.txt").write_text("Version 1")

        runner.invoke(app, ["add", str(src_dir)])

        # Update content and re-add
        (src_dir / "file1.txt").write_text("Version 2")
        result = runner.invoke(app, ["add", str(src_dir)])
        assert result.exit_code == 0
        assert (cfg.DOCUMENTS_DIR / "docs" / "file1.txt").read_text() == "Version 2"


class TestAsk:
    @mock.patch("lilbee.query.ask_stream")
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_ask_prints_response(self, _sync, mock_stream):
        mock_stream.return_value = iter(["Hello", " world"])
        result = runner.invoke(app, ["ask", "test question"])
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream")
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
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
        return_value={
            "added": ["new.pdf"],
            "updated": [],
            "removed": [],
            "unchanged": 0,
            "failed": [],
        },
    )
    @mock.patch("lilbee.query.ask_stream", return_value=iter(["answer"]))
    def test_auto_sync_prints_summary(self, _stream, _sync):
        result = runner.invoke(app, ["ask", "test"])
        assert result.exit_code == 0
        assert "Synced:" in result.output


class TestChat:
    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_chat_quit(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="question\n/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_chat_slash_quit(self, _sync, _stream):
        """Bare /quit exits immediately."""
        result = runner.invoke(app, ["chat"], input="/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_chat_empty_input_skipped(self, _sync, _stream):
        result = runner.invoke(app, ["chat"], input="\n/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_chat_eof_exits(self, _sync):
        # Empty input simulates EOF
        result = runner.invoke(app, ["chat"], input="")
        assert result.exit_code == 0

    @mock.patch("lilbee.query.ask_stream")
    @mock.patch(
        "lilbee.ingest.sync",
        return_value={"added": [], "updated": [], "removed": [], "unchanged": 0},
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


class TestApplyOverrides:
    def test_data_dir_override(self, tmp_path):
        import lilbee.config as cfg
        from lilbee.cli import _apply_overrides

        _apply_overrides(data_dir=tmp_path)
        assert tmp_path / "documents" == cfg.DOCUMENTS_DIR

    def test_model_override(self):
        import lilbee.config as cfg
        from lilbee.cli import _apply_overrides

        _apply_overrides(model="phi3")
        assert cfg.CHAT_MODEL == "phi3"

    def test_none_values_are_noop(self):
        import lilbee.config as cfg
        from lilbee.cli import _apply_overrides

        original_model = cfg.CHAT_MODEL
        _apply_overrides(data_dir=None, model=None)
        assert original_model == cfg.CHAT_MODEL


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
    """Test _dispatch_slash in isolation."""

    def test_non_slash_returns_false(self):
        from lilbee.cli import _dispatch_slash

        assert _dispatch_slash("hello", console) is False

    def test_known_command_returns_true(self):
        from lilbee.cli import _dispatch_slash

        assert _dispatch_slash("/help", console) is True

    def test_unknown_command_prints_error(self):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli import _dispatch_slash

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        result = _dispatch_slash("/foobar", con)
        assert result is True
        assert "Unknown command" in buf.getvalue()

    def test_slash_help_output(self):
        from io import StringIO

        from rich.console import Console as RichConsole

        from lilbee.cli import _dispatch_slash

        buf = StringIO()
        con = RichConsole(file=buf, force_terminal=False, no_color=True)
        _dispatch_slash("/help", con)
        output = buf.getvalue()
        assert "/status" in output
        assert "/add" in output
        assert "/help" in output
        assert "/quit" in output


class TestSlashStatus:
    """Test /status inside the chat loop."""

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_status_in_chat(self, _sync):
        result = runner.invoke(app, ["chat"], input="/status\n/quit\n")
        assert result.exit_code == 0
        assert "Documents:" in result.output


class TestSlashHelp:
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_help_in_chat(self, _sync):
        result = runner.invoke(app, ["chat"], input="/help\n/quit\n")
        assert result.exit_code == 0
        assert "/status" in result.output


class TestSlashAdd:
    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_add_with_path(self, _sync, _e, _eb, isolated_env, tmp_path):
        src = tmp_path / "source" / "test.txt"
        src.parent.mkdir()
        src.write_text("content")

        # Re-mock sync for the _add_paths call (it calls sync() internally)
        _sync.return_value = {
            "added": ["test.txt"],
            "updated": [],
            "removed": [],
            "unchanged": 0,
            "failed": [],
        }

        result = runner.invoke(app, ["chat"], input=f"/add {src}\n/quit\n")
        assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_add_nonexistent_path(self, _sync):
        result = runner.invoke(app, ["chat"], input="/add /tmp/nope_xyz_999\n/quit\n")
        assert result.exit_code == 0
        assert "Path not found" in result.output

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
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

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_add_interactive_eof(self, _sync):
        """When user hits Ctrl+C / EOF in prompt_toolkit, /add exits gracefully."""
        with mock.patch("prompt_toolkit.prompt", side_effect=EOFError):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_add_interactive_empty_input(self, _sync):
        """When user enters empty path in prompt_toolkit, /add does nothing."""
        with mock.patch("prompt_toolkit.prompt", return_value="   "):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_slash_add_interactive_nonexistent(self, _sync):
        """When user enters nonexistent path in prompt_toolkit prompt."""
        with mock.patch("prompt_toolkit.prompt", return_value="/tmp/nope_xyz_999"):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0
            assert "Path not found" in result.output

    @mock.patch("lilbee.embedder.embed_batch", return_value=[[0.1] * 768])
    @mock.patch("lilbee.embedder.embed", return_value=[0.1] * 768)
    @mock.patch("lilbee.ingest.sync")
    def test_slash_add_interactive_success(self, mock_sync, _e, _eb, tmp_path):
        """Full /add interactive flow with successful file."""
        src = tmp_path / "source" / "doc.txt"
        src.parent.mkdir()
        src.write_text("some content")
        mock_sync.return_value = {
            "added": ["doc.txt"],
            "updated": [],
            "removed": [],
            "unchanged": 0,
            "failed": [],
        }

        with mock.patch("prompt_toolkit.prompt", return_value=str(src)):
            result = runner.invoke(app, ["chat"], input="/add\n/quit\n")
            assert result.exit_code == 0


class TestSlashUnknown:
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_unknown_slash_command(self, _sync):
        result = runner.invoke(app, ["chat"], input="/foobar\n/quit\n")
        assert result.exit_code == 0
        assert "Unknown command" in result.output


class TestDefaultInvokesChatLoop:
    """Invoking `lilbee` with no subcommand enters chat mode."""

    @mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello"]))
    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_default_chat(self, _sync, _stream):
        result = runner.invoke(app, [], input="question\n/quit\n")
        assert result.exit_code == 0
        assert "lilbee chat" in result.output

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
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
        completer = _make_completer()
        return [c.text for c in completer.get_completions(doc, None)]

    def test_slash_shows_all_commands(self):
        results = self._complete("/")
        assert "/status" in results
        assert "/add" in results
        assert "/help" in results
        assert "/quit" in results

    def test_slash_s_narrows(self):
        results = self._complete("/s")
        assert results == ["/status"]

    def test_add_path_delegates(self):
        results = self._complete("/add /")
        assert len(results) > 0

    def test_plain_text_no_completions(self):
        results = self._complete("hello")
        assert results == []


class TestQuitChat:
    """Test _QuitChat sentinel."""

    def test_quit_chat_is_exception(self):
        assert issubclass(_QuitChat, Exception)

    def test_slash_quit_raises(self):
        from lilbee.cli import _handle_slash_quit

        with pytest.raises(_QuitChat):
            _handle_slash_quit("", console)


class TestPromptSessionBranch:
    """Test that TTY branch uses PromptSession."""

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
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
            from lilbee.cli import _chat_loop

            _chat_loop(console)
            mock_session.prompt.assert_called()

    @mock.patch("lilbee.ingest.sync", return_value=_SYNC_NOOP)
    def test_tty_import_error_falls_back(self, _sync):
        """When prompt_toolkit import fails in TTY mode, falls back to con.input."""
        from lilbee.cli import _chat_loop

        mock_con = mock.MagicMock()
        mock_con.input.side_effect = EOFError

        with (
            mock.patch("sys.stdin") as mock_stdin,
            mock.patch.dict("sys.modules", {"prompt_toolkit": None}),
        ):
            mock_stdin.isatty.return_value = True
            _chat_loop(mock_con)
            # Verify it fell back to con.input (not PromptSession)
            mock_con.input.assert_called()
