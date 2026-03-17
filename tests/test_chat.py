"""Tests for chat slash commands: /settings, /set, and Ctrl+C cancellation."""

from io import StringIO
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from rich.console import Console as RichConsole
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.cli.chat.slash import (
    _SETTINGS_MAP,
    _format_setting_value,
    dispatch_slash,
    handle_slash_set,
    handle_slash_settings,
)
from lilbee.config import cfg
from lilbee.ingest import SyncResult

runner = CliRunner()
_SYNC_NOOP = SyncResult()


@pytest.fixture(autouse=True)
def _skip_model_validation():
    with (
        mock.patch("lilbee.embedder.validate_model"),
        mock.patch("lilbee.models.ensure_chat_model"),
    ):
        yield


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.json_mode = False
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _make_console() -> tuple[RichConsole, StringIO]:
    buf = StringIO()
    con = RichConsole(file=buf, force_terminal=False, no_color=True)
    return con, buf


class TestFormatSettingValue:
    def test_none_shows_not_set(self):
        assert "(not set)" in _format_setting_value(None)

    def test_empty_string_shows_not_set(self):
        assert "(not set)" in _format_setting_value("")

    def test_long_string_shows_length(self):
        result = _format_setting_value("x" * 80)
        assert "80 chars" in result

    def test_short_string_shows_value(self):
        assert _format_setting_value("hello") == "hello"

    def test_int_shows_value(self):
        assert _format_setting_value(42) == "42"

    def test_float_shows_value(self):
        assert _format_setting_value(0.7) == "0.7"

    def test_none_with_model_default(self):
        result = _format_setting_value(None, model_default="0.6")
        assert "model default: 0.6" in result

    def test_empty_with_model_default(self):
        result = _format_setting_value("", model_default="20")
        assert "model default: 20" in result

    def test_set_value_ignores_model_default(self):
        result = _format_setting_value(0.3, model_default="0.6")
        assert result == "0.3"
        assert "model default" not in result


class TestGetModelDefaults:
    @mock.patch("ollama.show")
    def test_parses_ollama_show_parameters(self, mock_show):
        from lilbee.cli.chat.slash import _get_model_defaults

        mock_resp = mock.Mock()
        mock_resp.parameters = (
            "temperature                    0.6\n"
            "top_k                          20\n"
            "top_p                          0.95\n"
            "repeat_penalty                 1\n"
        )
        mock_show.return_value = mock_resp
        defaults = _get_model_defaults()
        assert defaults["temperature"] == "0.6"
        assert defaults["top_k_sampling"] == "20"
        assert defaults["top_p"] == "0.95"
        assert defaults["repeat_penalty"] == "1"

    @mock.patch("ollama.show")
    def test_skips_non_setting_params(self, mock_show):
        from lilbee.cli.chat.slash import _get_model_defaults

        mock_resp = mock.Mock()
        mock_resp.parameters = (
            'stop                           "<|im_start|>"\ntemperature                    0.6\n'
        )
        mock_show.return_value = mock_resp
        defaults = _get_model_defaults()
        assert "stop" not in defaults
        assert defaults["temperature"] == "0.6"

    @mock.patch("ollama.show")
    def test_returns_empty_on_error(self, mock_show):
        from lilbee.cli.chat.slash import _get_model_defaults

        mock_show.side_effect = ConnectionError("connection refused")
        assert _get_model_defaults() == {}

    @mock.patch("ollama.show")
    def test_returns_empty_when_no_parameters(self, mock_show):
        from lilbee.cli.chat.slash import _get_model_defaults

        mock_resp = mock.Mock()
        mock_resp.parameters = None
        mock_show.return_value = mock_resp
        assert _get_model_defaults() == {}


class TestSlashSettings:
    @mock.patch("lilbee.cli.chat.slash._get_model_defaults", return_value={})
    def test_shows_all_settings(self, mock_defaults):
        con, buf = _make_console()
        handle_slash_settings("", con)
        output = buf.getvalue()
        for name in _SETTINGS_MAP:
            assert name in output

    @mock.patch("lilbee.cli.chat.slash._get_model_defaults", return_value={})
    def test_shows_current_chat_model(self, mock_defaults):
        con, buf = _make_console()
        handle_slash_settings("", con)
        assert cfg.chat_model in buf.getvalue()

    @mock.patch("lilbee.cli.chat.slash._get_model_defaults", return_value={})
    def test_shows_not_set_for_none_value(self, mock_defaults):
        cfg.temperature = None
        con, buf = _make_console()
        handle_slash_settings("", con)
        assert "(not set)" in buf.getvalue()

    @mock.patch(
        "lilbee.cli.chat.slash._get_model_defaults",
        return_value={"temperature": "0.6", "top_p": "0.95"},
    )
    def test_shows_model_defaults_for_unset_values(self, mock_defaults):
        cfg.temperature = None
        con, buf = _make_console()
        handle_slash_settings("", con)
        assert "model default: 0.6" in buf.getvalue()

    @mock.patch("lilbee.cli.chat.slash._get_model_defaults", return_value={})
    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_settings_in_chat_loop(self, mock_sync, mock_defaults):
        result = runner.invoke(app, ["chat"], input="/settings\n/quit\n")
        assert result.exit_code == 0
        assert "chat_model" in result.output


class TestSlashSet:
    def test_no_args_shows_usage(self):
        con, buf = _make_console()
        handle_slash_set("", con)
        assert "Usage" in buf.getvalue()

    def test_unknown_param_shows_error(self):
        con, buf = _make_console()
        handle_slash_set("nonexistent 42", con)
        assert "Unknown setting" in buf.getvalue()

    def test_show_current_value(self):
        cfg.temperature = 0.7
        con, buf = _make_console()
        handle_slash_set("temperature", con)
        assert "0.7" in buf.getvalue()

    def test_show_none_value(self):
        cfg.temperature = None
        con, buf = _make_console()
        handle_slash_set("temperature", con)
        assert "(not set)" in buf.getvalue()

    def test_set_float_value(self):
        con, buf = _make_console()
        handle_slash_set("temperature 0.3", con)
        assert cfg.temperature == 0.3
        assert "0.3" in buf.getvalue()
        assert "saved" in buf.getvalue()

    def test_set_int_value(self):
        con, buf = _make_console()
        handle_slash_set("seed 42", con)
        assert cfg.seed == 42
        assert "saved" in buf.getvalue()

    def test_set_string_value(self):
        con, buf = _make_console()
        handle_slash_set("chat_model llama3.2:latest", con)
        assert cfg.chat_model == "llama3.2:latest"
        assert "saved" in buf.getvalue()

    def test_clear_nullable_with_off(self):
        cfg.temperature = 0.5
        con, buf = _make_console()
        handle_slash_set("temperature off", con)
        assert cfg.temperature is None
        assert "cleared" in buf.getvalue()

    def test_clear_nullable_with_none(self):
        cfg.seed = 42
        con, _buf = _make_console()
        handle_slash_set("seed none", con)
        assert cfg.seed is None

    def test_clear_nullable_with_default(self):
        cfg.top_p = 0.9
        con, _buf = _make_console()
        handle_slash_set("top_p default", con)
        assert cfg.top_p is None

    def test_clear_non_nullable_rejected(self):
        con, buf = _make_console()
        handle_slash_set("chat_model off", con)
        assert "cannot be cleared" in buf.getvalue()
        assert cfg.chat_model != ""

    def test_clear_vision_model_sets_empty_string(self):
        cfg.vision_model = "llava:latest"
        con, buf = _make_console()
        handle_slash_set("vision_model off", con)
        assert cfg.vision_model == ""
        assert "cleared" in buf.getvalue()

    def test_invalid_float_shows_error(self):
        con, buf = _make_console()
        handle_slash_set("temperature abc", con)
        assert "Invalid" in buf.getvalue()

    def test_invalid_int_shows_error(self):
        con, buf = _make_console()
        handle_slash_set("seed abc", con)
        assert "Invalid" in buf.getvalue()

    def test_persists_to_settings_file(self):
        from lilbee import settings

        con, _buf = _make_console()
        handle_slash_set("temperature 0.5", con)
        assert settings.get(cfg.data_root, "temperature") == "0.5"

    def test_clear_removes_from_settings_file(self):
        from lilbee import settings

        settings.set_value(cfg.data_root, "temperature", "0.5")
        con, _buf = _make_console()
        handle_slash_set("temperature off", con)
        assert settings.get(cfg.data_root, "temperature") is None

    @mock.patch("lilbee.ingest.sync", new_callable=AsyncMock, return_value=_SYNC_NOOP)
    def test_set_in_chat_loop(self, mock_sync):
        result = runner.invoke(app, ["chat"], input="/set temperature 0.3\n/quit\n")
        assert result.exit_code == 0
        assert "saved" in result.output


class TestSlashSetValidation:
    def test_negative_temperature_rejected(self):
        con, buf = _make_console()
        handle_slash_set("temperature -0.5", con)
        assert "greater than or equal to 0" in buf.getvalue()
        assert cfg.temperature != -0.5

    def test_top_p_above_one_rejected(self):
        con, buf = _make_console()
        handle_slash_set("top_p 1.5", con)
        assert "less than or equal to 1" in buf.getvalue()

    def test_top_p_negative_rejected(self):
        con, buf = _make_console()
        handle_slash_set("top_p -0.1", con)
        assert "greater than or equal to 0" in buf.getvalue()

    def test_top_p_at_bounds_accepted(self):
        con, _buf = _make_console()
        handle_slash_set("top_p 0.0", con)
        assert cfg.top_p == 0.0
        handle_slash_set("top_p 1.0", con)
        assert cfg.top_p == 1.0

    def test_top_k_zero_rejected(self):
        con, buf = _make_console()
        handle_slash_set("top_k 0", con)
        assert "greater than or equal to 1" in buf.getvalue()

    def test_top_k_sampling_zero_rejected(self):
        con, buf = _make_console()
        handle_slash_set("top_k_sampling 0", con)
        assert "greater than or equal to 1" in buf.getvalue()

    def test_num_ctx_zero_rejected(self):
        con, buf = _make_console()
        handle_slash_set("num_ctx 0", con)
        assert "greater than or equal to 1" in buf.getvalue()

    def test_repeat_penalty_negative_rejected(self):
        con, buf = _make_console()
        handle_slash_set("repeat_penalty -1.0", con)
        assert "greater than or equal to 0" in buf.getvalue()

    def test_seed_any_value_accepted(self):
        con, buf = _make_console()
        handle_slash_set("seed -42", con)
        assert cfg.seed == -42
        assert "saved" in buf.getvalue()

    def test_valid_temperature_accepted(self):
        con, buf = _make_console()
        handle_slash_set("temperature 1.5", con)
        assert cfg.temperature == 1.5
        assert "saved" in buf.getvalue()


class TestSlashHelpIncludesNewCommands:
    def test_help_shows_settings(self):
        con, buf = _make_console()
        dispatch_slash("/help", con)
        output = buf.getvalue()
        assert "/settings" in output
        assert "/set" in output


class TestSetTabCompletion:
    def test_completes_setting_names(self):
        from lilbee.cli.chat import make_completer

        completer = make_completer()
        from prompt_toolkit.document import Document

        doc = Document("/set ", len("/set "))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert "temperature" in names
        assert "seed" in names
        assert "chat_model" in names

    def test_filters_by_prefix(self):
        from lilbee.cli.chat import make_completer

        completer = make_completer()
        from prompt_toolkit.document import Document

        doc = Document("/set temp", len("/set temp"))
        completions = list(completer.get_completions(doc, None))
        names = [c.text for c in completions]
        assert "temperature" in names
        assert "seed" not in names


class TestStreamResponseCancellation:
    def test_keyboard_interrupt_prints_stopped(self):
        con, buf = _make_console()
        history: list[dict] = []

        def interrupted_stream(*_args, **_kwargs):
            yield "partial "
            raise KeyboardInterrupt

        with mock.patch("lilbee.query.ask_stream", side_effect=interrupted_stream):
            from lilbee.cli.chat.stream import stream_response

            stream_response("test", history, con)

        assert "partial" in buf.getvalue()
        assert "(stopped)" in buf.getvalue()

    def test_partial_response_added_to_history(self):
        con, _buf = _make_console()
        history: list[dict] = []

        def interrupted_stream(*_args, **_kwargs):
            yield "partial "
            yield "answer"
            raise KeyboardInterrupt

        with mock.patch("lilbee.query.ask_stream", side_effect=interrupted_stream):
            from lilbee.cli.chat.stream import stream_response

            stream_response("test", history, con)

        assert len(history) == 2
        assert history[0]["content"] == "test"
        assert history[1]["content"] == "partial answer"

    def test_empty_partial_not_added_to_history(self):
        con, _buf = _make_console()
        history: list[dict] = []

        def interrupted_stream(*_args, **_kwargs):
            raise KeyboardInterrupt
            yield  # type: ignore[misc]

        with mock.patch("lilbee.query.ask_stream", side_effect=interrupted_stream):
            from lilbee.cli.chat.stream import stream_response

            stream_response("test", history, con)

        assert len(history) == 0

    def test_interrupt_during_thinking_spinner(self):
        con, buf = _make_console()
        history: list[dict] = []

        def interrupted_on_first(*_args, **_kwargs):
            raise KeyboardInterrupt
            yield  # type: ignore[misc]

        with mock.patch("lilbee.query.ask_stream", side_effect=interrupted_on_first):
            from lilbee.cli.chat.stream import stream_response

            stream_response("test", history, con)

        assert "(stopped)" in buf.getvalue()
        assert len(history) == 0

    def test_normal_response_still_works(self):
        con, _buf = _make_console()
        history: list[dict] = []

        with mock.patch("lilbee.query.ask_stream", return_value=iter(["Hello", " world"])):
            from lilbee.cli.chat.stream import stream_response

            stream_response("test", history, con)

        assert len(history) == 2
        assert history[1]["content"] == "Hello world"


class TestSyncToolbar:
    def test_returns_styled_text_when_active(self):
        from lilbee.cli.chat.loop import sync_toolbar
        from lilbee.cli.chat.sync import SyncStatus

        status = SyncStatus()
        status.text = "⟳ Syncing [1/3]: x.pdf"
        result = sync_toolbar(status)
        assert result == [("class:bottom-toolbar", "⟳ Syncing [1/3]: x.pdf")]

    def test_returns_empty_when_no_text(self):
        from lilbee.cli.chat.loop import sync_toolbar
        from lilbee.cli.chat.sync import SyncStatus

        status = SyncStatus()
        assert sync_toolbar(status) == ""

    def test_extract_event_updates_toolbar(self):
        from lilbee.cli.chat.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType

        status = SyncStatus()
        callback = _chat_sync_callback(status)
        callback(
            EventType.EXTRACT,
            {"file": "scan.pdf", "page": 2, "total_pages": 5},
        )
        assert status.text == "⟳ Vision OCR [2/5]: scan.pdf"

    def test_returns_empty_after_clear(self):
        from lilbee.cli.chat.loop import sync_toolbar
        from lilbee.cli.chat.sync import SyncStatus

        status = SyncStatus()
        status.text = "something"
        status.clear()
        assert sync_toolbar(status) == ""

    def test_pending_defaults_to_zero(self):
        from lilbee.cli.chat.sync import SyncStatus

        status = SyncStatus()
        assert status.pending == 0

    def test_toolbar_shows_queued_count(self):
        from lilbee.cli.chat.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType

        status = SyncStatus()
        status.pending = 2
        callback = _chat_sync_callback(status)
        callback(
            EventType.FILE_START,
            {"file": "a.pdf", "current_file": 1, "total_files": 1},
        )
        assert "(+2 queued)" in status.text

    def test_toolbar_no_queued_suffix_when_zero(self):
        from lilbee.cli.chat.sync import SyncStatus, _chat_sync_callback
        from lilbee.progress import EventType

        status = SyncStatus()
        status.pending = 0
        callback = _chat_sync_callback(status)
        callback(
            EventType.FILE_START,
            {"file": "a.pdf", "current_file": 1, "total_files": 1},
        )
        assert "queued" not in status.text
