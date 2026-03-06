"""CLI entry point for lilbee."""

# Mutable state shared across submodules — _commands.py reads, _app.py writes.
_state: dict[str, bool] = {"json_mode": False}

# App and console must be imported before _commands (which registers decorators on app).
from lilbee.cli._app import _apply_overrides, app, console  # noqa: E402
from lilbee.cli._chat import (  # noqa: E402
    _chat_loop,
    _dispatch_slash,
    _handle_slash_model,
    _handle_slash_quit,
    _handle_slash_reset,
    _handle_slash_version,
    _list_ollama_models,
    _make_completer,
    _QuitChat,
)
from lilbee.cli._commands import _CHUNK_PREVIEW_LEN as _CHUNK_PREVIEW_LEN  # noqa: E402
from lilbee.cli._helpers import (  # noqa: E402
    _auto_sync,
    _clean_result,
    _copy_paths,
    _get_version,
    _json_output,
    _perform_reset,
    _sync_result_to_json,
)

__all__ = [
    "_QuitChat",
    "_apply_overrides",
    "_auto_sync",
    "_chat_loop",
    "_clean_result",
    "_copy_paths",
    "_dispatch_slash",
    "_get_version",
    "_handle_slash_model",
    "_handle_slash_quit",
    "_handle_slash_reset",
    "_handle_slash_version",
    "_json_output",
    "_list_ollama_models",
    "_make_completer",
    "_perform_reset",
    "_state",
    "_sync_result_to_json",
    "app",
    "console",
]
