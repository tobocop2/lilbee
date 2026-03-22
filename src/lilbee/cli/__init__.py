"""CLI entry point for lilbee."""

# App and console must be imported before commands (which registers decorators on app).
from lilbee.cli.app import app, apply_overrides, console
from lilbee.cli.chat import (
    QuitChat,
    chat_loop,
    dispatch_slash,
    list_installed_models,
    make_completer,
)
from lilbee.cli.commands import CHUNK_PREVIEW_LEN as CHUNK_PREVIEW_LEN
from lilbee.cli.helpers import (
    CopyResult,
    auto_sync,
    clean_result,
    copy_files,
    copy_paths,
    get_version,
    json_output,
    perform_reset,
    sync_result_to_json,
)

__all__ = [
    "CHUNK_PREVIEW_LEN",
    "CopyResult",
    "QuitChat",
    "app",
    "apply_overrides",
    "auto_sync",
    "chat_loop",
    "clean_result",
    "console",
    "copy_files",
    "copy_paths",
    "dispatch_slash",
    "get_version",
    "json_output",
    "list_installed_models",
    "make_completer",
    "perform_reset",
    "sync_result_to_json",
]
