"""CLI entry point for lilbee."""

# App and console must be imported before commands (which registers decorators on app).
from lilbee.cli.app import app, apply_overrides, console
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
from lilbee.cli.model import model_app

app.add_typer(model_app)

__all__ = [
    "CHUNK_PREVIEW_LEN",
    "CopyResult",
    "app",
    "apply_overrides",
    "auto_sync",
    "clean_result",
    "console",
    "copy_files",
    "copy_paths",
    "get_version",
    "json_output",
    "perform_reset",
    "sync_result_to_json",
]
