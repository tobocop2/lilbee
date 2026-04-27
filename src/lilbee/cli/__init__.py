"""CLI entry point for lilbee."""

# App and console must be imported before commands (which registers decorators on app).
from lilbee.cli.app import app, apply_overrides, console
from lilbee.cli.commands import CHUNK_PREVIEW_LEN as CHUNK_PREVIEW_LEN
from lilbee.cli.model import model_app

app.add_typer(model_app)

__all__ = [
    "CHUNK_PREVIEW_LEN",
    "app",
    "apply_overrides",
    "console",
]
