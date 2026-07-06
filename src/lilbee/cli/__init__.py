"""CLI entry point for lilbee."""

# Importing `commands` registers each command on the shared Typer ``app``;
# its own first import is `from lilbee.cli.app import app`, so loading
# happens in the right order even though `app` is listed below alphabetically.
from lilbee.cli import commands as _commands  # noqa: F401  side-effect: command registration
from lilbee.cli.app import app, apply_overrides, console
from lilbee.cli.model import model_app
from lilbee.cli.placement import placement_app

app.add_typer(model_app)
app.add_typer(placement_app)

__all__ = [
    "app",
    "apply_overrides",
    "console",
]
