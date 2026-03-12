"""App creation, console, and global callback."""

from pathlib import Path

import typer
from rich.console import Console

from lilbee.cli.helpers import auto_sync, get_version
from lilbee.cli.helpers import json_output as json_out
from lilbee.config import cfg
from lilbee.models import ensure_tag

app = typer.Typer(help="lilbee — Local RAG knowledge base", invoke_without_command=True)
console = Console()

data_dir_option = typer.Option(
    None,
    "--data-dir",
    "-d",
    help="Override data directory (default: platform-specific, see 'lilbee status')",
)

model_option = typer.Option(
    None,
    "--model",
    "-m",
    help="Override chat model (default: $LILBEE_CHAT_MODEL or 'qwen3:8b')",
)

_json_option = typer.Option(
    False,
    "--json",
    "-j",
    help="Emit structured JSON output (for agent/script consumption).",
)


def apply_overrides(
    data_dir: Path | None = None,
    model: str | None = None,
) -> None:
    """Apply CLI overrides to config before any work begins."""
    if data_dir is not None:
        cfg.documents_dir = data_dir / "documents"
        cfg.data_dir = data_dir / "data"
        cfg.lancedb_dir = data_dir / "data" / "lancedb"

    if model is not None:
        cfg.chat_model = ensure_tag(model)


@app.callback()
def _default(
    ctx: typer.Context,
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    json_output: bool = _json_option,
    show_version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    """Start interactive chat when no command is given."""
    if show_version:
        typer.echo(f"lilbee {get_version()}")
        raise SystemExit(0)

    cfg.json_mode = json_output
    if ctx.invoked_subcommand is None:
        apply_overrides(data_dir=data_dir, model=model)
        if cfg.json_mode:
            json_out({"error": "Interactive chat requires a terminal, not --json"})
            raise SystemExit(1)
        from lilbee.embedder import validate_model
        from lilbee.models import ensure_chat_model

        ensure_chat_model()
        validate_model()
        auto_sync(console)
        from lilbee.cli.chat import chat_loop

        chat_loop(console)
