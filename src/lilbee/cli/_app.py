"""App creation, console, and global callback."""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="lilbee — Local RAG knowledge base", invoke_without_command=True)
console = Console()

_data_dir_option = typer.Option(
    None,
    "--data-dir",
    "-d",
    help="Override data directory (default: platform-specific, see 'lilbee status')",
)

_model_option = typer.Option(
    None,
    "--model",
    "-m",
    help="Override chat model (default: $LILBEE_CHAT_MODEL or 'qwen3-coder:30b')",
)

_json_option = typer.Option(
    False,
    "--json",
    "-j",
    help="Emit structured JSON output (for agent/script consumption).",
)


def _apply_overrides(
    data_dir: Path | None = None,
    model: str | None = None,
) -> None:
    """Apply CLI overrides to config before any work begins."""
    import lilbee.config as cfg
    import lilbee.store as store_mod

    if data_dir is not None:
        cfg.DOCUMENTS_DIR = data_dir / "documents"
        cfg.DATA_DIR = data_dir / "data"
        cfg.LANCEDB_DIR = data_dir / "data" / "lancedb"
        store_mod.LANCEDB_DIR = cfg.LANCEDB_DIR

    if model is not None:
        cfg.CHAT_MODEL = model


@app.callback()
def _default(
    ctx: typer.Context,
    data_dir: Path | None = _data_dir_option,
    model: str | None = _model_option,
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
    from lilbee.cli._helpers import _auto_sync, _get_version
    from lilbee.cli._helpers import _json_output as json_out

    if show_version:
        typer.echo(f"lilbee {_get_version()}")
        raise SystemExit(0)
    from lilbee.cli import _state

    _state["json_mode"] = json_output
    if ctx.invoked_subcommand is None:
        _apply_overrides(data_dir=data_dir, model=model)
        if _state["json_mode"]:
            json_out({"error": "Interactive chat requires a terminal, not --json"})
            raise SystemExit(1)
        from lilbee.embedder import validate_model
        from lilbee.models import ensure_chat_model

        ensure_chat_model()
        validate_model()
        _auto_sync(console)
        from lilbee.cli._chat import _chat_loop

        _chat_loop(console)
