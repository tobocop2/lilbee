"""App creation, console, and global callback."""

import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from lilbee.cli.helpers import get_version
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

_global_option = typer.Option(
    False,
    "--global",
    "-g",
    help="Use the global database, ignoring any local .lilbee/ directory.",
)

_log_level_option = typer.Option(
    None,
    "--log-level",
    help="Set log level (DEBUG, INFO, WARNING, ERROR). Overrides LILBEE_LOG_LEVEL.",
)

temperature_option = typer.Option(None, "--temperature", "-t", help="Sampling temperature.")
top_p_option = typer.Option(None, "--top-p", help="Top-p (nucleus) sampling threshold.")
top_k_sampling_option = typer.Option(None, "--top-k-sampling", help="Top-k sampling count.")
repeat_penalty_option = typer.Option(None, "--repeat-penalty", help="Repeat penalty factor.")
num_ctx_option = typer.Option(None, "--num-ctx", help="Context window size (tokens).")
seed_option = typer.Option(None, "--seed", help="Random seed for reproducibility.")


def _apply_data_root(root: Path) -> None:
    """Point all cfg data paths at *root*."""
    cfg.data_root = root
    cfg.documents_dir = root / "documents"
    cfg.data_dir = root / "data"
    cfg.lancedb_dir = root / "data" / "lancedb"


def apply_overrides(
    data_dir: Path | None = None,
    model: str | None = None,
    use_global: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k_sampling: int | None = None,
    repeat_penalty: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
) -> None:
    """Apply CLI overrides to config before any work begins.

    Precedence (highest first):
    --data-dir / LILBEE_DATA  >  .lilbee/ (local walk-up)  >  global platform default
    """
    if data_dir is not None and use_global:
        raise typer.BadParameter("Cannot use --global with --data-dir")

    if use_global:
        from lilbee.platform import default_data_dir

        _apply_data_root(default_data_dir())
    elif data_dir is not None:
        _apply_data_root(data_dir)
    else:
        data_env = os.environ.get("LILBEE_DATA", "")
        if data_env:
            _apply_data_root(Path(data_env))

    if model is not None:
        cfg.chat_model = ensure_tag(model)
    if temperature is not None:
        cfg.temperature = temperature
    if top_p is not None:
        cfg.top_p = top_p
    if top_k_sampling is not None:
        cfg.top_k_sampling = top_k_sampling
    if repeat_penalty is not None:
        cfg.repeat_penalty = repeat_penalty
    if num_ctx is not None:
        cfg.num_ctx = num_ctx
    if seed is not None:
        cfg.seed = seed


@app.callback()
def _default(
    ctx: typer.Context,
    data_dir: Path | None = data_dir_option,
    model: str | None = model_option,
    json_output: bool = _json_option,
    use_global: bool = _global_option,
    log_level: str | None = _log_level_option,
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

    level_str = os.environ.get("LILBEE_LOG_LEVEL", "WARNING").upper()
    if log_level is not None:
        level_str = log_level.upper()
    _LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    level = _LOG_LEVELS.get(level_str, logging.WARNING)
    logging.basicConfig(
        level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    # basicConfig is a no-op when handlers already exist, so always set level explicitly
    logging.getLogger().setLevel(level)

    cfg.json_mode = json_output
    if ctx.invoked_subcommand is None:
        apply_overrides(data_dir=data_dir, model=model, use_global=use_global)
        if cfg.json_mode:
            json_out({"error": "Interactive chat requires a terminal, not --json"})
            raise SystemExit(1)
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            typer.echo("Error: Interactive chat requires a terminal.", err=True)
            raise SystemExit(1)
        from lilbee.cli.tui import run_tui

        run_tui(auto_sync=True)
