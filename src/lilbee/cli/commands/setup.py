"""Token (server auth), HuggingFace login, and crawler-setup commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.crawler import CrawlerBrowserError, bootstrap_chromium, chromium_installed
from lilbee.runtime.progress import EventType, SetupProgressEvent


def token(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print the auth token for a running server."""
    from lilbee.server.auth import server_json_path

    apply_overrides(data_dir=data_dir, use_global=use_global)
    path = server_json_path()
    if not path.exists():
        if cfg.json_mode:
            json_output({"error": "No running server found"})
        else:
            console.print("No running server found (server.json missing).")
        raise SystemExit(1)
    try:
        data = json.loads(path.read_text())
        tok = data.get("token", "")
    except (json.JSONDecodeError, OSError) as exc:
        if cfg.json_mode:
            json_output({"error": f"Could not read server.json: {exc}"})
        else:
            console.print(
                f"[{theme.ERROR}]Error:[/{theme.ERROR}] Could not read server.json: {exc}"
            )
        raise SystemExit(1) from None
    if cfg.json_mode:
        json_output({"token": tok})
        return
    console.print(tok)


def login() -> None:
    """Log in to HuggingFace for access to gated models (Mistral, Llama, etc.)."""
    import webbrowser

    from huggingface_hub import get_token
    from huggingface_hub import login as hf_login

    if get_token():
        typer.echo("Already logged in to HuggingFace.")
        if not typer.confirm("Log in again?", default=False):
            return

    typer.echo("Opening HuggingFace token page in your browser...")
    typer.echo("Create a token with 'Read' access, then paste it below.\n")
    webbrowser.open("https://huggingface.co/settings/tokens")

    token = typer.prompt("Paste your HuggingFace token", hide_input=True)
    if not token.strip():
        typer.echo("No token provided.", err=True)
        raise typer.Exit(1)

    hf_login(token=token.strip(), add_to_git_credential=False)
    typer.echo("Logged in! Gated models (Mistral, Llama, etc.) are now accessible.")


setup_app = typer.Typer(help="One-time setup for optional runtime components.")


@setup_app.command(name="crawler")
def setup_crawler_cmd() -> None:
    """Install Playwright's Chromium browser, needed for /crawl.

    No-op when Chromium is already present. Emits a simple progress
    readout; use '--json' mode on the top-level 'lilbee' command to get
    a single JSON blob with the final install state instead.
    """
    if chromium_installed():
        if cfg.json_mode:
            typer.echo(json.dumps({"component": "chromium", "already_installed": True}))
        else:
            typer.echo("Chromium already installed.")
        return

    last_pct: list[int] = [-1]

    def _on_progress(event_type: object, data: object) -> None:
        if event_type != EventType.SETUP_PROGRESS or not isinstance(data, SetupProgressEvent):
            return
        total = data.total_bytes or 0
        pct = int(data.downloaded_bytes * 100 / total) if total > 0 else 0
        if pct != last_pct[0] and not cfg.json_mode:
            last_pct[0] = pct
            typer.echo(msg.SETUP_CHROMIUM_CLI_PROGRESS.format(pct=pct), err=True)

    try:
        asyncio.run(bootstrap_chromium(on_progress=_on_progress))
    except CrawlerBrowserError as exc:
        if cfg.json_mode:
            typer.echo(json.dumps({"component": "chromium", "error": str(exc)}))
        else:
            typer.secho(f"Install failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    if cfg.json_mode:
        typer.echo(json.dumps({"component": "chromium", "installed": True}))
    else:
        typer.echo("Chromium installed.")
