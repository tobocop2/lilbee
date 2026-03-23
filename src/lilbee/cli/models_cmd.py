"""CLI subcommands for model management — browse, install, remove, list."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.table import Table

from lilbee.cli.app import (
    _global_option,
    apply_overrides,
    console,
    data_dir_option,
)
from lilbee.cli.helpers import json_output
from lilbee.cli.theme import ACCENT, ERROR, LABEL, MUTED
from lilbee.config import cfg

models_app = typer.Typer(help="Browse, install, and remove models.")


@models_app.command(name="list")
def list_cmd(
    source: str | None = typer.Option(None, help="Filter by source: native, ollama"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """List installed models."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.model_manager import ModelSource, get_model_manager

    manager = get_model_manager()
    src = ModelSource(source) if source else None
    names = manager.list_installed(src)

    if cfg.json_mode:
        items = []
        for name in names:
            s = manager.get_source(name)
            items.append({"name": name, "source": s.value if s else "unknown"})
        json_output({"command": "models list", "models": items})
        return

    if not names:
        console.print(f"[{MUTED}]No models installed.[/{MUTED}]")
        return

    table = Table(title="Installed Models")
    table.add_column("Model", style=ACCENT)
    table.add_column("Source", style=MUTED)
    for name in names:
        s = manager.get_source(name)
        table.add_row(name, s.value if s else "unknown")
    console.print(table)


@models_app.command()
def browse(
    task: str | None = typer.Option(None, help="Filter: chat, embedding, vision"),
    search: str = typer.Option("", "--search", "-s", help="Search query"),
    size: str | None = typer.Option(None, help="Size filter: small, medium, large"),
    featured: bool = typer.Option(False, "--featured", "-f", help="Show featured only"),
    limit: int = typer.Option(20, "--limit", "-n", help="Results per page"),
    offset: int = typer.Option(0, "--offset", help="Skip N results"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Browse the model catalog (featured + HuggingFace)."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    if not cfg.json_mode and sys.stdin.isatty() and sys.stdout.isatty():
        _browse_interactive(task=task, search=search)
        return

    _browse_static(
        task=task,
        search=search,
        size=size,
        featured=featured,
        limit=limit,
        offset=offset,
    )


def _browse_interactive(task: str | None, search: str) -> None:
    """Launch the Textual TUI browser and optionally install the selected model."""
    from lilbee.cli.browser import run_browser

    selected = run_browser(task=task, search=search)
    if selected is None:
        return
    _install_from_catalog(selected.name)


def _install_from_catalog(name: str) -> None:
    """Install a model by catalog name, with progress display."""
    from lilbee.catalog import find_catalog_entry
    from lilbee.model_manager import ModelSource, get_model_manager

    manager = get_model_manager()
    if manager.is_installed(name):
        if cfg.json_mode:
            json_output({"command": "models install", "model": name, "already_installed": True})
        else:
            console.print(f"[{LABEL}]{name}[/{LABEL}] is already installed.")
        return

    entry = find_catalog_entry(name)
    if entry is not None:
        from lilbee.models import pull_with_progress

        pull_with_progress(name)
        if cfg.json_mode:
            json_output({"command": "models install", "model": name, "source": "native"})
        return

    try:
        manager.pull(name, ModelSource.OLLAMA)
        if cfg.json_mode:
            json_output({"command": "models install", "model": name, "source": "ollama"})
        else:
            console.print(f"[{LABEL}]{name}[/{LABEL}] installed via Ollama.")
    except RuntimeError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{ERROR}]Error:[/{ERROR}] {exc}")
        raise SystemExit(1) from None


def _browse_static(
    task: str | None,
    search: str,
    size: str | None,
    featured: bool,
    limit: int,
    offset: int,
) -> None:
    """Show a static Rich table of catalog models (for --json or non-TTY)."""
    from lilbee.catalog import get_catalog

    result = get_catalog(
        task=task,
        search=search,
        size=size,
        featured=featured or None,
        sort="featured",
        limit=limit,
        offset=offset,
    )

    if cfg.json_mode:
        json_output(
            {
                "command": "models browse",
                "total": result.total,
                "limit": result.limit,
                "offset": result.offset,
                "models": [
                    {
                        "name": m.name,
                        "repo": m.hf_repo,
                        "size_gb": m.size_gb,
                        "task": m.task,
                        "description": m.description,
                        "featured": m.featured,
                    }
                    for m in result.models
                ],
            }
        )
        return

    if not result.models:
        console.print(f"[{MUTED}]No models found.[/{MUTED}]")
        return

    title = "Model Catalog"
    if task:
        title = f"{task.title()} Models"

    table = Table(title=title)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Model", style=ACCENT)
    table.add_column("Task", style=MUTED)
    table.add_column("Size", justify="right")
    table.add_column("Description")

    for idx, m in enumerate(result.models, offset + 1):
        star = " \u2605" if m.featured else ""
        table.add_row(
            str(idx),
            f"{m.name}{star}",
            m.task,
            f"{m.size_gb:.1f} GB",
            m.description[:80],
        )

    console.print(table)
    if result.total > offset + limit:
        console.print(
            f"\n[{MUTED}]Showing {offset + 1}-{offset + len(result.models)} "
            f"of {result.total}. Use --offset to see more.[/{MUTED}]"
        )


@models_app.command()
def install(
    name: str = typer.Argument(..., help="Model name (catalog name or Ollama name)"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Install a model by catalog name or Ollama name."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _install_from_catalog(name)


@models_app.command()
def remove(
    name: str = typer.Argument(..., help="Model name to remove"),
    source: str | None = typer.Option(None, help="Source: native, ollama"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = _global_option,
) -> None:
    """Remove an installed model."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.model_manager import ModelSource, get_model_manager

    manager = get_model_manager()
    src = ModelSource(source) if source else None
    deleted = manager.remove(name, src)

    if cfg.json_mode:
        json_output({"command": "models remove", "model": name, "deleted": deleted})
        return

    if deleted:
        console.print(f"Removed [{ACCENT}]{name}[/{ACCENT}]")
    else:
        console.print(f"[{ERROR}]Not found:[/{ERROR}] {name}")
        raise SystemExit(1)
