"""Local memory commands: add, list, recall, remove."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from lilbee.app.memory import (
    MEMORY_DISABLED_HINT,
    forget,
    list_memories,
    memory_enabled,
    recall,
    remember,
)
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg
from lilbee.data.store import MemoryKind

memory_app = typer.Typer(help="Manage your local long-term memory (off unless enabled).")

_ID_PREVIEW_CHARS = 8


def _disabled() -> None:
    """Report that memory is off, as JSON or a console line."""
    if cfg.json_mode:
        json_output({"error": MEMORY_DISABLED_HINT})
    else:
        console.print(MEMORY_DISABLED_HINT)


@memory_app.command(name="add")
def memory_add(
    text: str,
    preference: bool = typer.Option(
        False, "--preference", "-p", help="Store as an always-recalled preference."
    ),
    shared: bool = typer.Option(False, "--shared", help="Also expose this memory to agents."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Remember a fact (or preference) in your local memory."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not memory_enabled():
        _disabled()
        return
    kind = MemoryKind.PREFERENCE if preference else MemoryKind.FACT
    memory_id = remember(text, kind=kind, shared=shared)
    if cfg.json_mode:
        json_output({"id": memory_id, "kind": kind.value})
        return
    console.print(f"Remembered ({kind.value}).")


@memory_app.command(name="list")
def memory_list_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List your stored memories."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    # Listing memories is a pure store read; don't warm the inference fleet.
    cfg.worker_pool_eager_start = False
    if not memory_enabled():
        _disabled()
        return
    memories = list_memories()
    if cfg.json_mode:
        json_output(
            {
                "memories": [
                    {
                        "id": m.id,
                        "kind": m.kind.value,
                        "shared": m.shared,
                        "text": m.text,
                    }
                    for m in memories
                ]
            }
        )
        return
    if not memories:
        console.print("No memories stored.")
        return
    table = Table("id", "kind", "shared", "text")
    for m in memories:
        table.add_row(m.id[:_ID_PREVIEW_CHARS], m.kind.value, "yes" if m.shared else "no", m.text)
    console.print(table)


@memory_app.command(name="recall")
def memory_recall_cmd(
    query: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Recall facts relevant to a query."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not memory_enabled():
        _disabled()
        return
    memories = recall(query)
    if cfg.json_mode:
        json_output({"memories": [{"id": m.id, "text": m.text} for m in memories]})
        return
    if not memories:
        console.print("No relevant memories.")
        return
    for m in memories:
        console.print(f"- {m.text}")


@memory_app.command(name="remove")
def memory_remove(
    memory_id: str,
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete a memory by id."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not memory_enabled():
        _disabled()
        return
    deleted = forget(memory_id)
    if cfg.json_mode:
        json_output({"id": memory_id, "deleted": deleted})
        if not deleted:
            raise typer.Exit(1)
        return
    console.print(f"Removed {memory_id}." if deleted else f"No memory {memory_id} found.")
    # Exit non-zero on not-found, matching `model remove` / `remove`.
    if not deleted:
        raise typer.Exit(1)
