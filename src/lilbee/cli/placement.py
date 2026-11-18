"""`lilbee placement` sub-app: inspect, preview, and set GPU placement."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import typer
from rich.table import Table
from rich.text import Text

from lilbee.app.placement import (
    PLACEMENT_READ_ERRORS,
    SAVED_PLACEMENT_WAITS_FOR_A_MODEL,
    SAVED_PLACEMENT_WAITS_FOR_ENGINE,
    PlacementView,
    get_placement,
    preview_placement,
    set_placement,
)
from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.cli.helpers import json_output
from lilbee.core.config import cfg
from lilbee.providers.fleet.placement_spec import PlacementSpec
from lilbee.runtime.console import styled

placement_app = typer.Typer(
    name="placement",
    help="Inspect and override multi-GPU model placement.",
    no_args_is_help=True,
)

_GIB = 1024**3
_CLEAR_INSTRUCTION = "run 'lilbee placement clear' or set a new one"
_WAITING_REFUSALS = frozenset({SAVED_PLACEMENT_WAITS_FOR_ENGINE, SAVED_PLACEMENT_WAITS_FOR_A_MODEL})


def _read_spec(spec: str | None) -> PlacementSpec | None:
    """Parse a spec from a file path or stdin ('-'); return None when omitted."""
    if spec is None:
        return None
    if spec == "-":
        raw = sys.stdin.read()
    elif spec.lstrip().startswith("{"):
        raw = spec  # inline JSON rather than a file path
    else:
        raw = Path(spec).read_text(encoding="utf-8")
    return PlacementSpec.from_json(raw)


def _apply_placement_overrides(*, data_dir: Path | None, use_global: bool) -> None:
    """Apply CLI overrides for a placement command, with the fleet left cold.

    Planning reads the model registry through the services container, which would
    start a fleet warm-up in this short-lived process. That warm races the command
    and its exit, and reports a load failure after a placement that was applied.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    cfg.worker_pool_eager_start = False


def _guard(action: Callable[[], PlacementView]) -> None:
    """Run a placement action and render it, turning known failures into a clean exit."""
    try:
        view = action()
    except PLACEMENT_READ_ERRORS as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(Text(str(exc), style=theme.ERROR), soft_wrap=True)
        raise typer.Exit(code=1) from exc
    if cfg.json_mode:
        # The same canonical shape the HTTP and MCP surfaces return.
        from lilbee.server.models import PlacementResponse

        json_output(PlacementResponse.from_view(view).model_dump(mode="json"))
    else:
        _render_view(view)


def _render_view(view: PlacementView) -> None:
    """Print a Rich table of GPU rows plus per-role and unplaceable lines."""
    title = "Placement (manual)" if view.manual else "Placement (auto)"
    table = Table(title=title)
    table.add_column("GPU")
    table.add_column("Name")
    table.add_column("Free / Total")
    table.add_column("Roles")

    placed: dict[int, list[str]] = {g.index: [] for g in view.gpus}
    for role_view in view.roles:
        for idx in role_view.devices:
            placed.setdefault(idx, []).append(role_view.role.value)

    for g in view.gpus:
        free_gib = g.free_bytes / _GIB
        total_gib = g.total_bytes / _GIB
        table.add_row(
            g.label,
            g.name or "(unnamed)",
            f"{free_gib:.0f} / {total_gib:.0f} GiB",
            Text(", ".join(placed.get(g.index, [])) or "-"),
        )
    console.print(table)

    for role_view in view.roles:
        split_info = f" split={list(role_view.tensor_split)}" if role_view.tensor_split else ""
        console.print(
            Text(
                f"  {role_view.role.value}: target={role_view.target.value}"
                f" devices={list(role_view.devices)}"
                f" replicas={role_view.replicas}{split_info}"
                f" charge={role_view.charge_bytes / _GIB:.1f} GiB  {role_view.model}"
            ),
            soft_wrap=True,
        )

    if view.co_tenants:
        names = ", ".join(role.value for role in view.co_tenants)
        console.print(f"  {names}: share memory, one loaded at a time", style=theme.MUTED)

    for role in view.unplaceable:
        console.print(f"  {role.value}: does not fit, no server", style=theme.ERROR)

    for skipped in view.skipped_not_installed:
        console.print(
            Text.assemble(
                (f"  {skipped.role.value}: ", theme.WARNING),
                skipped.model,
                (" not downloaded, pull it to place it", theme.WARNING),
            ),
            soft_wrap=True,
        )

    _render_model_notices(view)

    if view.hidden_note:
        console.print(Text(f"  {view.hidden_note}"), style=theme.MUTED, soft_wrap=True)

    _render_rejected(view)


def _render_model_notices(view: PlacementView) -> None:
    """Print a line for each role left out for want of an engine, and each of unknown size."""
    for no_engine in view.skipped_no_engine:
        console.print(
            Text.assemble(
                (f"  {no_engine.role.value}: ", theme.WARNING),
                (no_engine.notice, theme.WARNING),
            ),
            soft_wrap=True,
        )

    for unknown in view.size_unknown:
        console.print(Text(f"  {unknown.notice}", style=theme.WARNING), soft_wrap=True)

    for tight in view.tight:
        console.print(Text(f"  {tight.notice}", style=theme.WARNING), soft_wrap=True)


def _render_rejected(view: PlacementView) -> None:
    """Print why a saved placement is not in use, when it is not."""
    if view.refusal:
        console.print(Text(f"  {view.refusal}", style=theme.ERROR), soft_wrap=True)
        if view.refusal not in _WAITING_REFUSALS:
            console.print(Text(f"  {_CLEAR_INSTRUCTION}", style=theme.WARNING), soft_wrap=True)
    elif view.rejected_spec_json:
        console.print(
            styled(
                "  ",
                (
                    "a saved placement does not fit this hardware and is being ignored; "
                    f"{_CLEAR_INSTRUCTION}",
                    theme.WARNING,
                ),
            )
        )


@placement_app.command("show")
def show(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show the current effective placement."""
    _apply_placement_overrides(data_dir=data_dir, use_global=use_global)
    _guard(get_placement)


@placement_app.command("preview")
def preview(
    spec: str | None = typer.Option(
        None, "--spec", help="Spec JSON file, or - for stdin; omit for auto."
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Preview what a spec (or auto) would place, without applying it."""
    _apply_placement_overrides(data_dir=data_dir, use_global=use_global)
    _guard(lambda: preview_placement(_read_spec(spec)))


@placement_app.command("set")
def set_cmd(
    spec: str = typer.Option(..., "--spec", help="Spec JSON file, or - for stdin."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Validate, persist, and apply a manual placement spec."""
    _apply_placement_overrides(data_dir=data_dir, use_global=use_global)
    _guard(lambda: set_placement(_read_spec(spec)))


@placement_app.command("clear")
def clear(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Clear the manual placement and return to automatic placement."""
    _apply_placement_overrides(data_dir=data_dir, use_global=use_global)
    _guard(lambda: set_placement(None))
