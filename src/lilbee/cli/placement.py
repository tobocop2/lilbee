"""`lilbee placement` sub-app: inspect, preview, and set GPU placement."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.table import Table

from lilbee.app.placement import (
    PlacementView,
    get_placement,
    preview_placement,
    set_placement,
)
from lilbee.cli import theme
from lilbee.cli.app import apply_overrides, console, data_dir_option, global_option
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec

placement_app = typer.Typer(
    name="placement",
    help="Inspect and override multi-GPU model placement.",
    no_args_is_help=True,
)

_GIB = 1024**3


def _read_spec(spec: str | None) -> PlacementSpec | None:
    """Parse a spec from a file path or stdin ('-'); return None when omitted."""
    if spec is None:
        return None
    raw = sys.stdin.read() if spec == "-" else Path(spec).read_text()
    return PlacementSpec.from_json(raw)


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
            ", ".join(placed.get(g.index, [])) or "-",
        )
    console.print(table)

    for role_view in view.roles:
        split_info = f" split={list(role_view.tensor_split)}" if role_view.tensor_split else ""
        console.print(
            f"  {role_view.role.value}: devices={list(role_view.devices)}"
            f" replicas={role_view.replicas}{split_info}  {role_view.model}"
        )

    for role in view.unplaceable:
        console.print(f"  [{theme.ERROR}]{role.value}: does not fit, no server[/{theme.ERROR}]")


@placement_app.command("show")
def show(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show the current effective placement."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _render_view(get_placement())


@placement_app.command("preview")
def preview(
    spec: str | None = typer.Option(
        None, "--spec", help="Spec JSON file, or - for stdin; omit for auto."
    ),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Preview what a spec (or auto) would place, without applying it."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    try:
        _render_view(preview_placement(_read_spec(spec)))
    except PlacementError as exc:
        console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(code=1) from exc


@placement_app.command("set")
def set_cmd(
    spec: str = typer.Option(..., "--spec", help="Spec JSON file, or - for stdin."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Validate, persist, and apply a manual placement spec."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    try:
        _render_view(set_placement(_read_spec(spec)))
    except PlacementError as exc:
        console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(code=1) from exc


@placement_app.command("clear")
def clear(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Clear the manual placement and return to automatic placement."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _render_view(set_placement(None))
