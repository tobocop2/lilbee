"""Engine lifecycle commands: manage a warm fleet without opening the TUI."""

from __future__ import annotations

import typer

from lilbee.cli.app import console, json_out
from lilbee.core.config import cfg
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.swap_manager import find_detached_state, reap_stale

engine_app = typer.Typer(
    name="engine",
    help="Manage the local inference engine.",
    no_args_is_help=True,
)

_STOPPED = "Stopped the warm engine and freed its memory."
_NOTHING_RUNNING = "No warm engine is running."


@engine_app.command("stop")
def stop() -> None:
    """Stop a warm engine left running by keep_engine_warm."""
    detached = [
        group.value for group in SwapGroup if find_detached_state(cfg.data_dir, group) is not None
    ]
    reap_stale(cfg.data_dir)
    if cfg.json_mode:
        json_out({"command": "engine stop", "stopped": detached})
        return
    console.print(_STOPPED if detached else _NOTHING_RUNNING)
