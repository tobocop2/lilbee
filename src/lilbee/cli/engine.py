"""Engine lifecycle commands: manage the shared engine without opening the TUI."""

from __future__ import annotations

import typer

from lilbee.cli.app import console, json_out
from lilbee.core.config import cfg
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.swap_manager import find_live_state, stop_engine
from lilbee.runtime.engine_lock import machine_engine_dir, private_engine_dir

engine_app = typer.Typer(
    name="engine",
    help="Manage the local inference engine.",
    no_args_is_help=True,
)

_STOPPED = "Stopped the engine and freed its memory."
_NOTHING_RUNNING = "No engine is running."


@engine_app.command("stop")
def stop() -> None:
    """Stop the shared engine now, whoever started it."""
    stopped: list[str] = []
    for engine_dir in (machine_engine_dir(), private_engine_dir(cfg.data_root)):
        for group in SwapGroup:
            if find_live_state(engine_dir, group) is not None:
                stopped.append(group.value)
        stop_engine(engine_dir)
    if cfg.json_mode:
        json_out({"command": "engine stop", "stopped": sorted(set(stopped))})
        return
    console.print(_STOPPED if stopped else _NOTHING_RUNNING)
