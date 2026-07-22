"""Engine lifecycle commands: manage the shared engine without opening the TUI."""

from __future__ import annotations

import typer

from lilbee.cli.app import console, json_out
from lilbee.core.config import cfg
from lilbee.providers.fleet.swap_manager import stop_engine
from lilbee.runtime.engine_lock import build_lock, machine_engine_dir, private_engine_dir

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
    # Machine slot, this root's overflow dir, and the pre-upgrade legacy location
    # (base builds recorded state directly under data_dir): a stale legacy engine
    # must still be stoppable after an upgrade relocated the engine dirs.
    dirs = dict.fromkeys((machine_engine_dir(), private_engine_dir(cfg.data_root), cfg.data_dir))
    for engine_dir in dirs:
        # Serialize against a concurrent builder, whose spawn + state write + health
        # wait all run under this lock; an unlocked stop landing mid-build would kill
        # the just-spawned engine and unlink its fresh record. best_effort so a wedged
        # builder cannot make the off switch hang.
        with build_lock(engine_dir, best_effort=True):
            stopped.extend(stop_engine(engine_dir))
    if cfg.json_mode:
        json_out({"command": "engine stop", "stopped": sorted(set(stopped))})
        return
    console.print(_STOPPED if stopped else _NOTHING_RUNNING)
