"""What a fan-out worker process was told about its own slice of the machine.

A worker pins its card, its engine slot and its CPU share. These values live
here, in plain Python state, and not in ``os.environ``: ``setenv`` reallocates
the C ``environ`` array, and a native thread that reads it at that moment
(pyarrow, lancedb) can fault. A worker has native threads as soon as its
imports finish, so no variable can be written in it safely.

The readers take the value from here. An engine child gets the card pin in its
own ``env=`` at spawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ShardProcessState:
    """The slice values of this process; every field is unset outside a worker."""

    engine_dir: Path | None = None
    cpu_quota: int | None = None
    visible_devices: dict[str, str] = field(default_factory=dict)

    def bind(self, engine_dir: Path, cpu_quota: int, visible_devices: dict[str, str]) -> None:
        """Record the slice. Called once, first, by the worker's main thread."""
        self.engine_dir = engine_dir
        self.cpu_quota = cpu_quota
        self.visible_devices = dict(visible_devices)


shard_process = ShardProcessState()
