"""Per-scope active config: how the library API runs against its own Config.

The process-global ``cfg`` singleton backs the CLI, TUI, and HTTP daemon. The
library API (:class:`lilbee.Lilbee`) instead binds a caller-supplied Config for
the duration of each public method via :func:`config_scope`, and the ingest path
reads it through :func:`active_config` instead of the global. Scoping is a
ContextVar, so it stays isolated to the entering task and propagates into the
``to_ingest_thread`` workers (they copy the calling context), without mutating
the process-global cfg that other clients share.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from lilbee.core.config.model import cfg

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lilbee.core.config.model import Config

_active: ContextVar[Config | None] = ContextVar("lilbee_active_config", default=None)


def active_config() -> Config:
    """Return the scoped Config if one is active, else the process-global ``cfg``."""
    return _active.get() or cfg


@contextmanager
def config_scope(config: Config) -> Iterator[None]:
    """Bind *config* as the active config for the duration of the block."""
    token = _active.set(config)
    try:
        yield
    finally:
        _active.reset(token)
