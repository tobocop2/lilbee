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

from pydantic import ValidationError

from lilbee.core.config.model import cfg

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lilbee.core.config.model import Config

_active: ContextVar[Config | None] = ContextVar("lilbee_active_config", default=None)


def active_config() -> Config:
    """Return the scoped Config if one is active, else the process-global ``cfg``."""
    return _active.get() or cfg


def validate_ocr_timeout(ocr_timeout: float | None) -> None:
    """Raise ``ValueError`` unless *ocr_timeout* satisfies the config field's own bound.

    ``None`` always passes (it means "keep the current setting"). Assigns into
    a throwaway copy of the active config, so the ``ocr_timeout`` field's own
    ``validate_assignment`` rule decides, the same rule the CLI already
    triggers through a direct ``cfg.ocr_timeout = value`` assignment.
    """
    if ocr_timeout is None:
        return
    scratch = active_config().model_copy()
    try:
        scratch.ocr_timeout = ocr_timeout
    except ValidationError as exc:
        raise ValueError(f"ocr_timeout: {exc.errors()[0]['msg']}") from exc


@contextmanager
def config_scope(config: Config) -> Iterator[None]:
    """Bind *config* as the active config for the duration of the block."""
    token = _active.set(config)
    try:
        yield
    finally:
        _active.reset(token)
