"""One warning per ingest or search for the embed inputs a client or server cuts."""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

log = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass
class _Tally:
    cut: int = 0
    unwarned: int = 0
    total: int = 0
    limit: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_active: ContextVar[_Tally] = ContextVar("lilbee_embed_truncation")


def _warn(tally: _Tally) -> None:
    limit = "" if tally.limit is None else f"{tally.limit}-token "
    log.warning(
        "%d of %d embed inputs are past the embedder's %slimit and the cut text is not in "
        "the vectors. Lower chunk_size, or use an embedding model with a larger input limit.",
        tally.cut,
        tally.total,
        limit,
    )


def note_truncated(cut: int, total: int, limit: int | None) -> None:
    """Record *cut* of *total* embed inputs past *limit*; log at once outside a scope."""
    tally = _active.get(None)
    if tally is None:
        if cut:
            _warn(_Tally(cut=cut, total=total, limit=limit))
        return
    with tally.lock:
        tally.cut += cut
        tally.total += total
        if tally.limit is None:
            tally.limit = limit


def note_cut_to_limit(size: str, limit: int) -> None:
    """Record one input of *size* tokens cut to *limit*: a run counts it, else log it."""
    tally = _active.get(None)
    if tally is None:
        log.warning("Truncating oversize embed input: %s tokens > cap %d", size, limit)
        return
    log.debug("Truncating oversize embed input: %s tokens > cap %d", size, limit)
    with tally.lock:
        tally.cut += 1
        if tally.limit is None:
            tally.limit = limit


def note_char_cut() -> None:
    """Count one input cut by a character limit in the current scope, with no warning."""
    tally = _active.get(None)
    if tally is not None:
        with tally.lock:
            tally.unwarned += 1


def cut_inputs_in_scope() -> int:
    """Embed inputs cut so far in the current scope; a ``LookupError`` outside one."""
    tally = _active.get()
    with tally.lock:
        return tally.cut + tally.unwarned


@contextmanager
def truncation_scope() -> Iterator[None]:
    """Collect the embed truncation counts of one run and log one warning at its end."""
    tally = _Tally()
    token = _active.set(tally)
    try:
        yield
    finally:
        _active.reset(token)
        if tally.cut:
            _warn(tally)


def one_truncation_warning(
    fn: Callable[_P, Coroutine[Any, Any, _R]],
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    """Run the async *fn* inside a truncation scope."""

    @functools.wraps(fn)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with truncation_scope():
            return await fn(*args, **kwargs)

    return wrapper
