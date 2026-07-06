"""Targeted sys.modules injection for tests.

``mock.patch.dict("sys.modules", ...)`` restores by clearing the whole module
registry and re-filling it from a snapshot; any import that lands in that
window (including from a background thread) leaves split-brain module state
that can poison every later test in the process. This helper touches only the
injected keys.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

_MISSING = object()


@contextmanager
def inject_modules(mapping: Mapping[str, object]) -> Iterator[None]:
    """Temporarily set entries in ``sys.modules``, restoring only those keys.

    A ``None`` value makes ``import <name>`` raise ImportError, mirroring the
    ``patch.dict`` idiom for simulating an uninstalled package.
    """
    saved = {name: sys.modules.get(name, _MISSING) for name in mapping}
    sys.modules.update(mapping)  # type: ignore[arg-type]
    try:
        yield
    finally:
        for name, old in saved.items():
            if old is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old  # type: ignore[assignment]
