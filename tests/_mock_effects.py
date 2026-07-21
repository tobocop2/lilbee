"""side_effect sequences that tolerate more calls than scripted.

An exact-length ``side_effect=[...]`` list raises ``StopIteration`` the
moment the mocked call happens once more than the author assumed, which
presents as an unrelated failure. ``repeat_last`` keeps yielding the final
effect instead, so a surplus call trips the test's own assertions.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain, repeat
from typing import Any


def repeat_last(*effects: Any) -> Iterator[Any]:
    """Yield each effect in order, then the last one forever."""
    return chain(effects, repeat(effects[-1]))
