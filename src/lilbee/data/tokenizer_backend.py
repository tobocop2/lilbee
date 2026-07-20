"""lilbee's embedder tokenizer exposed as a xberg plugin tokenizer backend.

xberg's chunk sizer can budget chunks in tokens, but only with tokenizers it loads
itself (a HuggingFace id or a local ``tokenizer.json``). lilbee's embedder is a
GGUF model served by llama-server, whose vocab is not published that way.
Registering this backend lets ChunkSizing count chunk budgets with the exact
tokenizer the embedder consumes, so ``chunk_size`` is a real token ceiling instead
of a chars-per-token guess (which mis-sizes token-dense text by 2-4x).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lilbee.data.ingest.types import TokenizerBackendName

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

# Conservative (over-counting) chars-per-token used only when the exact count is
# unavailable. Over-counting keeps the fallback on the safe side of the budget --
# it splits a touch early rather than emitting an over-length chunk. Mirrors the
# embed client's own estimate (providers.fleet.client._EMBED_EST_CHARS_PER_TOKEN).
_FALLBACK_CHARS_PER_TOKEN = 3


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate from character length (ceiling division)."""
    return max(1, -(-len(text) // _FALLBACK_CHARS_PER_TOKEN))


class LilbeeTokenizerBackend:
    """Counts chunk-sizing tokens with lilbee's embedder tokenizer.

    ``count_fn`` is the provider's exact token count (read live, so an
    embedding-model swap needs no re-registration). xberg calls ``count_tokens``
    synchronously inside the chunk sizer's boundary search -- many times per chunk,
    on its own worker threads -- and requires a non-zero count for non-empty text
    both at registration (a probe) and at runtime. So any failure of the exact
    count degrades to the character estimate instead of raising: the embedder may be
    unloaded when this runs (the registration probe, or chunking outside an ingest),
    and a raise would fail registration or abort extraction while a zero would make
    every span look like it fits the budget.
    """

    def __init__(self, *, count_fn: Callable[[str], int]) -> None:
        self._count_fn = count_fn

    def name(self) -> str:
        return TokenizerBackendName.LILBEE

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            count = self._count_fn(text)
        except Exception:  # embedder unreachable/erroring: degrade, never crash chunking
            log.debug("exact token count failed; using character estimate", exc_info=True)
            return _estimate_tokens(text)
        return count if count > 0 else _estimate_tokens(text)
