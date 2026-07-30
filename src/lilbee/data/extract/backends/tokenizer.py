"""lilbee's embedder tokenizer exposed as a xberg plugin tokenizer backend.

xberg's chunk sizer can budget in tokens, but only with tokenizers it loads itself;
lilbee's embedder is a GGUF model whose vocab isn't published that way. This lets
``chunk_size`` be a real token ceiling instead of a chars-per-token guess (off by
2-4x on token-dense text).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lilbee.data.types import TokenizerBackendName

from .registry import BackendKind, XbergBinding, register_binding

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

# Over-counting fallback for when the exact count is unavailable: splits a touch
# early rather than emitting an over-length chunk. Mirrors providers.fleet.client.
_FALLBACK_CHARS_PER_TOKEN = 3


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate from character length (ceiling division)."""
    return max(1, -(-len(text) // _FALLBACK_CHARS_PER_TOKEN))


class LilbeeTokenizerBackend:
    """Counts chunk-sizing tokens with lilbee's embedder tokenizer.

    ``count_fn`` is read live (an embedding-model swap needs no re-registration).
    xberg requires a non-zero count for non-empty text, so any failure degrades to
    the character estimate rather than raising: the embedder may be unloaded (the
    registration probe, or chunking outside an ingest), and a raise or zero would
    abort extraction or make every span look within budget.
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


register_binding(
    XbergBinding(
        kind=BackendKind.TOKENIZER,
        name=TokenizerBackendName.LILBEE,
        enabled=lambda cfg: cfg.token_sizing,
        make=lambda provider, cfg: LilbeeTokenizerBackend(count_fn=provider.count_tokens),
    )
)
