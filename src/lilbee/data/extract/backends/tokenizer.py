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

    from lilbee.core.config.model import Config
    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

# Over-counting fallback for when the exact count is unavailable: splits a touch
# early rather than emitting an over-length chunk. Mirrors providers.fleet.client.
_FALLBACK_CHARS_PER_TOKEN = 3

# The text xberg's registration probe counts. Its bridge reports an exception as a zero count.
_XBERG_PROBE_TEXT = "a"


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate from character length (ceiling division)."""
    return max(1, -(-len(text) // _FALLBACK_CHARS_PER_TOKEN))


class LilbeeTokenizerBackend:
    """Counts chunk-sizing tokens with lilbee's embedder tokenizer.

    ``count_fn`` is read live (an embedding-model swap needs no re-registration).
    Only a backend without a local tokenizer degrades to the character estimate;
    any other failure raises to the caller. xberg requires a non-zero count for
    non-empty text, so a zero count degrades to the estimate as well.
    After :meth:`probe`, xberg's registration probe is answered once without a
    provider call, so a provider that stops after the probe cannot fail the
    registration. Later counts of the same text go to the provider.
    """

    def __init__(self, *, count_fn: Callable[[str], int]) -> None:
        self._count_fn = count_fn
        self._probe_count: int | None = None

    def name(self) -> str:
        return TokenizerBackendName.LILBEE

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def probe(self) -> None:
        """Count xberg's probe text through the provider; a count failure raises here."""
        self._probe_count = self._count(_XBERG_PROBE_TEXT)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if text == _XBERG_PROBE_TEXT:
            answer, self._probe_count = self._probe_count, None
            if answer is not None:
                return answer
        return self._count(text)

    def _count(self, text: str) -> int:
        try:
            count = self._count_fn(text)
        except NotImplementedError:  # SDK embedders expose no local tokenizer
            log.debug("exact token count failed; using character estimate", exc_info=True)
            return _estimate_tokens(text)
        return count if count > 0 else _estimate_tokens(text)


def _make_tokenizer_backend(provider: LLMProvider, cfg: Config) -> LilbeeTokenizerBackend:
    """Build the tokenizer backend, probing its count before xberg does."""
    backend = LilbeeTokenizerBackend(count_fn=provider.count_tokens)
    backend.probe()
    return backend


register_binding(
    XbergBinding(
        kind=BackendKind.TOKENIZER,
        name=TokenizerBackendName.LILBEE,
        enabled=lambda cfg: cfg.token_sizing,
        make=_make_tokenizer_backend,
    )
)
