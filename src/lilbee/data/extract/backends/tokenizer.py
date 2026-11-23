"""lilbee's embedder tokenizer exposed as a xberg plugin tokenizer backend.

xberg's chunk sizer can budget in tokens, but only with tokenizers it loads itself;
lilbee's embedder is a GGUF model whose vocab isn't published that way. This lets
``chunk_size`` be a real token ceiling instead of a chars-per-token guess (off by
2-4x on token-dense text).
"""

from __future__ import annotations

import itertools
import logging
import threading
import weakref
from typing import TYPE_CHECKING

from lilbee.data.types import TokenizerBackendName
from lilbee.providers.base import ProviderError, ProviderErrorKind

from .released import ReleasedNames

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.providers.base import LLMProvider

log = logging.getLogger(__name__)

# Over-counting fallback for when the exact count is unavailable: splits a touch
# early rather than emitting an over-length chunk. Mirrors providers.fleet.client.
_FALLBACK_CHARS_PER_TOKEN = 3

# The ChunkSizing type that routes chunk sizing to a registered tokenizer backend.
TOKENIZER_SIZING_TYPE = "tokenizer"

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

    def __init__(
        self, *, count_fn: Callable[[str], int], name: str = TokenizerBackendName.LILBEE
    ) -> None:
        self._count_fn = count_fn
        self._name = name
        self._probe_count: int | None = None

    def name(self) -> str:
        return self._name

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


class TokenCountError(ProviderError):
    """The embedder failed to count tokens while chunks were sized."""

    def __init__(self) -> None:
        super().__init__(
            "The embedding model failed to count tokens while sizing chunks, so the text "
            "was not indexed. Check that the embedding model is running, then try again.",
            kind=ProviderErrorKind.CONNECTION,
        )


class _ProviderTokenizers:
    """One registered tokenizer backend per provider, and the count failures of each name.

    A name stays registered while its provider lives, so a config built for one
    provider keeps its tokenizer while another provider registers its own. Each
    backend holds its provider weakly. The name of a provider that is gone is
    unregistered by the next registration, and its configs fail their counts.
    """

    def __init__(self) -> None:
        self._names: weakref.WeakKeyDictionary[LLMProvider, str] = weakref.WeakKeyDictionary()
        self._serial = itertools.count(1)
        self._failures: dict[str, int] = {}
        self._released = ReleasedNames()
        self._lock = threading.Lock()

    def name_for(self, provider: LLMProvider) -> str:
        """The registered tokenizer name for *provider*, registering it on first use.

        Two first uses at once both register; the first name stored wins, the other is released.
        """
        self._unregister_released()
        with self._lock:
            name = self._names.get(provider)
            if name is not None:
                return name
            name = f"{TokenizerBackendName.LILBEE}-{next(self._serial)}"
        backend = LilbeeTokenizerBackend(count_fn=self._counter(provider, name), name=name)
        backend.probe()
        _register(backend)
        with self._lock:
            stored = self._names.setdefault(provider, name)
            if stored != name:
                self._released.release(name)
                return stored
            self._failures[name] = 0
        self._released.release_when_collected(provider, name)
        return name

    def failures(self, name: str) -> int | None:
        """How many counts of *name* have failed, or None when its provider is gone."""
        with self._lock:
            return self._failures.get(name)

    def _unregister_released(self) -> None:
        """Unregister the name of each released provider and drop its count."""
        for name in self._released.take():
            with self._lock:
                self._failures.pop(name, None)
            _unregister(name)

    def _counter(self, provider: LLMProvider, name: str) -> Callable[[str], int]:
        """A count through *provider*, held weakly, that records each failure against *name*."""
        ref = weakref.ref(provider)

        def count(text: str) -> int:
            live = ref()
            try:
                if live is None:
                    raise ReferenceError("the provider of this tokenizer no longer exists")
                return live.count_tokens(text)
            except NotImplementedError:
                raise
            except Exception:
                log.debug("token count for chunk sizing failed", exc_info=True)
                with self._lock:
                    if name in self._failures:
                        self._failures[name] += 1
                raise

        return count


def _register(backend: LilbeeTokenizerBackend) -> None:
    """Add *backend* to xberg's tokenizer registry."""
    import xberg

    xberg.register_tokenizer_backend(backend)


def _unregister(name: str) -> None:
    """Remove the backend *name* from xberg's tokenizer registry."""
    import xberg

    xberg.unregister_tokenizer_backend(name)


provider_tokenizers = _ProviderTokenizers()
