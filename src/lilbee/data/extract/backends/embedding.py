"""lilbee's embedder exposed as a xberg plugin embedding backend.

xberg's semantic chunker needs embeddings to detect topic boundaries. This routes
them to lilbee's own embedder, so the same model vectorizes and splits chunks;
without it xberg falls back to its bundled ONNX preset.
"""

from __future__ import annotations

import itertools
import threading
import weakref
from typing import TYPE_CHECKING, ClassVar

from lilbee.core.config import cfg
from lilbee.data.types import EmbeddingBackendName

from .released import ReleasedNames

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.core.vectors import Vector
    from lilbee.providers.base import LLMProvider


class LilbeeEmbeddingBackend:
    """Routes xberg's boundary-detection embeddings to lilbee's embedder.

    ``embed_fn``/``dim_fn`` are read live, so an embedding-model swap needs no
    re-registration. xberg calls these sync methods on its own worker threads.
    """

    def __init__(
        self,
        *,
        name: str,
        embed_fn: Callable[[list[str]], list[Vector]],
        dim_fn: Callable[[], int],
    ) -> None:
        self._name = name
        self._embed_fn = embed_fn
        self._dim_fn = dim_fn

    def name(self) -> str:
        return self._name

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def dimensions(self) -> int:
        return self._dim_fn()

    def embed(self, texts: list[str]) -> list[Vector]:
        return self._embed_fn(texts)


class _ProviderEmbedders:
    """One registered embedding backend per provider.

    A name stays registered while its provider lives, so a config built for one
    provider keeps its embedder while another provider registers its own. The name
    of a provider that is gone is unregistered by the next name lookup. Each backend
    holds its provider weakly; an embed already in flight when its provider is gone
    fails.
    """

    _serial: ClassVar[itertools.count[int]] = itertools.count(1)
    """Numbers names across every table, as xberg's registry is one per process."""

    def __init__(self) -> None:
        self._names: weakref.WeakKeyDictionary[LLMProvider, str] = weakref.WeakKeyDictionary()
        self._released = ReleasedNames()
        self._lock = threading.Lock()

    def name_for(self, provider: LLMProvider) -> str:
        """The registered embedding name for *provider*, registering it on first use.

        The lock guards the name table only, so a slow registration never blocks the
        name of a provider that is already registered. Two first uses at once both
        register; the first name stored wins, the other is released.
        """
        for released in self._released.take():
            _unregister(released)
        with self._lock:
            name = self._names.get(provider)
            if name is not None:
                return name
            name = f"{EmbeddingBackendName.LILBEE}-{next(self._serial)}"
        _register(
            LilbeeEmbeddingBackend(
                name=name,
                embed_fn=self._embedder(provider),
                dim_fn=lambda: cfg.embedding_dim,
            )
        )
        with self._lock:
            stored = self._names.setdefault(provider, name)
        if stored != name:
            self._released.release(name)
        else:
            self._released.release_when_collected(provider, name)
        return stored

    @staticmethod
    def _embedder(provider: LLMProvider) -> Callable[[list[str]], list[Vector]]:
        """An embed through *provider*, held weakly."""
        ref = weakref.ref(provider)

        def embed(texts: list[str]) -> list[Vector]:
            live = ref()
            if live is None:
                raise ReferenceError("the provider of this embedding backend no longer exists")
            return live.embed(texts)

        return embed


def _register(backend: LilbeeEmbeddingBackend) -> None:
    """Add *backend* to xberg's embedding registry."""
    import xberg

    xberg.register_embedding_backend(backend)


def _unregister(name: str) -> None:
    """Remove the backend *name* from xberg's embedding registry; a no-op when absent."""
    import xberg

    xberg.unregister_embedding_backend(name)


provider_embedders = _ProviderEmbedders()
