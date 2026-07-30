"""lilbee's embedder exposed as a xberg plugin embedding backend.

xberg's semantic chunker needs embeddings to detect topic boundaries. This routes
them to lilbee's own embedder, so the same model vectorizes and splits chunks;
without it xberg falls back to its bundled ONNX preset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.data.types import EmbeddingBackendName

from .registry import BackendKind, XbergBinding, register_binding

if TYPE_CHECKING:
    from collections.abc import Callable

    from lilbee.core.vectors import Vector


class LilbeeEmbeddingBackend:
    """Routes xberg's boundary-detection embeddings to lilbee's embedder.

    ``embed_fn``/``dim_fn`` are read live, so an embedding-model swap needs no
    re-registration. xberg calls these sync methods on its own worker threads.
    """

    def __init__(
        self,
        *,
        embed_fn: Callable[[list[str]], list[Vector]],
        dim_fn: Callable[[], int],
    ) -> None:
        self._embed_fn = embed_fn
        self._dim_fn = dim_fn

    def name(self) -> str:
        return EmbeddingBackendName.LILBEE

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def dimensions(self) -> int:
        return self._dim_fn()

    def embed(self, texts: list[str]) -> list[Vector]:
        return self._embed_fn(texts)


register_binding(
    XbergBinding(
        kind=BackendKind.EMBEDDING,
        name=EmbeddingBackendName.LILBEE,
        enabled=lambda cfg: True,
        make=lambda provider, cfg: LilbeeEmbeddingBackend(
            embed_fn=provider.embed, dim_fn=lambda: cfg.embedding_dim
        ),
    )
)
