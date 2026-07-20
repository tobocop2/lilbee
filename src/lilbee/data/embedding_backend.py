"""lilbee's embedder exposed as a xberg plugin embedding backend.

xberg's semantic chunker needs embeddings to detect topic boundaries. Registering
this backend routes those embeddings to lilbee's own embedder (whatever model the
fleet serves), so the model that vectorizes chunks for retrieval is the same one
that decides where chunks split. Without it, xberg falls back to its bundled ONNX
preset, a different model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.data.ingest.types import EmbeddingBackendName

if TYPE_CHECKING:
    from collections.abc import Callable


class LilbeeEmbeddingBackend:
    """Routes xberg's boundary-detection embeddings to lilbee's embedder.

    The functional path is injected: ``embed_fn`` is the provider's batch embed
    (read live, so an embedding-model swap needs no re-registration) and
    ``dim_fn`` reports the current vector width. xberg calls ``initialize`` once
    at registration, then ``dimensions`` and ``embed`` while chunking. Methods are
    sync; xberg invokes them on its own worker threads.
    """

    def __init__(
        self,
        *,
        embed_fn: Callable[[list[str]], list[list[float]]],
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed_fn(texts)
