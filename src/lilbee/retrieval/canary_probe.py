"""The canary probe of a store: it embeds canaries with the configured embedder."""

from __future__ import annotations

import uuid

from lilbee.core.config import Config
from lilbee.data.store.canary import CanaryProbe
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.base import ProviderError
from lilbee.providers.engine_params import resolve_model_path
from lilbee.providers.fleet.binary import engine_build_id
from lilbee.providers.model_ref import parse_model_ref
from lilbee.retrieval.embedder import Embedder

# A remote embedder has no build to name, so each process names it afresh and checks once.
_PROCESS_TOKEN = uuid.uuid4().hex


def embedder_trigger(config: Config, registry: ModelRegistry) -> str | None:
    """Name the engine build and the model file that embed, or the process for a remote embedder."""
    model = config.embedding_model
    if not model:
        return None
    if parse_model_ref(model).is_remote:
        return f"process:{_PROCESS_TOKEN}:{model}"
    try:
        path = resolve_model_path(model, registry)
        stat = path.stat()
    except (ProviderError, OSError):
        return None
    return f"{engine_build_id()}|{path}@{stat.st_size}-{stat.st_mtime_ns}"


def canary_probe(config: Config, embedder: Embedder, registry: ModelRegistry) -> CanaryProbe:
    """The probe for a store whose vectors *embedder* makes."""
    return CanaryProbe(
        embed=embedder.embed_batch, trigger=lambda: embedder_trigger(config, registry)
    )
