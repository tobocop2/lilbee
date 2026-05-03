"""Subprocess workers for llama-cpp inference (embed, chat, rerank, vision).

Isolates llama-cpp native code in subprocesses so the parent event loop
stays responsive. Two transports coexist:

* :class:`WorkerPool` (the default): persistent per-role worker processes
  exchanging pickled messages over a duplex :class:`multiprocessing.Pipe`.
  See :mod:`lilbee.providers.worker.pool` and
  :mod:`lilbee.providers.worker.channel`.
* :class:`WorkerManager` (legacy): per-call subprocess for embed and
  vision only, reachable when ``cfg.worker_pool_enabled = False``.
  Preserved so existing tests that mock ``Llama`` directly keep working.
"""

from lilbee.providers.worker.legacy_protocol import (
    ConfigSnapshot,
    EmbedRequest,
    EmbedResponse,
    LoadModelRequest,
    ShutdownRequest,
    VisionRequest,
    VisionResponse,
)
from lilbee.providers.worker.manager import WorkerManager

__all__ = [
    "ConfigSnapshot",
    "EmbedRequest",
    "EmbedResponse",
    "LoadModelRequest",
    "ShutdownRequest",
    "VisionRequest",
    "VisionResponse",
    "WorkerManager",
]
