"""Subprocess worker for llama-cpp embedding and vision operations.

Isolates llama-cpp native code (which holds the GIL) in a separate
process so the parent event loop stays responsive. Communication uses
multiprocessing queues with the spawn start method (fork-unsafe with
threads).
"""

from lilbee.providers.worker.manager import WorkerManager
from lilbee.providers.worker.protocol import (
    ConfigSnapshot,
    EmbedRequest,
    EmbedResponse,
    LoadModelRequest,
    ShutdownRequest,
    VisionRequest,
    VisionResponse,
)

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
