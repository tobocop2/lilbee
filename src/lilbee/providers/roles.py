"""Engine-neutral role identifiers shared across the provider stack.

``WorkerRole`` names the four inference roles a local engine serves; the fleet
maps each to one llama-server instance. ``OcrBackend`` names the PDF-OCR paths.
These outlive any particular engine, so they live here rather than inside an
engine-specific module.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class WorkerRole(StrEnum):
    """Inference role identifier; addresses one llama-server in the fleet."""

    EMBED = "embed"
    RERANK = "rerank"
    CHAT = "chat"
    VISION = "vision"


OcrBackend = Literal["vision"]
"""PDF-OCR backends routed to the engine. Tesseract runs inline, not on a server."""
