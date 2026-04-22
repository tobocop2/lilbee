"""Shared visual tokens for catalog widgets (grid card + list row)."""

from __future__ import annotations

from lilbee.models import ModelTask

MIDDLE_DOT = "·"

# Rerank shares the embedding pill colour because both are retrieval-precision
# adjuncts, keeping the grid legible without a dedicated palette entry.
TASK_COLORS: dict[str, str] = {
    ModelTask.CHAT: "$primary",
    ModelTask.EMBEDDING: "$secondary",
    ModelTask.VISION: "$warning",
    ModelTask.RERANK: "$secondary",
}
