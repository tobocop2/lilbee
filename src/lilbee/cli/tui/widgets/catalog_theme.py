"""Shared visual tokens for catalog widgets (grid card + list row)."""

from __future__ import annotations

from lilbee.catalog.types import ModelTask

MIDDLE_DOT = "·"

TASK_COLORS: dict[str, str] = {
    ModelTask.CHAT: "$primary",
    ModelTask.EMBEDDING: "$secondary",
    ModelTask.VISION: "$warning",
    ModelTask.RERANK: "$success",
}
