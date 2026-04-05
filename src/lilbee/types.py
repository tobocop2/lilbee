"""Shared type definitions for lilbee."""

from typing import Literal

# Model task classification — the only valid values for task fields.
TaskType = Literal["chat", "embedding", "vision"]

TASK_CHAT: TaskType = "chat"
TASK_EMBEDDING: TaskType = "embedding"
TASK_VISION: TaskType = "vision"
