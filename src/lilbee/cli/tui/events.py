"""Typed Textual Message subclasses for cross-widget communication.

These complement messages.py (which holds user-facing string constants).
Widgets post these messages for structured, typed event handling.
"""

from dataclasses import dataclass

from textual.message import Message

from lilbee.models import ModelTask


@dataclass
class ModelChanged(Message):
    """Fired when the active chat, embedding, or vision model changes."""

    role: ModelTask
    name: str


@dataclass
class TaskStateChanged(Message):
    """Fired when a background task changes state."""

    task_id: str
    status: str  # "queued" | "active" | "done" | "failed" | "cancelled"


@dataclass
class ViewSwitched(Message):
    """Fired when navigation switches to a different view."""

    view_name: str


@dataclass
class SyncRequested(Message):
    """Request a document sync from any screen."""


@dataclass
class CatalogViewToggled(Message):
    """Toggle between grid and list view in catalog."""

    view_mode: str  # "grid" | "list"
