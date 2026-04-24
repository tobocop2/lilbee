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
