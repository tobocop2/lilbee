"""A per-test record of every toast Textual shows, kept after Textual's own list drops it."""

from __future__ import annotations

from textual.notifications import Notification, Notifications

_posted: list[Notification] = []


def posted_toasts() -> list[Notification]:
    """Every toast shown since the current test began, oldest first, expired ones included."""
    return list(_posted)


def reset_posted_toasts() -> None:
    """Forget the toasts of the previous test."""
    _posted.clear()


def recording_add(original):
    """Wrap ``Notifications.add`` so each added toast is also appended to the record."""

    def add(self: Notifications, notification: Notification) -> Notifications:
        _posted.append(notification)
        return original(self, notification)

    return add
