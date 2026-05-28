"""Shared picker-dismiss logic for the model rail and settings screen."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from lilbee.app.services import get_services
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_active_model
from lilbee.providers.worker.transport import WorkerRole

if TYPE_CHECKING:
    from textual.app import App
    from textual.widget import Widget

log = logging.getLogger(__name__)

# Single source of truth for "after a model-key write, which worker pool role
# needs to respawn so the next call picks up the new ref?". Used by both the
# Settings picker dismiss path and the chat-screen model rail's button.
_MODEL_KEY_TO_WORKER_ROLE: dict[str, WorkerRole] = {
    "chat_model": WorkerRole.CHAT,
    "embedding_model": WorkerRole.EMBED,
    "reranker_model": WorkerRole.RERANK,
    "vision_model": WorkerRole.VISION,
}


def apply_model_pick(
    host: Widget,
    *,
    key: str,
    ref: str | None,
    on_done: Callable[[], None],
) -> None:
    """Persist a picker selection and reload the affected worker.

    ``ref is None`` means the user cancelled (Esc); leave the field alone.
    ``ref == ""`` for a nullable field means the user picked the explicit
    "disabled" row; clear the field. Embedding-model swaps against a
    populated store route through a confirm modal first so the user is
    not surprised by the rebuild requirement. ``on_done`` runs after a
    successful write, never after a cancel.
    """
    if ref is None:
        return
    defn = SETTINGS_MAP.get(key)
    if not ref and (defn is None or not defn.nullable):
        return
    if key == "embedding_model" and ref and get_services().store.has_chunks():
        _push_embed_swap_confirm(host, key, ref, on_done)
        return
    _persist(host.app, key, ref, on_done)


def _push_embed_swap_confirm(host: Widget, key: str, ref: str, on_done: Callable[[], None]) -> None:
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    host.app.push_screen(
        ConfirmDialog(msg.EMBED_SWAP_CONFIRM_TITLE, msg.EMBED_SWAP_CONFIRM_MESSAGE),
        lambda confirmed: _on_embed_confirm(host.app, key, ref, confirmed, on_done),
    )


def _on_embed_confirm(
    app: App,
    key: str,
    ref: str,
    confirmed: bool | None,
    on_done: Callable[[], None],
) -> None:
    if not confirmed:
        app.notify(msg.EMBED_SWAP_CANCELLED)
        return
    _persist(app, key, ref, on_done)


def _persist(app: App, key: str, ref: str, on_done: Callable[[], None]) -> None:
    apply_active_model(app, key, ref)
    role = _MODEL_KEY_TO_WORKER_ROLE.get(key)
    if role is not None:
        get_services().reload_role(role)
    on_done()
