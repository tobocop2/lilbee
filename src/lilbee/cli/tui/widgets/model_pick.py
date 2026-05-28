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

    from lilbee.cli.tui.screens.model_picker import PickerScope

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


def config_key_for_scope(scope: PickerScope) -> str:
    """Inverse of ``model_field_to_picker_scope``: scope -> config attribute name."""
    from lilbee.cli.tui.screens.settings_widgets import model_field_to_picker_scope

    for key, sc in model_field_to_picker_scope().items():
        if sc == scope:
            return key
    raise KeyError(scope)


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
    "disabled" row; clear the field. ``ref == BROWSE_CATALOG_REF`` is the
    on-ramp action: open the Catalog focused on the role's task tab.
    Embedding-model swaps against a populated store route through a
    confirm modal first so the user is not surprised by the rebuild
    requirement. ``on_done`` runs after a successful write, never after
    a cancel and never after the catalog jump.
    """
    if ref is None:
        return
    from lilbee.cli.tui.screens.model_picker import BROWSE_CATALOG_REF

    if ref == BROWSE_CATALOG_REF:
        _open_catalog_for_key(host, key)
        return
    defn = SETTINGS_MAP.get(key)
    if not ref and (defn is None or not defn.nullable):
        return
    if key == "embedding_model" and ref and get_services().store.has_chunks():
        _push_embed_swap_confirm(host, key, ref, on_done)
        return
    _persist(host.app, key, ref, on_done)


def _open_catalog_for_key(host: Widget, key: str) -> None:
    """Push CatalogScreen focused on the task tab matching the role's key."""
    # circular: model_pick -> catalog (catalog imports settings_widgets which
    # imports model_picker, which transitively pulls in model_pick).
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TASK_TO_TAB_ID
    from lilbee.cli.tui.screens.settings_widgets import (
        model_field_to_picker_scope,
        picker_scope_to_task,
    )

    scope = model_field_to_picker_scope().get(key)
    if scope is None:
        log.debug("Cannot open catalog for unknown model key %r", key)
        return
    tab_id = TASK_TO_TAB_ID[picker_scope_to_task(scope)]
    host.app.push_screen(CatalogScreen(focus_task=tab_id))


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
