"""Shared picker-dismiss logic for the model rail and settings screen."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from lilbee.app.services import get_services
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_active_model
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from textual.app import App
    from textual.widget import Widget

    from lilbee.cli.tui.screens.model_picker import PickerScope

log = logging.getLogger(__name__)

# Name for the thread worker that persists + reloads a non-chat role off the
# event loop (the chat scope has its own worker in chat.py).
_PERSIST_WORKER_NAME = "model_swap_persist"

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
    reload_worker: bool = True,
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

    Pass ``reload_worker=False`` when the caller resets the worker another
    way (the chat screen cancels its stream and resets services on a chat
    swap, so reloading the chat role here too would tear that work down twice).
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
        _push_embed_swap_confirm(host, key, ref, on_done, reload_worker)
        return
    _persist(host.app, key, ref, on_done, reload_worker)


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


def _push_embed_swap_confirm(
    host: Widget, key: str, ref: str, on_done: Callable[[], None], reload_worker: bool
) -> None:
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    host.app.push_screen(
        ConfirmDialog(msg.EMBED_SWAP_CONFIRM_TITLE, msg.EMBED_SWAP_CONFIRM_MESSAGE),
        lambda confirmed: _on_embed_confirm(host.app, key, ref, confirmed, on_done, reload_worker),
    )


def _on_embed_confirm(
    app: App,
    key: str,
    ref: str,
    confirmed: bool | None,
    on_done: Callable[[], None],
    reload_worker: bool,
) -> None:
    if not confirmed:
        app.notify(msg.EMBED_SWAP_CANCELLED)
        return
    _persist(app, key, ref, on_done, reload_worker)


def _persist(
    app: App, key: str, ref: str, on_done: Callable[[], None], reload_worker: bool
) -> None:
    """Persist the picked ref and reload the affected worker without freezing the UI.

    A worker reload is a multi-second fleet restart, so when one is needed the
    write and reload run in a thread worker behind an indicator toast and
    ``on_done`` runs back on the main thread. The chat scope passes
    ``reload_worker=False`` and resets services itself (see
    ``ChatScreen.apply_model_change``), so its cheap config write stays inline.
    """
    role = _MODEL_KEY_TO_WORKER_ROLE.get(key)
    if not (reload_worker and role is not None):
        apply_active_model(app, key, ref)
        on_done()
        return

    target_role = role  # narrowed to non-None; bind for the worker closure
    app.notify(msg.MODEL_SWAP_APPLYING)

    def _runner() -> None:
        try:
            apply_active_model(app, key, ref)
            get_services().reload_role(target_role)
        except Exception as exc:  # any reload failure becomes a toast, never a crash
            call_from_thread(
                app, app.notify, msg.MODEL_SWAP_FAILED.format(error=exc), severity="error"
            )
            return
        call_from_thread(app, _finish)

    def _finish() -> None:
        on_done()
        app.notify(msg.MODEL_SWAP_DONE.format(name=ref))

    app.run_worker(_runner, thread=True, exit_on_error=False, name=_PERSIST_WORKER_NAME)
