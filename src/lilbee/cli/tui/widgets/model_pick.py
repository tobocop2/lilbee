"""Shared picker-dismiss logic for the model rail and settings screen."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from lilbee.app.services import get_services
from lilbee.app.settings import (
    SettingsSnapshot,
    embed_pick_needs_rebuild,
    index_has_chunks,
    restore_settings,
    snapshot_settings,
)
from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_active_model
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.core.config import cfg
from lilbee.providers.roles import MODEL_FIELD_TO_ROLE

if TYPE_CHECKING:
    from textual.app import App
    from textual.widget import Widget

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.model_picker import PickerScope

log = logging.getLogger(__name__)

# Name for the thread worker that reloads a non-chat role off the event loop
# (the chat scope has its own worker in chat.py).
_PERSIST_WORKER_NAME = "model_swap_persist"
# What LanceDB raises from a read or a write, and what config.toml raises from a write.
STORE_ERRORS = (ValueError, OSError, RuntimeError)


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
    An embedding-model swap against a populated store asks one question
    before anything changes: the swap needs a rebuild of the index. Yes
    applies the swap and starts the rebuild; No changes nothing. A swap with
    no question offers the rebuild after it when the write reports a stale
    index. ``on_done`` runs after a successful write and reload, never after a
    cancel, a refused write, a failed reload or the catalog jump.

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
    if key == "embedding_model" and ref != cfg.embedding_model:
        try:
            ask = index_has_chunks() and embed_pick_needs_rebuild(ref)
        except STORE_ERRORS as exc:
            host.app.notify(msg.MODEL_ASSIGN_REJECTED.format(error=exc), severity="error")
            return
        if ask:
            _ask_embed_swap(host.app, key, ref, on_done, reload_worker)
            return
    _swap(host.app, key, ref, on_done, reload_worker, asked=False)


def _ask_embed_swap(
    app: App, key: str, ref: str, on_done: Callable[[], None], reload_worker: bool
) -> None:
    """Ask before an embedder swap that needs a rebuild; only Yes changes anything."""
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    def _answered(rebuild: bool | None) -> None:
        if rebuild:
            _swap(app, key, ref, on_done, reload_worker, asked=True)

    app.push_screen(
        ConfirmDialog(msg.CMD_REBUILD_CONFIRM_TITLE, msg.EMBED_SWAP_REBUILD_MESSAGE),
        _answered,
    )


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


def _swap(
    app: App,
    key: str,
    ref: str,
    on_done: Callable[[], None],
    reload_worker: bool,
    *,
    asked: bool,
) -> None:
    """Write the ref, reload its worker, restore the old ref on failure, then settle the rebuild.

    A refused write ends here. A worker reload is a multi-second fleet restart,
    so it runs in a thread worker behind an indicator toast, and the rest runs
    back on the main thread. The write's verdict decides the rebuild: Yes
    (*asked*) starts it, and a swap with no question offers it.
    """
    before = snapshot_settings([key])
    result = apply_active_model(app, key, ref)
    if result is None:
        return
    lilbee_app = cast("LilbeeApp", app)

    def _settle() -> None:
        on_done()
        if not result.reindex_required:
            return
        if asked:
            lilbee_app.start_rebuild()
        else:
            lilbee_app.offer_rebuild(msg.SETTINGS_REINDEX_MESSAGE)

    role = MODEL_FIELD_TO_ROLE.get(key)
    if not (reload_worker and role is not None):
        _settle()
        return

    target_role = role  # narrowed to non-None; bind for the worker closure
    app.notify(msg.MODEL_SWAP_APPLYING)

    def _runner() -> None:
        try:
            # wait=True runs the reload in this worker thread (already off the event
            # loop) so a failure is caught here and the done toast fires only once
            # the fleet has actually restarted, not before.
            get_services().reload_role(target_role, wait=True)
        except Exception as exc:  # any reload failure becomes a toast, never a crash
            call_from_thread(app, _roll_back, lilbee_app, key, before, exc)
            return
        call_from_thread(app, _finish)

    def _finish() -> None:
        app.notify(msg.MODEL_SWAP_DONE.format(name=ref))
        _settle()

    app.run_worker(_runner, thread=True, exit_on_error=False, name=_PERSIST_WORKER_NAME)


def _roll_back(app: LilbeeApp, key: str, before: SettingsSnapshot, error: Exception) -> None:
    """Put the old model and its config entry back; the toast names the model now in use."""
    try:
        restore_settings(before)
    except STORE_ERRORS:
        log.warning("Could not restore %s after a failed reload", key, exc_info=True)
    in_use = getattr(cfg, key)
    toast = (
        msg.MODEL_SWAP_FAILED.format(error=error)
        if in_use == before.values[key]
        else msg.MODEL_SWAP_STUCK.format(error=error, name=in_use)
    )
    app.notify(toast, severity="error")
    app.publish_settings([key])
