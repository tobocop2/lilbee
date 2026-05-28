"""Left-pinned panel that surfaces all four model roles at once."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from lilbee.catalog.types import ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.settings_widgets import model_field_to_picker_scope
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.model_bar import (
    ChatModeToggle,
    ModelOption,
    ModelPickerButton,
    _cloud_provider_label,
    classify_installed_models_full,
)
from lilbee.cli.tui.widgets.model_pick import config_key_for_scope
from lilbee.core.config import cfg

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.model_picker import PickerScope

log = logging.getLogger(__name__)

_CSS_FILE = Path(__file__).parent / "model_rail.tcss"

_NULLABLE_SCOPES: frozenset[str] = frozenset({"vision", "rerank"})

_SCOPE_TO_LABEL: dict[str, str] = {
    "chat": msg.MODEL_RAIL_CHAT_LABEL,
    "embed": msg.MODEL_RAIL_EMBED_LABEL,
    "vision": msg.MODEL_RAIL_VISION_LABEL,
    "rerank": msg.MODEL_RAIL_RERANK_LABEL,
}

_SCOPE_TINT_CLASS: dict[str, str] = {"vision": "-vision", "rerank": "-rerank"}

_CLOUD_WARNING_ID = "rail-cloud-warning"


class RoleRow(Widget, can_focus=False):
    """One row of the model rail: status dot, role label, picker button."""

    def __init__(self, *, scope: PickerScope) -> None:
        super().__init__()
        self.scope: PickerScope = scope
        self._key: str = config_key_for_scope(scope)
        tint = _SCOPE_TINT_CLASS.get(scope)
        if tint is not None:
            self.add_class(tint)

    def compose(self) -> ComposeResult:
        yield Static("", classes="role-dot")
        yield Static(_SCOPE_TO_LABEL[self.scope], classes="role-label")
        yield ModelPickerButton(scope=self.scope, button_id=f"rail-pick-{self.scope}")

    def on_mount(self) -> None:
        self.refresh_state()

    @property
    def is_active(self) -> bool:
        return bool(getattr(cfg, self._key))

    def refresh_state(self) -> None:
        """Repaint the dot + active/off classes from current cfg."""
        active = self.is_active
        self.set_class(active, "-active")
        self.set_class(not active, "-off")
        with contextlib.suppress(Exception):
            dot = self.query_one(".role-dot", Static)
            dot.update(msg.ACTIVE_DOT if active else msg.INACTIVE_DOT)


class ModelRail(Widget, can_focus=False):
    """Left-pinned rail listing the four model roles and the chat/search mode toggle."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    _ACTIVE_SCOPES: ClassVar[tuple[str, ...]] = ("chat", "embed")
    _OPTIONAL_SCOPES: ClassVar[tuple[str, ...]] = ("vision", "rerank")

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._options_cache: dict[str, tuple[tuple[str, str], ...]] = {}

    def compose(self) -> ComposeResult:
        yield Static(msg.MODEL_RAIL_HEADING, classes="rail-heading")
        for scope in self._ACTIVE_SCOPES:
            yield RoleRow(scope=scope)
        yield Static(msg.MODEL_RAIL_OPTIONAL_HEADING, classes="rail-heading")
        for scope in self._OPTIONAL_SCOPES:
            yield RoleRow(scope=scope)
        yield ChatModeToggle()
        yield Static("", id=_CLOUD_WARNING_ID, classes="cloud-warning")

    def on_mount(self) -> None:
        self._refresh_cloud_warning()
        self._scan_models()
        self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        key, _ = payload
        scope = model_field_to_picker_scope().get(key)
        if scope is None:
            return
        for row in self.query(RoleRow):
            if row.scope == scope:
                row.refresh_state()
        if key == "chat_model":
            self._refresh_cloud_warning()

    @work(thread=True)
    def _scan_models(self) -> None:
        """Scan installed models off the UI thread and populate every role button."""
        buckets = classify_installed_models_full()
        scope_to_options: dict[str, list[ModelOption]] = {
            "chat": list(buckets.get(ModelTask.CHAT, [])),
            "embed": list(buckets.get(ModelTask.EMBEDDING, [])),
            "vision": list(buckets.get(ModelTask.VISION, [])),
            "rerank": list(buckets.get(ModelTask.RERANK, [])),
        }
        call_from_thread(self, self._populate, scope_to_options)

    def _populate(self, scope_to_options: dict[str, list[ModelOption]]) -> None:
        for row in self.query(RoleRow):
            opts = scope_to_options.get(row.scope, []) or [
                ModelOption(label=msg.MODEL_VALUE_NONE, ref="")
            ]
            fingerprint = tuple((o.label, o.ref) for o in opts)
            if self._options_cache.get(row.scope) != fingerprint:
                row.query_one(ModelPickerButton).set_options(opts)
                self._options_cache[row.scope] = fingerprint
        self._refresh_cloud_warning()

    def _refresh_cloud_warning(self) -> None:
        """Show a warning if the active chat model routes to a cloud provider."""
        warning = self.query_one(f"#{_CLOUD_WARNING_ID}", Static)
        label = _cloud_provider_label(cfg.chat_model)
        if label is None:
            warning.remove_class("-visible")
            return
        warning.update(msg.MODEL_BAR_CLOUD_PROVIDER_WARNING.format(provider=label))
        warning.add_class("-visible")

    def refresh_models(self) -> None:
        """Re-scan installed models (called after downloads complete)."""
        self._scan_models()


__all__ = ["ModelRail", "RoleRow"]
