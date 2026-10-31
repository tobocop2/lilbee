"""The Presets tab of the Settings screen, with its add and edit dialog."""

from __future__ import annotations

import logging
from typing import ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from lilbee.app.presets import (
    ServerPreset,
    add_preset,
    edit_preset,
    failure_text,
    list_presets,
    remove_preset,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.preset_commands import key_state_text, preset_line
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import RerankWire

log = logging.getLogger(__name__)

PRESETS_PANE_ID = "settings-tab-presets"


class PresetDialog(ModalScreen[str | None]):
    """Add a preset, or edit one. It dismisses with the notice to show, or None on cancel."""

    CSS_PATH = "settings_presets.tcss"
    AUTO_FOCUS = "#preset-url-input"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, preset: ServerPreset | None = None) -> None:
        super().__init__()
        self._preset = preset

    def compose(self) -> ComposeResult:
        preset = self._preset
        roles = set(preset.roles) if preset else set()
        wire = preset.rerank_wire if preset and preset.rerank_wire else RerankWire.V1
        with VerticalScroll():
            title = (
                msg.PRESET_DIALOG_ADD_TITLE
                if preset is None
                else msg.PRESET_DIALOG_EDIT_TITLE.format(name=preset.name)
            )
            yield Static(title, id="preset-title", markup=False)
            if preset is None:
                yield Label(msg.PRESET_DIALOG_NAME_LABEL)
                yield Input(id="preset-name-input")
            yield Label(msg.PRESET_DIALOG_URL_LABEL)
            yield Input(
                value=preset.base_url if preset else "",
                placeholder=msg.PRESET_DIALOG_URL_PLACEHOLDER,
                id="preset-url-input",
            )
            yield Label(msg.PRESET_DIALOG_ROLES_LABEL)
            for role in WorkerRole:
                yield Checkbox(role.value, value=role in roles, id=f"preset-role-{role.value}")
            yield Label(msg.PRESET_DIALOG_WIRE_LABEL, id="preset-wire-label")
            yield Select(
                [(w.value, w) for w in RerankWire],
                value=wire,
                allow_blank=False,
                id="preset-wire-select",
            )
            yield Label(msg.PRESET_DIALOG_KEY_LABEL)
            yield Input(
                password=True,
                placeholder=msg.PRESET_DIALOG_KEY_ADD_PLACEHOLDER
                if preset is None
                else msg.PRESET_DIALOG_KEY_EDIT_PLACEHOLDER,
                id="preset-key-input",
            )
            if preset is not None:
                yield Static(
                    msg.PRESET_DIALOG_KEY_STATE.format(state=key_state_text(preset)),
                    id="preset-key-state",
                    markup=False,
                )
            yield Static("", id="preset-error", markup=False)
            with Center(), Horizontal(id="preset-buttons"):
                yield Button(msg.PRESET_DIALOG_SAVE, variant="primary", id="preset-save")
                if preset is not None and preset.has_key:
                    yield Button(
                        msg.PRESET_DIALOG_CLEAR_KEY, name=preset.name, id="preset-clear-key"
                    )
                yield Button(msg.PRESET_DIALOG_CANCEL, id="preset-cancel")

    def on_mount(self) -> None:
        self._show_wire()

    @on(Checkbox.Changed, "#preset-role-rerank")
    def _show_wire(self) -> None:
        """The wire choice shows while the rerank role is on."""
        rerank_on = self.query_one("#preset-role-rerank", Checkbox).value
        self.query_one("#preset-wire-label").display = rerank_on
        self.query_one("#preset-wire-select").display = rerank_on

    def _chosen_roles(self) -> list[str]:
        return [
            r.value for r in WorkerRole if self.query_one(f"#preset-role-{r.value}", Checkbox).value
        ]

    def _chosen_wire(self) -> str:
        return str(self.query_one("#preset-wire-select", Select).value)

    def _set_busy(self, busy: bool) -> None:
        for button in self.query(Button):
            button.disabled = busy

    def _show_error(self, text: str) -> None:
        self.query_one("#preset-error", Static).update(text)
        self._set_busy(False)

    def _finish(self, notice: str) -> None:
        self.dismiss(notice)

    @on(Button.Pressed, "#preset-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#preset-save")
    @on(Input.Submitted)
    def _on_save(self) -> None:
        self._set_busy(True)
        self.query_one("#preset-error", Static).update("")
        if self._preset is None:
            self._save_new(
                self.query_one("#preset-name-input", Input).value,
                self.query_one("#preset-url-input", Input).value,
                self._chosen_roles(),
                self._chosen_wire(),
                self.query_one("#preset-key-input", Input).value,
            )
        else:
            self._save_edit(
                self._preset,
                self.query_one("#preset-url-input", Input).value,
                self._chosen_roles(),
                self._chosen_wire(),
                self.query_one("#preset-key-input", Input).value,
            )

    @on(Button.Pressed, "#preset-clear-key")
    def _on_clear_key(self, event: Button.Pressed) -> None:
        self._set_busy(True)
        self._run_edit(str(event.button.name), clear_key=True)

    @work(thread=True)
    def _save_new(self, name: str, url: str, roles: list[str], wire: str, key: str) -> None:
        try:
            preset = add_preset(name, url, roles=roles, api_key=key, rerank_wire=wire)
        except (ValueError, OSError) as exc:
            call_from_thread(self, self._show_error, failure_text(exc))
            return
        call_from_thread(self, self._finish, msg.PRESET_ADDED.format(name=preset.name))

    def _save_edit(
        self, preset: ServerPreset, url: str, roles: list[str], wire: str, key: str
    ) -> None:
        """Send only what differs from the stored preset; the service says when none does."""
        changes: dict[str, object] = {}
        if url != preset.base_url:
            changes["url"] = url
        if set(roles) != {r.value for r in preset.roles}:
            changes["roles"] = roles
        if WorkerRole.RERANK.value in roles and wire != preset.rerank_wire:
            changes["rerank_wire"] = wire
        if key:
            changes["api_key"] = key
        self._run_edit(preset.name, **changes)

    @work(thread=True)
    def _run_edit(self, name: str, **changes: object) -> None:
        try:
            preset = edit_preset(name, **changes)  # type: ignore[arg-type]
        except (ValueError, OSError) as exc:
            call_from_thread(self, self._show_error, failure_text(exc))
            return
        call_from_thread(self, self._finish, msg.PRESET_UPDATED.format(name=preset.name))


class PresetsPane(VerticalScroll):
    """The list of stored presets with add, edit and remove buttons."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._presets: dict[str, ServerPreset] = {}

    def compose(self) -> ComposeResult:
        yield Button(msg.SETTINGS_PRESETS_ADD, variant="primary", id="presets-add")
        yield Static(msg.SETTINGS_PRESETS_EMPTY, id="presets-status", markup=False)
        yield Vertical(id="presets-rows")

    def on_mount(self) -> None:
        self.reload()

    @work(thread=True, exclusive=True)
    def reload(self) -> None:
        """Read the stored presets off the event loop, then show them."""
        try:
            presets = list_presets()
        except (ValueError, OSError) as exc:
            call_from_thread(self, self._show_status, failure_text(exc))
            return
        call_from_thread(self, self._show_presets, presets)

    def _show_status(self, text: str) -> None:
        self.query_one("#presets-status", Static).update(text)

    def _show_presets(self, presets: list[ServerPreset]) -> None:
        self._show_status("" if presets else msg.SETTINGS_PRESETS_EMPTY)
        self._presets = {preset.name: preset for preset in presets}
        rows = self.query_one("#presets-rows", Vertical)
        rows.remove_children()
        for preset in presets:
            rows.mount(
                Horizontal(
                    Static(preset_line(preset), classes="preset-line", markup=False),
                    Button(msg.SETTINGS_PRESETS_EDIT, name=preset.name, classes="preset-edit"),
                    Button(msg.SETTINGS_PRESETS_REMOVE, name=preset.name, classes="preset-remove"),
                    classes="preset-row",
                )
            )

    def _after_dialog(self, notice: str | None) -> None:
        if notice is None:
            return
        self.notify(notice)
        self.reload()

    @on(Button.Pressed, "#presets-add")
    def _on_add(self) -> None:
        self.app.push_screen(PresetDialog(), self._after_dialog)

    @on(Button.Pressed, ".preset-edit")
    def _on_edit(self, event: Button.Pressed) -> None:
        self.app.push_screen(
            PresetDialog(self._presets[str(event.button.name)]), self._after_dialog
        )

    @on(Button.Pressed, ".preset-remove")
    def _on_remove(self, event: Button.Pressed) -> None:
        name = str(event.button.name)

        def confirmed(answer: bool | None) -> None:
            if answer:
                self._remove(name)

        self.app.push_screen(
            ConfirmDialog(
                msg.PRESET_REMOVE_CONFIRM_TITLE, msg.PRESET_REMOVE_CONFIRM_MESSAGE.format(name=name)
            ),
            confirmed,
        )

    @work(thread=True)
    def _remove(self, name: str) -> None:
        try:
            remove_preset(name)
        except (ValueError, OSError) as exc:
            call_from_thread(self, self.notify, failure_text(exc), severity="error")
            return
        call_from_thread(self, self._after_dialog, msg.PRESET_REMOVED.format(name=name))
