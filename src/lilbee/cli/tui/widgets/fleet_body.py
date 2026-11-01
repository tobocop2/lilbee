"""FleetBody: the GPU table, live panel, and interactive placement editor.

Reusable widget that can be mounted both as the top-level Fleet view and as
a modal overlay.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from lilbee.app import engine_service
from lilbee.app.placement import (
    get_placement,
    preview_placement,
    set_placement,
    wait_chat_ready,
)
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_commands import ENGINE_SETTING_KEYS, chips_text
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import REPLICATED_ROLES, RoleTarget, WorkerRole

if TYPE_CHECKING:
    from lilbee.app.placement import PlacementView
    from lilbee.cli.tui.app import LilbeeApp

log = logging.getLogger(__name__)

_GGUF_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}$")
_GGUF_QUANT_RE = re.compile(r"-(?:Q\d[\w.]*|IQ\d[\w.]*|F16|BF16|FP16|F32)$", re.IGNORECASE)
_GGUF_REPO_SUFFIXES = ("-GGUF", "-gguf")


def _clean_model_name(ref: str) -> str:
    """Short, human-friendly model name from a GGUF reference.

    "Qwen/Qwen3-235B-A22B-GGUF/Q4_K_M/...-00001-of-00005.gguf" -> "Qwen3-235B-A22B".
    """
    parts = ref.split("/")
    for component in parts[1:]:  # skip the org; the repo name is the friendly one
        if component.endswith(_GGUF_REPO_SUFFIXES):
            return component.rsplit("-", 1)[0]
    stem = parts[-1].removesuffix(".gguf")
    stem = _GGUF_SHARD_RE.sub("", stem)
    return _GGUF_QUANT_RE.sub("", stem)


_CSS_FILE = Path(__file__).parent / "fleet_body.tcss"

_EDITOR_ID = "#placement-editor"
_SKIPPED_ID = "#placement-skipped"
_TITLE_ID = "#placement-title"
_STATE_ID = "#placement-state"
_COMMANDS_ID = "#placement-commands"
_HINT_WIDGET_ID = "#placement-hint"
_FLEET_PANEL_ID = "#gpu-fleet-panel"

# Only the replicated roles show a replica stepper; the others always serve one.
_REPLICA_ROLES = REPLICATED_ROLES
_HINT = (
    "Toggle a GPU for each role; -/+ sets replicas.  ctrl+r preview · ctrl+s apply · ctrl+x auto"
)
_CMD_PREVIEW = "cmd-preview"
_CMD_APPLY = "cmd-apply"
_CMD_AUTO = "cmd-auto"
# GPUs shown per page in the placement grid; more than this paginate.
_PLACEMENT_PAGE_SIZE = 8
# rerank is a single pinned instance on one card (a small cross-encoder that never
# tensor-splits), so its GPU choice is single-select, unlike the multi-GPU roles.
_SINGLE_ROLES = (WorkerRole.RERANK,)
# The targets a role row offers beside its cards: a role off the cards, or planned for it.
_TARGET_CHOICES = (RoleTarget.CPU, RoleTarget.AUTO)
# Editor row order: the multi-GPU roles first, rerank last -- the single-card odd
# one out sits at the bottom instead of between the replicated roles.
_EDITOR_ROLE_ORDER = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION, WorkerRole.RERANK)


def _role_kind(role: WorkerRole) -> str:
    """How a role occupies GPUs: 'mirror' (a copy per card), 'single' (one pinned
    card), or 'split' (one model tensor-split across cards)."""
    if role in _REPLICA_ROLES:
        return "mirror"
    if role in _SINGLE_ROLES:
        return "single"
    return "split"


class FleetPill(Static, can_focus=True):
    """Focusable one-line pill; Enter / Space / click presses it.

    The editor's toggles, steppers, pager, and command controls are all pills
    (the ``Static, can_focus=True`` + bindings pattern from
    ``widgets/confirm_dialog.py`` / ``model_bar.py::ChatModePill``): state and
    focus ride the fill and text style, so a row costs one line instead of the
    three rows of Button chrome.
    """

    class Pressed(Message):
        """Posted on activation; carries the pill for id-based dispatch."""

        def __init__(self, pill: FleetPill) -> None:
            super().__init__()
            self.pill = pill

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "press", "Press", show=False),
        Binding("space", "press", "Press", show=False),
    ]

    def press(self) -> None:
        """Activate this pill (shared by the mouse and keyboard paths)."""
        self.post_message(self.Pressed(self))

    def action_press(self) -> None:
        self.press()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.press()


@dataclass
class _RoleEdit:
    """Mutable editor state for one role.

    ``tensor_split`` carries a manual split from the loaded view so re-applying an
    unedited placement preserves it (an even split would OOM the smaller of two
    unequal cards). It is cleared the moment the user toggles this role's devices,
    since a split sized for the old card set no longer applies to the new one.
    """

    role: WorkerRole
    model: str
    devices: set[int]
    replicas: int
    tensor_split: tuple[int, ...] | None = None
    target: RoleTarget | None = None
    # True while the role shares a swap group the plan made and the user named no
    # card for it: pinning it to its card would drop the group, so the saved spec
    # leaves the role to the planner.
    planned: bool = False


def _edits_from_view(view: PlacementView) -> dict[WorkerRole, _RoleEdit]:
    """One editor row per role: what the saved spec names, else what the plan runs.

    A refused spec holds the user's choices, so its roles keep their rows and
    targets even when the plan runs something else for them.
    """
    spec_json = view.spec_json or view.rejected_spec_json
    named = PlacementSpec.from_json(spec_json).roles if spec_json else {}
    edits: dict[WorkerRole, _RoleEdit] = {}
    for r in view.roles:
        entry = named.get(r.role)
        if entry is None:
            edits[r.role] = _RoleEdit(
                r.role,
                r.model,
                set(r.devices),
                r.replicas,
                r.tensor_split,
                planned=r.role in view.co_tenants,
            )
        elif view.rejected_spec_json:
            edits[r.role] = _edit_from_entry(r.role, r.model, entry)
        else:
            edits[r.role] = _RoleEdit(
                r.role, r.model, set(r.devices), r.replicas, r.tensor_split, entry.target
            )
    if view.rejected_spec_json:
        for role, entry in named.items():
            if role not in edits:
                edits[role] = _edit_from_entry(role, "", entry)
    return edits


def _edit_from_entry(role: WorkerRole, model: str, entry: RolePlacement) -> _RoleEdit:
    return _RoleEdit(
        role, model, set(entry.devices), entry.replicas, entry.tensor_split, entry.target
    )


class FleetBody(Widget):
    """Live fleet GPU table and interactive placement editor."""

    app: LilbeeApp  # type: ignore[assignment]
    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    applying: reactive[bool] = reactive(False)

    class PlacementReloading(Message):
        """Posted while an apply/clear reloads the fleet, so the chat screen can
        hold submissions until the reload finishes instead of racing it."""

        def __init__(self, active: bool) -> None:
            self.active = active
            super().__init__()

    def __init__(self) -> None:
        super().__init__(id="fleet-body")
        self._edits: dict[WorkerRole, _RoleEdit] = {}
        self._device_indices: tuple[int, ...] = ()
        self._view_manual = False
        self._page = 0
        self._command_actions: dict[str, Callable[[], None]] = {
            _CMD_PREVIEW: self.action_preview,
            _CMD_APPLY: self.action_apply,
            _CMD_AUTO: self.action_clear,
        }

    def watch_applying(self, applying: bool) -> None:
        """Disable the editor controls while an apply/clear is in flight, surface a
        'Rebuilding fleet…' status so the reload isn't a silent idle screen, and tell
        the chat screen to hold submissions until the fleet finishes reloading."""
        self.post_message(self.PlacementReloading(applying))
        with contextlib.suppress(NoMatches):
            self.query_one(_EDITOR_ID, Vertical).disabled = applying
        if applying:
            self._show_rebuilding()

    def _show_rebuilding(self) -> None:
        """Hold a 'Rebuilding fleet…' status until the reloaded fleet reports ready.

        The reload restarts a role and warms its model off-thread (tens of seconds
        for a tensor-split giant); without a positive status the editor is just a
        greyed, silent screen and the user cannot tell the apply took, or whether a
        stale draft still stands. The re-render that ``_change_placement`` runs once
        the fleet is ready (or the change failed) clears this back to the live state.
        """
        with contextlib.suppress(NoMatches):
            state = self.query_one(_STATE_ID, Static)
            state.update(msg.FLEET_STATE_REBUILDING)
            state.set_class(False, "-edited")
            state.set_class(False, "-manual")
            state.set_class(True, "-rebuilding")

    def compose(self) -> ComposeResult:
        with Vertical(id="placement-layout"):
            with Horizontal(id="placement-titlebar"):
                yield Static(msg.FLEET_TITLE, id="placement-title")
                yield Static("", id="placement-state")
                help_icon = Static(msg.FLEET_HELP_ICON, id="placement-help")
                help_icon.tooltip = msg.FLEET_HELP_TOOLTIP
                yield help_icon
            yield GpuFleetPanel()
            yield Static("", id="placement-skipped", markup=False)
            yield Vertical(id="placement-editor")
            with Horizontal(id="placement-commands"):
                yield FleetPill(msg.FLEET_CMD_PREVIEW, id=_CMD_PREVIEW, classes="cmd-pill")
                yield FleetPill(msg.FLEET_CMD_APPLY, id=_CMD_APPLY, classes="cmd-pill")
                yield FleetPill(msg.FLEET_CMD_AUTO, id=_CMD_AUTO, classes="cmd-pill")
            yield Static("", id="placement-engines", markup=False)
            yield Static(_HINT, id="placement-hint")

    def on_mount(self) -> None:
        self._paint_engines()
        self.app.settings_changed_signal.subscribe(self, self._on_settings_changed)
        self._load_worker()

    def _on_settings_changed(self, payload: tuple[str, object]) -> None:
        if payload[0] in ENGINE_SETTING_KEYS:
            self._paint_engines()

    def _paint_engines(self) -> None:
        """Show the engine of each role, and the notice for a role that fell back."""
        views = engine_service.roles()
        notices = list(dict.fromkeys(v.notice for v in views if v.notice))
        self.query_one("#placement-engines", Static).update(
            "\n".join([chips_text(views), *notices])
        )

    @work(thread=True, exit_on_error=False)
    def _load_worker(self) -> None:
        """Fetch placement off the UI thread and populate the widget.

        get_placement resolves the plan, which can spawn a device-probe
        subprocess on a cold cache -- too slow for the event loop.
        """
        try:
            view = get_placement()
        except Exception as exc:
            log.debug("Failed to load placement", exc_info=True)
            call_from_thread(self, self._render_load_failure, str(exc))
            return
        call_from_thread(self, self._render_view, view)

    def _render_load_failure(self, reason: str) -> None:
        """Name the placement-load failure in the panel instead of probing forever."""
        self.query_one(_FLEET_PANEL_ID, GpuFleetPanel).set_probe_failed(reason)
        self.notify(reason, severity="error")

    # -- rendering -------------------------------------------------------

    def _render_view(self, view: PlacementView) -> None:
        """Reset the widget (title, live table, editor) from a resolved placement view."""
        self._view_manual = view.manual
        self._device_indices = tuple(g.index for g in view.gpus)
        edits = _edits_from_view(view)
        # Rows render in _EDITOR_ROLE_ORDER; any role beyond it keeps plan order.
        self._edits = {role: edits.pop(role) for role in _EDITOR_ROLE_ORDER if role in edits}
        self._edits.update(edits)
        self._page = 0
        self._build_editor()
        self._refresh_title(dirty=False)
        self._update_fleet_panel(view)
        self._render_skipped(view)
        if view.co_tenants:
            names = ", ".join(role.value for role in view.co_tenants)
            self.notify(f"Sharing memory, one loaded at a time: {names}", severity="information")
        if view.unplaceable:
            names = ", ".join(role.value for role in view.unplaceable)
            self.notify(f"Does not fit: {names}", severity="warning")
        if view.rejected_spec_json and not view.refusal:
            self.notify(msg.FLEET_SAVED_PLACEMENT_IGNORED, severity="warning")

    def _render_skipped(self, view: PlacementView) -> None:
        """Show a line per role skipped, and why a saved spec is refused."""
        widget = self.query_one(_SKIPPED_ID, Static)
        lines = [
            msg.FLEET_MODEL_NOT_DOWNLOADED.format(
                role=skipped.role.value, model=_clean_model_name(skipped.model)
            )
            for skipped in view.skipped_not_installed
        ]
        lines.extend(
            msg.FLEET_ROLE_WITHOUT_ENGINE.format(role=skipped.role.value, notice=skipped.notice)
            for skipped in view.skipped_no_engine
        )
        if view.refusal:
            lines.append(view.refusal)
        widget.display = bool(lines)
        widget.update("\n".join(lines))

    def _page_devices(self) -> tuple[int, ...]:
        """The GPU indices visible on the current page."""
        start = self._page * _PLACEMENT_PAGE_SIZE
        return self._device_indices[start : start + _PLACEMENT_PAGE_SIZE]

    def _page_count(self) -> int:
        """Number of GPU pages at the current fleet size."""
        n = len(self._device_indices)
        return max(1, (n + _PLACEMENT_PAGE_SIZE - 1) // _PLACEMENT_PAGE_SIZE)

    def _build_editor(self) -> None:
        """Rebuild the placement grid: GPU header, one row per role, optional pager.

        A single-GPU fleet keeps the grid: every role can only live on that card,
        or on the CPU, so the cpu choice is what the drawer offers there.
        """
        container = self.query_one(_EDITOR_ID, Vertical)
        container.remove_children()
        if len(self._device_indices) <= 1:
            container.mount(Label(msg.FLEET_SINGLE_GPU_NOTE, classes="single-gpu-note"))
        devices = self._page_devices()
        widgets: list[Horizontal] = [self._gpu_header_row(devices)]
        for role, edit in self._edits.items():
            kind = _role_kind(role)
            children: list[FleetPill | Label] = [Label(f"{role.value:<7}", classes="role-name")]
            for idx in devices:
                on = " on" if idx in edit.devices else ""
                children.append(
                    FleetPill(
                        f" {idx} ", id=f"dev-{role.value}-{idx}", classes=f"dev-toggle {kind}{on}"
                    )
                )
            for target in _TARGET_CHOICES:
                on = " on" if edit.target is target else ""
                children.append(
                    FleetPill(
                        f" {target.value} ",
                        id=f"tgt-{role.value}-{target.value}",
                        classes=f"dev-toggle {kind}{on}",
                    )
                )
            if role in _REPLICA_ROLES:
                children.append(FleetPill(" - ", id=f"rep-{role.value}-dec", classes="rep-pill"))
                children.append(
                    Label(f"x{edit.replicas}", id=f"repn-{role.value}", classes="rep-count")
                )
                children.append(FleetPill(" + ", id=f"rep-{role.value}-inc", classes="rep-pill"))
            elif role in _SINGLE_ROLES:
                children.append(Label(msg.FLEET_TAG_SINGLE, classes="role-tag"))
            elif len(edit.devices) > 1:
                children.append(Label(msg.FLEET_TAG_SPLIT, classes="role-tag"))
            widgets.append(Horizontal(*children, classes="role-row"))
        container.mount(*widgets)
        if self._page_count() > 1:
            container.mount(self._pager_row())

    def _gpu_header_row(self, devices: tuple[int, ...]) -> Horizontal:
        """A header labelling the visible GPU columns."""
        cells: list[Label] = [Label("GPU", classes="role-name gpu-hdr-lead")]
        cells += [Label(str(idx), classes="gpu-hdr") for idx in devices]
        return Horizontal(*cells, classes="gpu-header-row")

    def _pager_row(self) -> Horizontal:
        """Prev/next controls and a page indicator for fleets past one page."""
        first = self._page * _PLACEMENT_PAGE_SIZE
        last = first + len(self._page_devices()) - 1
        info = f"GPUs {first}-{last}  ·  page {self._page + 1}/{self._page_count()}"
        return Horizontal(
            FleetPill(" ◄ ", id="pg-prev", classes="pg-pill"),
            Label(info, classes="pg-info"),
            FleetPill(" ► ", id="pg-next", classes="pg-pill"),
            classes="pager-row",
        )

    def _refresh_title(self, *, dirty: bool) -> None:
        """Reflect the placement mode in the state segment beside the title pill."""
        state = self.query_one(_STATE_ID, Static)
        if dirty:
            state.update(msg.FLEET_STATE_EDITED)
        else:
            state.update(msg.FLEET_STATE_MANUAL if self._view_manual else msg.FLEET_STATE_AUTO)
        state.set_class(dirty, "-edited")
        state.set_class(not dirty and self._view_manual, "-manual")
        state.set_class(False, "-rebuilding")

    def _update_fleet_panel(self, view: PlacementView) -> None:
        """Push the current device list and roles into the fleet panel."""
        try:
            panel = self.query_one(_FLEET_PANEL_ID, GpuFleetPanel)
        except NoMatches:
            return
        labels = {g.index: g.label for g in view.gpus}
        roles: dict[int, str] = {}
        for r in view.roles:
            short_model = _clean_model_name(r.model) if r.model else ""
            badge = f"{r.role.value} - {short_model}" if short_model else r.role.value
            for idx in r.devices:
                roles[idx] = badge
        panel.set_devices(view.gpus, labels=labels, roles=roles)

    # -- editor state ----------------------------------------------------

    def _spec_from_editor(self) -> PlacementSpec | None:
        """Build a PlacementSpec from the editor; None when nothing is configured."""
        if not self._edits:
            return None
        roles: dict[WorkerRole, RolePlacement] = {}
        for edit in self._edits.values():
            if edit.planned:
                roles[edit.role] = RolePlacement(devices=(), target=RoleTarget.AUTO)
                continue
            if edit.target in _TARGET_CHOICES:
                replicas = edit.replicas if edit.target is RoleTarget.CPU else 1
                roles[edit.role] = RolePlacement(devices=(), replicas=replicas, target=edit.target)
                continue
            if not edit.devices:
                raise PlacementError(f"{edit.role.value} needs at least one GPU")
            devices = tuple(sorted(edit.devices))
            # Keep a loaded manual split only while it still matches the device set;
            # a stale-length split would be rejected by the spec validator.
            split = (
                edit.tensor_split
                if edit.tensor_split and len(edit.tensor_split) == len(devices)
                else None
            )
            roles[edit.role] = RolePlacement(
                devices=devices, replicas=edit.replicas, tensor_split=split, target=edit.target
            )
        return PlacementSpec(roles=roles)

    def _choose_target(self, key: str) -> None:
        """Toggle the cpu or auto choice named by *key* (``role-target``) on its role."""
        role_value, target_value = key.rsplit("-", 1)
        edit = self._edits[WorkerRole(role_value)]
        chosen = RoleTarget(target_value)
        edit.planned = False
        edit.target = None if edit.target is chosen else chosen
        if edit.target is not None:
            edit.devices = set()
            edit.tensor_split = None
        self._build_editor()

    def _toggle_card(self, pill: FleetPill, key: str) -> None:
        """Toggle the card named by *key* (``role-index``) on its role."""
        role_value, idx_str = key.rsplit("-", 1)
        role = WorkerRole(role_value)
        edit = self._edits[role]
        idx = int(idx_str)
        edit.planned = False
        # A card the machine no longer has cannot stay beside the one just picked.
        edit.devices &= set(self._device_indices)
        # Editing the device set invalidates any loaded manual split (its
        # length was sized for the old cards); fall back to a capacity split.
        edit.tensor_split = None
        if edit.target in _TARGET_CHOICES:
            # A card pressed on a cpu or auto role puts the role on that card alone.
            edit.target = None
            edit.devices = {idx}
            self._build_editor()
        elif role in _SINGLE_ROLES:
            # Single pinned instance: the picked card becomes the only one.
            edit.devices = {idx}
            for other in self._page_devices():
                self.query_one(f"#dev-{role.value}-{other}", FleetPill).set_class(
                    other == idx, "on"
                )
        elif idx in edit.devices:
            if len(edit.devices) > 1:  # keep at least one GPU per role
                edit.devices.discard(idx)
                pill.remove_class("on")
        else:
            edit.devices.add(idx)
            pill.add_class("on")

    def on_fleet_pill_pressed(self, event: FleetPill.Pressed) -> None:
        """Handle a GPU toggle, a replica -/+ press, or a command pill."""
        bid = event.pill.id or ""
        command = self._command_actions.get(bid)
        if command is not None:
            command()
            return
        if bid in ("pg-prev", "pg-next"):
            step = -1 if bid == "pg-prev" else 1
            self._page = min(max(0, self._page + step), self._page_count() - 1)
            self._build_editor()
            return
        if bid.startswith("tgt-"):
            self._choose_target(bid.removeprefix("tgt-"))
        elif bid.startswith("dev-"):
            self._toggle_card(event.pill, bid.removeprefix("dev-"))
        elif bid.startswith("rep-"):
            role_value, op = bid.removeprefix("rep-").rsplit("-", 1)
            role = WorkerRole(role_value)
            edit = self._edits[role]
            edit.planned = False
            edit.replicas = max(1, edit.replicas + (1 if op == "inc" else -1))
            self.query_one(f"#repn-{role.value}", Label).update(f"x{edit.replicas}")
        else:
            return
        self._refresh_title(dirty=True)

    # -- actions ---------------------------------------------------------

    def action_preview(self) -> None:
        """Resolve the edited placement against the hardware without applying it."""
        try:
            spec = self._spec_from_editor()
        except PlacementError as exc:
            self.notify(str(exc), severity="error")
            return
        self._preview_worker(spec)

    @work(thread=True, exit_on_error=False)
    def _preview_worker(self, spec: PlacementSpec | None) -> None:
        try:
            view = preview_placement(spec)
            call_from_thread(self, self._render_view, view)
        except Exception as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")

    def action_apply(self) -> None:
        """Apply the edited placement (persists and reloads the fleet)."""
        if self.applying:
            return
        try:
            spec = self._spec_from_editor()
        except PlacementError as exc:
            self.notify(str(exc), severity="error")
            return
        self.applying = True
        self._apply_worker(spec)

    @work(thread=True, exit_on_error=False)
    def _apply_worker(self, spec: PlacementSpec | None) -> None:
        self._change_placement(lambda: set_placement(spec))

    def action_clear(self) -> None:
        """Restore automatic placement."""
        if self.applying:
            return
        self.applying = True
        self._clear_worker()

    @work(thread=True, exit_on_error=False)
    def _clear_worker(self) -> None:
        self._change_placement(lambda: set_placement(None))

    def _change_placement(self, change: Callable[[], PlacementView]) -> None:
        """Apply a placement change off the UI thread, then re-render the live view.

        ``applying`` (and with it the chat screen's submit hold) stays up until the
        restarted chat role actually serves: releasing when the reload returns
        still leaves the model warming, and a prompt sent then races the restart
        instead of getting an answer. The view is re-rendered from the live
        placement whether the change applied or failed, so a rejected or failed
        draft returns to the current read-only state instead of stranding the
        stale 'edited'/won't-fit editor.
        """
        error: str | None = None
        try:
            change()
            wait_chat_ready()
        except Exception as exc:
            error = str(exc)
        try:
            view = get_placement()
            call_from_thread(self, self._render_view, view)
        except Exception as exc:
            error = error or str(exc)
        if error is not None:
            call_from_thread(self, self.notify, error, severity="error")
        call_from_thread(self, setattr, self, "applying", False)
