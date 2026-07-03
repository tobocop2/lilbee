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

from lilbee.app.placement import get_placement, preview_placement, set_placement
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.cli.tui.widgets.gpu_fleet_panel import GpuFleetPanel
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import REPLICATED_ROLES, WorkerRole

if TYPE_CHECKING:
    from lilbee.app.placement import PlacementView

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
_TITLE_ID = "#placement-title"
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
    """Mutable editor state for one role."""

    role: WorkerRole
    model: str
    devices: set[int]
    replicas: int


class FleetBody(Widget):
    """Live fleet GPU table and interactive placement editor."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    applying: reactive[bool] = reactive(False)

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
        """Disable the editor controls while an apply/clear is in flight."""
        with contextlib.suppress(NoMatches):
            self.query_one(_EDITOR_ID, Vertical).disabled = applying

    def compose(self) -> ComposeResult:
        with Vertical(id="placement-layout"):
            with Horizontal(id="placement-titlebar"):
                yield Static("", id="placement-title")
                help_icon = Static(msg.FLEET_HELP_ICON, id="placement-help")
                help_icon.tooltip = msg.FLEET_HELP_TOOLTIP
                yield help_icon
            yield GpuFleetPanel()
            yield Vertical(id="placement-editor")
            with Horizontal(id="placement-commands"):
                yield FleetPill(msg.FLEET_CMD_PREVIEW, id=_CMD_PREVIEW, classes="cmd-pill")
                yield FleetPill(msg.FLEET_CMD_APPLY, id=_CMD_APPLY, classes="cmd-pill")
                yield FleetPill(msg.FLEET_CMD_AUTO, id=_CMD_AUTO, classes="cmd-pill")
            yield Static(_HINT, id="placement-hint")

    def on_mount(self) -> None:
        self._load_worker()

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
            call_from_thread(self, self.notify, str(exc), severity="error")
            return
        call_from_thread(self, self._render_view, view)

    # -- rendering -------------------------------------------------------

    def _render_view(self, view: PlacementView) -> None:
        """Reset the widget (title, live table, editor) from a resolved placement view."""
        self._view_manual = view.manual
        self._device_indices = tuple(g.index for g in view.gpus)
        edits = {r.role: _RoleEdit(r.role, r.model, set(r.devices), r.replicas) for r in view.roles}
        # Rows render in _EDITOR_ROLE_ORDER; any role beyond it keeps plan order.
        self._edits = {role: edits.pop(role) for role in _EDITOR_ROLE_ORDER if role in edits}
        self._edits.update(edits)
        self._page = 0
        self._build_editor()
        self._refresh_title(dirty=False)
        self._update_fleet_panel(view)
        if view.unplaceable:
            names = ", ".join(role.value for role in view.unplaceable)
            self.notify(f"Does not fit: {names}", severity="warning")

    def _page_devices(self) -> tuple[int, ...]:
        """The GPU indices visible on the current page."""
        start = self._page * _PLACEMENT_PAGE_SIZE
        return self._device_indices[start : start + _PLACEMENT_PAGE_SIZE]

    def _page_count(self) -> int:
        """Number of GPU pages at the current fleet size."""
        n = len(self._device_indices)
        return max(1, (n + _PLACEMENT_PAGE_SIZE - 1) // _PLACEMENT_PAGE_SIZE)

    def _build_editor(self) -> None:
        """Rebuild the placement grid: GPU header, one row per role, optional pager."""
        container = self.query_one(_EDITOR_ID, Vertical)
        container.remove_children()
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
        if dirty:
            text = "Placement (edited; ctrl+s to apply, ctrl+x for auto)"
        else:
            text = "Placement (manual)" if self._view_manual else "Placement (auto)"
        self.query_one(_TITLE_ID, Static).update(f"[bold]{text}[/bold]")

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
            if not edit.devices:
                raise PlacementError(f"{edit.role.value} needs at least one GPU")
            roles[edit.role] = RolePlacement(
                devices=tuple(sorted(edit.devices)), replicas=edit.replicas
            )
        return PlacementSpec(roles=roles)

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
        if bid.startswith("dev-"):
            _, role_value, idx_str = bid.split("-")
            role = WorkerRole(role_value)
            edit = self._edits[role]
            idx = int(idx_str)
            if role in _SINGLE_ROLES:
                # Single pinned instance: the picked card becomes the only one.
                edit.devices = {idx}
                for other in self._page_devices():
                    self.query_one(f"#dev-{role.value}-{other}", FleetPill).set_class(
                        other == idx, "on"
                    )
            elif idx in edit.devices:
                if len(edit.devices) > 1:  # keep at least one GPU per role
                    edit.devices.discard(idx)
                    event.pill.remove_class("on")
            else:
                edit.devices.add(idx)
                event.pill.add_class("on")
        elif bid.startswith("rep-"):
            _, role_value, op = bid.split("-")
            role = WorkerRole(role_value)
            edit = self._edits[role]
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
        try:
            set_placement(spec)
            view = get_placement()
            call_from_thread(self, self._render_view, view)
        except Exception as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")
        finally:
            call_from_thread(self, setattr, self, "applying", False)

    def action_clear(self) -> None:
        """Restore automatic placement."""
        if self.applying:
            return
        self.applying = True
        self._clear_worker()

    @work(thread=True, exit_on_error=False)
    def _clear_worker(self) -> None:
        try:
            set_placement(None)
            view = get_placement()
            call_from_thread(self, self._render_view, view)
        except Exception as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")
        finally:
            call_from_thread(self, setattr, self, "applying", False)
