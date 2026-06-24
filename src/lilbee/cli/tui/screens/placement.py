"""GPU placement screen: inspect and configure multi-GPU model placement.

The editor is interactive: each role has a GPU toggle per device and a replica
stepper, so placement is configured by clicking, not by hand-writing JSON. The
equivalent spec is shown read-only for use with the CLI/HTTP/MCP surfaces.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Label, Static

from lilbee.app.placement import get_placement, preview_placement, set_placement
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from lilbee.app.placement import PlacementView


log = logging.getLogger(__name__)

_GPU_TABLE_ID = "#placement-gpus"
_EDITOR_ID = "#placement-editor"
_GENERATED_ID = "#placement-generated"
_TITLE_ID = "#placement-title"

_GIB = 1024**3
# Only these roles run multiple replicas; the others always serve one instance.
_REPLICA_ROLES = (WorkerRole.EMBED, WorkerRole.VISION)
_HINT = (
    "Toggle a GPU for each role; -/+ sets replicas.  ctrl+r preview · ctrl+s apply · ctrl+x auto"
)


@dataclass
class _RoleEdit:
    """Mutable editor state for one role."""

    role: WorkerRole
    model: str
    devices: set[int]
    replicas: int


def _fmt_gib(n: int) -> str:
    """Format bytes as a GiB string."""
    return f"{n / _GIB:.1f} GiB"


class PlacementScreen(Screen[None]):
    """GPU placement viewer and interactive editor."""

    # Lilbee always hosts screens on a LilbeeApp, so narrowing the type lets
    # the screen call switch_view without per-call type: ignore comments.
    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "placement.tcss"
    AUTO_FOCUS = _GPU_TABLE_ID
    HELP = "Configure GPU placement. ctrl+r preview, ctrl+s apply, ctrl+x auto, q back."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        # priority so they fire even when a button/editor child has focus.
        Binding("ctrl+r", "preview", "Preview", show=True, priority=True),
        Binding("ctrl+s", "apply", "Apply", show=True, priority=True),
        Binding("ctrl+x", "clear", "Auto", show=True, priority=True),
    ]

    applying: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._edits: dict[WorkerRole, _RoleEdit] = {}
        self._device_indices: tuple[int, ...] = ()
        self._gpu_meta: list[tuple[int, str, str, int, int]] = []
        self._view_manual = False

    def watch_applying(self, applying: bool) -> None:
        """Disable the editor controls while an apply/clear is in flight."""
        with contextlib.suppress(NoMatches):
            self.query_one(_EDITOR_ID, Vertical).disabled = applying

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        table: DataTable[str] = DataTable(id="placement-gpus")
        table.cursor_type = "row"

        with TopBars():
            yield ViewTabs()
        with Vertical(id="placement-layout"):
            yield Static("", id="placement-title")
            yield table
            yield Vertical(id="placement-editor")
            yield Static("", id="placement-generated")
            yield Static(_HINT, id="placement-hint")
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(_GPU_TABLE_ID, DataTable)
        table.add_columns("GPU", "Name", "Free", "Total", "Roles")
        self._load_placement()

    def _load_placement(self) -> None:
        """Fetch placement and populate the screen synchronously."""
        try:
            view = get_placement()
        except Exception as exc:
            log.debug("Failed to load placement", exc_info=True)
            self.notify(str(exc), severity="error")
            return
        self._render_view(view)

    # -- rendering -------------------------------------------------------

    def _render_view(self, view: PlacementView) -> None:
        """Reset the screen (title, table, editor) from a resolved placement view."""
        self._view_manual = view.manual
        self._device_indices = tuple(g.index for g in view.gpus)
        self._gpu_meta = [
            (g.index, g.label, g.name, g.free_bytes, g.total_bytes) for g in view.gpus
        ]
        self._edits = {
            r.role: _RoleEdit(r.role, r.model, set(r.devices), r.replicas) for r in view.roles
        }
        self._build_editor()
        self._refresh_table()
        self._refresh_title(dirty=False)
        self._refresh_generated()
        if view.unplaceable:
            names = ", ".join(role.value for role in view.unplaceable)
            self.notify(f"Does not fit: {names}", severity="warning")

    def _build_editor(self) -> None:
        """Mount one interactive row per role (GPU toggles + replica stepper)."""
        container = self.query_one(_EDITOR_ID, Vertical)
        container.remove_children()
        rows: list[Horizontal] = []
        for role, edit in self._edits.items():
            children: list[Button | Label] = [Label(f"{role.value:<7}", classes="role-name")]
            for idx in self._device_indices:
                cls = "dev-toggle on" if idx in edit.devices else "dev-toggle"
                children.append(Button(str(idx), id=f"dev-{role.value}-{idx}", classes=cls))
            if role in _REPLICA_ROLES:
                children.append(Button("-", id=f"rep-{role.value}-dec", classes="rep-btn"))
                children.append(
                    Label(f"x{edit.replicas}", id=f"repn-{role.value}", classes="rep-count")
                )
                children.append(Button("+", id=f"rep-{role.value}-inc", classes="rep-btn"))
            rows.append(Horizontal(*children, classes="role-row"))
        if rows:
            container.mount(*rows)

    def _refresh_table(self) -> None:
        """Repaint the GPU table's Roles column from the current editor state."""
        placed: dict[int, list[str]] = {}
        for edit in self._edits.values():
            for idx in edit.devices:
                placed.setdefault(idx, []).append(edit.role.value)
        table = self.query_one(_GPU_TABLE_ID, DataTable)
        table.clear()
        for idx, label, name, free, total in self._gpu_meta:
            table.add_row(
                label,
                name,
                _fmt_gib(free),
                _fmt_gib(total),
                ", ".join(placed.get(idx, [])) or "-",
                key=str(idx),
            )

    def _refresh_title(self, *, dirty: bool) -> None:
        if dirty:
            text = "Placement (edited; ctrl+s to apply, ctrl+x for auto)"
        else:
            text = "Placement (manual)" if self._view_manual else "Placement (auto)"
        self.query_one(_TITLE_ID, Static).update(f"[bold]{text}[/bold]")

    def _refresh_generated(self) -> None:
        """Show the spec the current controls produce (for CLI/HTTP/MCP parity)."""
        out = self.query_one(_GENERATED_ID, Static)
        try:
            spec = self._spec_from_editor()
        except PlacementError as exc:
            out.update(f"[red]{exc}[/red]")
            return
        out.update(f"equivalent spec:  {spec.to_json() if spec else '(auto)'}")

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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle a GPU toggle or a replica -/+ press."""
        bid = event.button.id or ""
        if bid.startswith("dev-"):
            _, role_value, idx_str = bid.split("-")
            edit = self._edits[WorkerRole(role_value)]
            idx = int(idx_str)
            if idx in edit.devices:
                if len(edit.devices) > 1:  # keep at least one GPU per role
                    edit.devices.discard(idx)
                    event.button.remove_class("on")
            else:
                edit.devices.add(idx)
                event.button.add_class("on")
        elif bid.startswith("rep-"):
            _, role_value, op = bid.split("-")
            role = WorkerRole(role_value)
            edit = self._edits[role]
            edit.replicas = max(1, edit.replicas + (1 if op == "inc" else -1))
            self.query_one(f"#repn-{role.value}", Label).update(f"x{edit.replicas}")
        else:
            return
        self._refresh_table()
        self._refresh_title(dirty=True)
        self._refresh_generated()

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

    def action_go_back(self) -> None:
        """Pop back to the previous screen."""
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        else:
            self.app.switch_view("Chat")
