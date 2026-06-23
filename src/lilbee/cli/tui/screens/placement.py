"""GPU placement screen: inspect, preview, and apply manual placement specs."""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static, TextArea

from lilbee.app.placement import get_placement, preview_placement, set_placement
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.providers.fleet.placement_spec import PlacementSpec

if TYPE_CHECKING:
    from lilbee.app.placement import PlacementView


log = logging.getLogger(__name__)

_GPU_TABLE_ID = "#placement-gpus"
_ROLE_SUMMARY_ID = "#placement-role-summary"
_SPEC_AREA_ID = "#placement-spec"

_GIB = 1024**3


def _fmt_gib(n: int) -> str:
    """Format bytes as GiB string."""
    return f"{n / _GIB:.1f} GiB"


def _render_roles(view: PlacementView) -> str:
    """Build the role summary text from a PlacementView."""
    lines: list[str] = []
    for r in view.roles:
        devs = ", ".join(str(d) for d in r.devices)
        ts = str(r.tensor_split) if r.tensor_split else "auto"
        lines.append(
            f"[bold]{r.role.value}[/bold]  model={r.model}  "
            f"devices=[{devs}]  tensor_split={ts}  replicas={r.replicas}"
        )
    for role in view.unplaceable:
        lines.append(f"[red]UNPLACEABLE: {role.value}[/red]")
    return "\n".join(lines) if lines else "(no roles placed)"


class PlacementScreen(Screen[None]):
    """GPU placement viewer and editor."""

    # Lilbee always hosts screens on a LilbeeApp, so narrowing the type lets
    # the screen call switch_view without per-call type: ignore comments.
    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "placement.tcss"
    AUTO_FOCUS = _GPU_TABLE_ID
    HELP = "Inspect GPU placement. ctrl+r preview, ctrl+s apply, ctrl+x clear, q back."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        Binding("ctrl+r", "preview", "Preview", show=True),
        Binding("ctrl+s", "apply", "Apply", show=True),
        Binding("ctrl+x", "clear", "Clear", show=True),
    ]

    applying: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._spec_text: str = ""

    def watch_applying(self, applying: bool) -> None:
        """Disable the spec editor while an apply/clear is in flight."""
        with contextlib.suppress(NoMatches):
            self.query_one(_SPEC_AREA_ID, TextArea).disabled = applying

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
            yield table
            yield Static("", id="placement-role-summary")
            yield TextArea(id="placement-spec")
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(_GPU_TABLE_ID, DataTable)
        table.add_columns("Label", "Name", "Free", "Total")
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

    def _render_view(self, view: PlacementView) -> None:
        """Populate the GPU table and role summary from a PlacementView."""
        table = self.query_one(_GPU_TABLE_ID, DataTable)
        table.clear()
        for gpu in view.gpus:
            table.add_row(
                gpu.label,
                gpu.name,
                _fmt_gib(gpu.free_bytes),
                _fmt_gib(gpu.total_bytes),
                key=str(gpu.index),
            )
        summary = self.query_one(_ROLE_SUMMARY_ID, Static)
        summary.update(_render_roles(view))
        if view.spec_json:
            self._spec_text = view.spec_json
            self.query_one(_SPEC_AREA_ID, TextArea).load_text(view.spec_json)

    def _parse_spec(self) -> PlacementSpec | None:
        """Parse the current spec text into a PlacementSpec or None."""
        raw = self._spec_text.strip()
        if not raw:
            return None
        return PlacementSpec.from_json(raw)

    def action_preview(self) -> None:
        """Preview the current spec without applying it."""
        try:
            spec = self._parse_spec()
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            self.notify(str(exc), severity="error")
            return
        self._preview_worker(spec)

    @work(thread=True, exit_on_error=False)
    def _preview_worker(self, spec: PlacementSpec | None) -> None:
        """Run preview_placement off the UI thread."""
        try:
            view = preview_placement(spec)
            call_from_thread(self, self._render_view, view)
        except Exception as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")

    def action_apply(self) -> None:
        """Apply the current spec and reload services."""
        if self.applying:
            return
        try:
            spec = self._parse_spec()
        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            self.notify(str(exc), severity="error")
            return
        self.applying = True
        self._apply_worker(spec)

    @work(thread=True, exit_on_error=False)
    def _apply_worker(self, spec: PlacementSpec | None) -> None:
        """Run set_placement off the UI thread."""
        try:
            set_placement(spec)
            view = get_placement()
            call_from_thread(self, self._render_view, view)
        except Exception as exc:
            call_from_thread(self, self.notify, str(exc), severity="error")
        finally:
            call_from_thread(self, setattr, self, "applying", False)

    def action_clear(self) -> None:
        """Clear the manual spec (restore auto placement)."""
        if self.applying:
            return
        self.applying = True
        self._clear_worker()

    @work(thread=True, exit_on_error=False)
    def _clear_worker(self) -> None:
        """Run set_placement(None) off the UI thread."""
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
