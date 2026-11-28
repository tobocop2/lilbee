"""Modal showing the catalog row's HuggingFace metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Markdown, Static

from lilbee.app import engine_service
from lilbee.app.engine_choice import engine_for_ref
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_install_flow import choose_for_model
from lilbee.cli.tui.engine_mark import engine_mark
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.screens.engine_picker import (
    EnginePickerModal,
    engine_picker_open,
    option_detail,
)
from lilbee.cli.tui.widgets.dialog import CancelButton, Dialog, DialogFooter
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines.choice import AUTO, NoEngineError

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

_SEPARATOR = ", "


def _current_engine(ref: str) -> str | None:
    """The engine a run of *ref* would use now, or None when no engine can load it."""
    try:
        return engine_for_ref(ref, cfg).name
    except NoEngineError:
        return None


def engine_section(ref: str) -> Content:
    """The Engine section of model info: the current engine, then the others that load *ref*."""
    heading = Content.styled(msg.MODEL_INFO_ENGINE_HEADING, "bold")
    try:
        current = engine_for_ref(ref, cfg).name
    except NoEngineError as exc:
        return Content.assemble(heading, "\n", str(exc))
    options = [o for o in engine_service.engine_options(ref) if o.name != AUTO]
    lines = [heading]
    lines.extend(
        Content.assemble(engine_mark(o.name), "  ", option_detail(o))
        for o in options
        if o.name == current
    )
    others = [
        Content.assemble(engine_mark(o.name), " ", option_detail(o))
        for o in options
        if o.name != current
    ]
    if others:
        label = Content.styled(msg.MODEL_INFO_ENGINE_OTHERS + " ", "$text-muted")
        lines.append(Content.assemble(label, Content(_SEPARATOR).join(others)))
    return Content("\n").join(lines)


class ModelInfoModal(Dialog[None]):
    """Read-only modal showing what we know about one catalog model."""

    CSS_PATH: ClassVar[str] = "model_info.tcss"

    app: LilbeeApp  # type: ignore[assignment]

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", "Close", show=True),
        Binding("q", "dismiss(None)", "Close", show=False),
        Binding("i", "dismiss(None)", "Close", show=False),
        Binding("e", "pick_engine", "Engine", show=False),
    ]

    def __init__(self, row: LocalCatalogRow) -> None:
        super().__init__()
        self._row = row

    def compose(self) -> ComposeResult:
        head, tail = self._markdown_parts()
        with Vertical(id="info-root"):
            yield Static(self._row.name, id="info-title", markup=False)
            yield Static(self._row.ref, id="info-ref", markup=False)
            with VerticalScroll(id="info-body"):
                yield Markdown(head, id="info-md")
                yield Static(engine_section(self._row.ref), id="info-engine")
                yield Markdown(tail, id="info-md-tail")
            with DialogFooter():
                yield Static(msg.MODEL_INFO_HINT, id="info-hint")
                yield CancelButton(msg.DIALOG_CLOSE_LABEL)

    def _markdown_parts(self) -> tuple[str, str]:
        """The fields before the Engine section (up to Status) and the fields after it."""
        row = self._row
        head: list[str] = []
        cm = row.catalog_model
        if cm is not None and cm.description:
            head.append(cm.description)
            head.append("")
        head.append("**Task:** " + (row.task or "-"))
        if row.params:
            head.append("**Parameters:** " + row.params)
        if row.size:
            head.append("**Download size:** " + row.size)
        if cm is not None and cm.min_ram_gb:
            head.append(f"**Recommended RAM:** {cm.min_ram_gb:g} GB")
        if row.quant:
            head.append("**Quantization:** " + row.quant)
        if row.downloads:
            head.append("**Downloads:** " + row.downloads)
        if row.installed:
            head.append("**Status:** installed")
        tail: list[str] = []
        if cm is not None and cm.gguf_filename and cm.gguf_filename != cm.hf_repo:
            tail.append("**GGUF file:** `" + cm.gguf_filename + "`")
        tail.append("")
        tail.append(msg.MODEL_INFO_HF_LINK.format(repo=row.ref))
        return "\n".join(head), "\n".join(tail)

    def _build_markdown(self) -> str:
        head, tail = self._markdown_parts()
        return head + "\n" + tail

    def action_pick_engine(self) -> None:
        """Open the engine picker for this model."""
        if engine_picker_open(self.app):
            return
        ref = self._row.ref
        options = engine_service.engine_options(ref)
        if len(options) <= 1:
            self.notify(msg.MODEL_INFO_NO_ENGINE_CHOICE, severity="warning")
            return
        modal = EnginePickerModal(
            msg.MODEL_INFO_PICKER_TITLE,
            ref,
            ref,
            _current_engine(ref),
            options,
        )
        self.app.push_screen(modal, self._on_engine_picked)

    def _on_engine_picked(self, option: engine_service.EngineOption | None) -> None:
        if option is None:
            return
        choose_for_model(self.app, self._row.ref, option, self._repaint_engine)

    def _repaint_engine(self) -> None:
        if not self.is_attached:
            return
        self.query_one("#info-engine", Static).update(engine_section(self._row.ref))
