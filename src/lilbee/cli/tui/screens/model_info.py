"""Modal showing the catalog row's HuggingFace metadata."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from lilbee.app.engine_choice import engine_for_ref
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines.choice import NoEngineError


def _engine_for(ref: str) -> str:
    """The engine a run of *ref* would use now, or the text that says none can load it."""
    try:
        return engine_for_ref(ref, cfg).name
    except NoEngineError as exc:
        return str(exc)


class ModelInfoModal(ModalScreen[None]):
    """Read-only modal showing what we know about one catalog model."""

    CSS_PATH: ClassVar[str] = "model_info.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(None)", "Close", show=True),
        Binding("q", "dismiss(None)", "Close", show=False),
        Binding("i", "dismiss(None)", "Close", show=False),
    ]

    def __init__(self, row: LocalCatalogRow) -> None:
        super().__init__()
        self._row = row

    def compose(self) -> ComposeResult:
        with Vertical(id="info-root"):
            yield Static(self._row.name, id="info-title", markup=False)
            yield Static(self._row.ref, id="info-ref", markup=False)
            with VerticalScroll(id="info-body"):
                yield Markdown(self._build_markdown(), id="info-md")
            yield Static(msg.MODEL_INFO_HINT, id="info-hint")

    def _build_markdown(self) -> str:
        row = self._row
        lines: list[str] = []
        cm = row.catalog_model
        if cm is not None and cm.description:
            lines.append(cm.description)
            lines.append("")
        lines.append("**Task:** " + (row.task or "-"))
        if row.params:
            lines.append("**Parameters:** " + row.params)
        if row.size:
            lines.append("**Download size:** " + row.size)
        if cm is not None and cm.min_ram_gb:
            lines.append(f"**Recommended RAM:** {cm.min_ram_gb:g} GB")
        if row.quant:
            lines.append("**Quantization:** " + row.quant)
        if row.downloads:
            lines.append("**Downloads:** " + row.downloads)
        if row.installed:
            lines.append("**Status:** installed")
        lines.append(msg.MODEL_INFO_ENGINE.format(engine=_engine_for(row.ref)))
        if cm is not None and cm.gguf_filename and cm.gguf_filename != cm.hf_repo:
            lines.append("**GGUF file:** `" + cm.gguf_filename + "`")
        lines.append("")
        lines.append(msg.MODEL_INFO_HF_LINK.format(repo=row.ref))
        return "\n".join(lines)
