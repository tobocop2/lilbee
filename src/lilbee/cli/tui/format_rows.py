"""The rows of the install format picker: size, fit and engine for each format of a model.

Building the rows reads the Hub for a safetensors model, so run it off the UI thread.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from textual.content import Content

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_mark import engine_mark, format_size
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.runtime.hardware import FitLevel, compute_fit

_BYTES_PER_GB = 1024**3
# Formats whose memory need is worked out from the Hub header, for a chat model.
_SIZED_FROM_HUB = frozenset({ModelFormat.SAFETENSORS, ModelFormat.MLX})
_FIT_TEXT: dict[FitLevel, tuple[str, str]] = {
    FitLevel.FITS: (msg.FORMAT_FIT_FITS, "$success"),
    FitLevel.TIGHT: (msg.FORMAT_FIT_TIGHT, "$warning"),
    FitLevel.WONT_RUN: (msg.FORMAT_FIT_WONT_RUN, "$error"),
}


@dataclass(frozen=True)
class FormatRow:
    """One format of a model: what it costs, whether it fits, and the engine that runs it.

    ``missing`` is the engine to install first, or ``None`` when the engine is installed
    or no engine loads the format.
    """

    fmt: ModelFormat
    default: bool
    engine: str | None
    size_bytes: int | None
    fit: FitLevel | None
    missing: EngineOption | None = None


def _engine_for(ref: str) -> tuple[str | None, EngineOption | None]:
    """The engine that runs *ref*, and the option to install when that engine is missing."""
    auto, *options = engine_service.engine_options(ref)
    if auto.auto_now is not None:
        return auto.auto_now, None
    pick = next((o for o in options if o.installed), options[0] if options else None)
    if pick is None:
        return None, None
    return pick.name, None if pick.installed else pick


def _size_bytes(model: CatalogModel, fmt: ModelFormat, engine: str | None) -> int | None:
    """The memory *fmt* of *model* needs, or ``None`` when no figure is known."""
    if fmt in _SIZED_FROM_HUB and model.task is ModelTask.CHAT and engine is not None:
        # heavy: lilbee.catalog and lilbee.providers.fleet (>50ms)
        from lilbee.catalog.hf_client import hf_token
        from lilbee.catalog.pull_fit import chat_pull_fit

        fit = chat_pull_fit(model.hf_repo, choice.ADAPTERS[EngineName(engine)], token=hf_token())
        return fit.device_bytes if fit.size_known else None
    if fmt is model.format and model.size_gb > 0:
        return int(model.size_gb * _BYTES_PER_GB)
    return None


def format_rows(model: CatalogModel, available_bytes: int | None) -> list[FormatRow]:
    """The rows for every format of *model*, the default format first, the rest in order."""
    ordered = [model.format, *sorted(model.formats - {model.format})]
    rows: list[FormatRow] = []
    for fmt in ordered:
        engine, missing = _engine_for(replace(model, format=fmt).pull_ref)
        size = _size_bytes(model, fmt, engine)
        fit = None
        if size is not None and available_bytes is not None:
            fit = compute_fit(size, available_bytes).level
        rows.append(FormatRow(fmt, fmt is model.format, engine, size, fit, missing))
    return rows


def format_prompt(row: FormatRow) -> Content:
    """The picker line of *row*: format, size, fit, engine, and the engine download when due."""
    name = Content.assemble(row.fmt.value)
    if row.default:
        name = Content.assemble(name, " ", Content.styled(msg.FORMAT_ROW_DEFAULT, "$text-muted"))
    parts = [name]
    parts.append(
        Content(format_size(row.size_bytes))
        if row.size_bytes is not None
        else Content(msg.FORMAT_SIZE_UNKNOWN)
    )
    if row.fit is not None:
        text, color = _FIT_TEXT[row.fit]
        parts.append(Content.styled(text, color))
    parts.append(
        engine_mark(row.engine) if row.engine is not None else Content(msg.FORMAT_NO_ENGINE)
    )
    if row.missing is not None:
        size = row.missing.download_bytes
        text = (
            msg.FORMAT_ENGINE_DOWNLOAD.format(size=format_size(size))
            if size is not None
            else msg.FORMAT_ENGINE_DOWNLOAD_NO_SIZE
        )
        parts.append(Content.styled(text, "$text-muted"))
    return Content(msg.FORMAT_ROW_SEPARATOR).join(parts)
