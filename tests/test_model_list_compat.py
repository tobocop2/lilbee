"""List view renders fit and compat indicators on the headline."""

from __future__ import annotations

import pytest

from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.model_list import (
    _compat_tag,
    _fit_tag,
    _render_local_headline,
)
from lilbee.runtime.hardware import FitChip, FitLevel


def _row(
    *,
    compat: ModelCompat = ModelCompat.SUPPORTED,
    fit: FitChip | None = None,
    installed: bool = False,
    featured: bool = False,
) -> LocalCatalogRow:
    return LocalCatalogRow(
        name="Foo",
        task="chat",
        params="",
        size="",
        quant="",
        downloads="",
        featured=featured,
        installed=installed,
        sort_downloads=0,
        sort_size=0,
        ref="acme/foo-GGUF",
        backend="native",
        compat=compat,
        fit=fit,
    )


def _fit_chip(level: FitLevel) -> FitChip:
    return FitChip(level=level, headroom_gb=1.0)


@pytest.mark.parametrize(
    "level,expected_text",
    [
        (FitLevel.FITS, "fits"),
        (FitLevel.TIGHT, "tight"),
        (FitLevel.WONT_RUN, "won't run"),
    ],
)
def test_fit_tag_renders_level(level: FitLevel, expected_text: str) -> None:
    parts = _fit_tag(_fit_chip(level))
    assert len(parts) == 1
    assert expected_text in parts[0].plain


def test_fit_tag_empty_when_no_fit_chip() -> None:
    assert _fit_tag(None) == []


def test_compat_tag_empty_for_supported() -> None:
    assert _compat_tag(ModelCompat.SUPPORTED) == []


def test_compat_tag_unsupported_text() -> None:
    parts = _compat_tag(ModelCompat.UNSUPPORTED)
    assert len(parts) == 1
    assert msg.COMPAT_PILL_UNSUPPORTED in parts[0].plain


def test_compat_tag_unknown_text() -> None:
    parts = _compat_tag(ModelCompat.UNKNOWN)
    assert len(parts) == 1
    assert msg.COMPAT_PILL_UNKNOWN in parts[0].plain


def test_headline_includes_fit_and_compat_for_unsupported_wont_run() -> None:
    headline = _render_local_headline(
        _row(compat=ModelCompat.UNSUPPORTED, fit=_fit_chip(FitLevel.WONT_RUN))
    )
    joined = "".join(p.plain for p in headline)
    assert "won't run" in joined
    assert msg.COMPAT_PILL_UNSUPPORTED in joined


def test_headline_omits_compat_for_supported_row() -> None:
    headline = _render_local_headline(
        _row(compat=ModelCompat.SUPPORTED, fit=_fit_chip(FitLevel.FITS))
    )
    joined = "".join(p.plain for p in headline)
    assert "fits" in joined
    assert msg.COMPAT_PILL_UNSUPPORTED not in joined
    assert msg.COMPAT_PILL_UNKNOWN not in joined


def test_headline_includes_unknown_compat_for_featured_row() -> None:
    headline = _render_local_headline(_row(compat=ModelCompat.UNKNOWN, featured=True))
    joined = "".join(p.plain for p in headline)
    assert msg.COMPAT_PILL_UNKNOWN in joined
