"""Unit tests for the compat pill rendering in ModelCard."""

from __future__ import annotations

import pytest

from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui.messages import COMPAT_PILL_UNKNOWN, COMPAT_PILL_UNSUPPORTED
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.model_card import _compat_pill, _render_local


def _row(compat: ModelCompat) -> LocalCatalogRow:
    return LocalCatalogRow(
        name="Foo Model",
        task="chat",
        params="0.6B",
        size="0.5 GB",
        quant="Q4",
        downloads="10",
        featured=False,
        installed=False,
        sort_downloads=10,
        sort_size=0.5,
        ref="acme/foo-GGUF",
        backend="native",
        compat=compat,
    )


def test_supported_has_no_compat_pill() -> None:
    assert _compat_pill(ModelCompat.SUPPORTED) is None


def test_unsupported_pill_text() -> None:
    out = _compat_pill(ModelCompat.UNSUPPORTED)
    assert out is not None
    assert COMPAT_PILL_UNSUPPORTED in out.plain


def test_unknown_pill_text() -> None:
    out = _compat_pill(ModelCompat.UNKNOWN)
    assert out is not None
    assert COMPAT_PILL_UNKNOWN in out.plain


@pytest.mark.parametrize(
    "compat,expected",
    [
        (ModelCompat.UNSUPPORTED, COMPAT_PILL_UNSUPPORTED),
        (ModelCompat.UNKNOWN, COMPAT_PILL_UNKNOWN),
    ],
)
def test_render_local_includes_compat_pill_for_non_supported(
    compat: ModelCompat, expected: str
) -> None:
    rendered = _render_local(_row(compat), selected=False)
    assert expected in rendered.plain


def test_render_local_omits_compat_pill_for_supported() -> None:
    rendered = _render_local(_row(ModelCompat.SUPPORTED), selected=False)
    assert COMPAT_PILL_UNSUPPORTED not in rendered.plain
    assert COMPAT_PILL_UNKNOWN not in rendered.plain
