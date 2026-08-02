"""Unit tests for the compat pill rendering in ModelCard."""

from __future__ import annotations

import pytest

from lilbee.catalog.types import ModelCompat, ModelTask
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


@pytest.mark.parametrize(
    "compat,expected",
    [
        (ModelCompat.UNSUPPORTED, COMPAT_PILL_UNSUPPORTED),
        (ModelCompat.UNKNOWN, COMPAT_PILL_UNKNOWN),
    ],
)
def test_grid_card_lines_include_compat_pill(compat: ModelCompat, expected: str) -> None:
    """The grid view's _local_lines path also renders the compat chip."""
    from lilbee.cli.tui.widgets.model_grid import _local_lines

    lines = _local_lines(_row(compat), selected=False)
    joined = "\n".join(line.plain for line in lines)
    assert expected in joined


def test_grid_card_lines_omit_compat_pill_for_supported() -> None:
    from lilbee.cli.tui.widgets.model_grid import _local_lines

    lines = _local_lines(_row(ModelCompat.SUPPORTED), selected=False)
    joined = "\n".join(line.plain for line in lines)
    assert COMPAT_PILL_UNSUPPORTED not in joined
    assert COMPAT_PILL_UNKNOWN not in joined


def test_unknown_pill_text_is_self_explanatory() -> None:
    """A bare '?' pill explains nothing; the copy must name the state."""
    assert COMPAT_PILL_UNKNOWN == "untested"


def test_catalog_to_row_marks_installed_rows_supported() -> None:
    """An installed model demonstrably runs, whatever the catalog probe said."""
    from lilbee.catalog.models import CatalogModel
    from lilbee.cli.tui.screens.catalog_utils import catalog_to_row

    model = CatalogModel(
        hf_repo="acme/foo-GGUF",
        gguf_filename="foo-Q8_0.gguf",
        size_gb=0.5,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        compat=ModelCompat.UNKNOWN,
    )
    assert catalog_to_row(model, installed=True).compat is ModelCompat.SUPPORTED
    assert catalog_to_row(model, installed=False).compat is ModelCompat.UNKNOWN


def test_installed_registry_row_is_supported() -> None:
    """Wizard/library rows built from an installed registry ref carry SUPPORTED."""
    from lilbee.cli.tui.screens.setup import _installed_name_to_row

    row = _installed_name_to_row("acme/foo-GGUF/foo-Q8_0.gguf", "chat")
    assert row.compat is ModelCompat.SUPPORTED


def test_remote_row_is_supported() -> None:
    """Rows for models a live local server reports are running are SUPPORTED."""
    from lilbee.cli.tui.screens.catalog_utils import remote_to_row
    from lilbee.modelhub.model_manager import RemoteModel

    rm = RemoteModel(
        name="llama3:latest",
        task=ModelTask.CHAT,
        family="llama",
        parameter_size="8B",
        provider="Ollama",
    )
    assert remote_to_row(rm).compat is ModelCompat.SUPPORTED


def test_variant_row_is_supported() -> None:
    """Family rows come from the curated featured catalog, so they are SUPPORTED."""
    from lilbee.catalog import get_families
    from lilbee.cli.tui.screens.catalog_utils import variant_to_row

    family = get_families()[0]
    row = variant_to_row(family.variants[0], family, installed=False)
    assert row.compat is ModelCompat.SUPPORTED


def test_native_backend_pill_is_dropped() -> None:
    """The implied 'native' backend pill is not rendered (parity with grid/list)."""
    out = _render_local(_row(ModelCompat.SUPPORTED), selected=False)
    assert "native" not in out.plain


def test_non_native_backend_pill_is_shown() -> None:
    row = _row(ModelCompat.SUPPORTED)
    row.backend = "Ollama"
    out = _render_local(row, selected=False)
    assert "Ollama" in out.plain


class TestWorstCaseCardLayout:
    """The widest a card can get: unsupported architecture that also won't fit."""

    @staticmethod
    def _worst_case_row():
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
        from lilbee.runtime.hardware import FitChip, FitLevel

        row = LocalCatalogRow(
            name="DavidAU/Qwen3.6-27B-Fable-Fusion-Uncensored-Heretic-NEO-MAX-MTP",
            task="chat",
            params="405B",
            size="431.1 GB",
            quant="Q4_K_M",
            downloads="4.6M",
            featured=True,
            installed=True,
            sort_downloads=4_600_000,
            sort_size=431.1,
            ref="a/b-GGUF",
            compat=ModelCompat.UNSUPPORTED,
        )
        row.fit = FitChip(level=FitLevel.WONT_RUN, headroom_gb=-646.0)
        return row

    def test_both_chips_render_on_the_secondary_line(self) -> None:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.model_grid import _local_lines

        lines = _local_lines(self._worst_case_row(), selected=True)
        secondary = lines[2].plain
        assert msg.COMPAT_PILL_UNSUPPORTED in secondary
        assert "won't run" in secondary

    def test_it_stays_inside_the_card_body_budget(self) -> None:
        """Overflowing the body pushes every card border below it out of line."""
        from lilbee.cli.tui.widgets.model_grid import _CARD_BODY_HEIGHT, _local_lines

        lines = _local_lines(self._worst_case_row(), selected=True)
        assert len(lines) <= _CARD_BODY_HEIGHT

    def test_no_line_exceeds_the_narrowest_grid_column(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _local_lines

        narrowest_column = 30  # GridSelect(min_column_width=30) on every catalog grid
        lines = _local_lines(self._worst_case_row(), selected=True)
        assert max(len(line.plain) for line in lines) <= narrowest_column
