"""Unit tests for the compat pill rendering in ModelCard."""

from __future__ import annotations

import pytest

from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.cli.tui.messages import COMPAT_PILL_UNKNOWN, COMPAT_PILL_UNSUPPORTED
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.catalog_card_shared import _compat_pill
from lilbee.cli.tui.widgets.model_card import _render_local


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


class TestSizeVariantStripFitsTheCard:
    """A family card's quant strip must never push the card border out of line."""

    @staticmethod
    def _variants(n: int):
        from lilbee.cli.tui.screens.catalog_utils import SizeVariant

        # Colliding quants force the long per-variant label, the widest form.
        return [
            SizeVariant(label=f"{27 + i}B Q4_K_M", quant="Q4_K_M", size_gb=15.0 + i, ref=f"a/{i}")
            for i in range(n)
        ]

    def test_it_never_exceeds_the_width_it_is_given(self) -> None:
        from lilbee.cli.tui.widgets.catalog_card_shared import _build_size_variant_strip

        for count in (2, 5, 12):
            for width in (20, 27, 40):
                strip = _build_size_variant_strip(self._variants(count), width)
                assert strip.cell_length <= width, f"{count} variants at width {width}"

    def test_dropped_chips_are_counted(self) -> None:
        from lilbee.cli.tui.widgets.catalog_card_shared import _build_size_variant_strip

        strip = _build_size_variant_strip(self._variants(5), 27)
        assert "+" in strip.plain

    def test_duplicate_labels_collapse(self) -> None:
        """Two repos in one family can share param count and quant."""
        from lilbee.cli.tui.screens.catalog_utils import SizeVariant
        from lilbee.cli.tui.widgets.catalog_card_shared import _build_size_variant_strip

        dupes = [
            SizeVariant(label="27B Q4_K_M", quant="Q4_K_M", size_gb=15.7, ref="a/1"),
            SizeVariant(label="27B Q4_K_M", quant="Q4_K_M", size_gb=15.8, ref="a/2"),
        ]
        assert _build_size_variant_strip(dupes, 60).plain.count("27B Q4_K_M") == 1

    def test_pad_line_truncates_an_over_wide_line(self) -> None:
        """The frame holds by construction, whatever a line builder produces."""
        from textual.content import Content

        from lilbee.cli.tui.widgets.model_grid import _pad_line

        assert _pad_line(Content("x" * 80), 27).cell_length == 27

    def test_a_family_card_row_stays_within_the_narrowest_column(self) -> None:
        from lilbee.catalog.types import ModelCompat
        from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
        from lilbee.cli.tui.widgets.model_grid import _local_lines
        from lilbee.runtime.hardware import FitChip, FitLevel

        row = LocalCatalogRow(
            name="Qwen3.6 27B",
            task="vision",
            params="27B",
            size="15.4 GB",
            quant="Q4_K_M",
            downloads="1M",
            featured=True,
            installed=False,
            sort_downloads=1,
            sort_size=15.4,
            ref="a/1",
            compat=ModelCompat.SUPPORTED,
        )
        row.fit = FitChip(level=FitLevel.FITS, headroom_gb=10.0)
        row.size_variants = self._variants(5)
        body_width = 27
        lines = _local_lines(row, selected=True, body_width=body_width)
        assert max(line.cell_length for line in lines) <= body_width


def test_watch_selected_before_compose_is_a_no_op() -> None:
    """A highlight that lands before the card composes must not crash the watch."""
    from lilbee.cli.tui.widgets.model_card import ModelCard

    card = ModelCard(_row(ModelCompat.SUPPORTED))
    card.watch_selected(True)  # un-mounted: no .card-body to update yet


async def test_watch_selected_rerenders_a_mounted_card() -> None:
    """Moving the highlight onto a mounted card repaints its body with the hint."""
    from textual.widgets import Static

    from lilbee.cli.tui.widgets.model_card import ModelCard
    from tests._lilbee_app_test_host import LilbeeAppHost

    class _Host(LilbeeAppHost):
        def compose(self):
            yield ModelCard(_row(ModelCompat.SUPPORTED))

    app = _Host()
    async with app.run_test(size=(60, 20)) as pilot:
        card = app.query_one(ModelCard)
        before = str(app.query_one(".card-body", Static).render())
        card.selected = True
        await pilot.pause()
        after = str(app.query_one(".card-body", Static).render())
        assert before != after  # the highlight-only hint appeared
