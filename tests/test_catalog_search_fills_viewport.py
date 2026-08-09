"""Catalog search must fill the viewport with matches.

Both failures reproduced here were laid-out geometry, not query limits: the
grouped sections spent the viewport on headings, and the scroll offset from
the pre-search dataset survived the filter pass.
"""

from __future__ import annotations

from unittest import mock

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import await_chat

# Terminal the assertions are calibrated against: 40 rows total, of which the
# catalog body gets 32 after the top and bottom bars.
_TERMINAL_SIZE = (120, 40)
# One card is 6 body lines plus a top and bottom border line.
_CARD_HEIGHT = 8


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture(autouse=True)
def _suppress_catalog_auto_hf_fetch():
    """Block the mount-time HF fetch so the fixture data is the whole dataset."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    with mock.patch.object(CatalogScreen, "_fetch_initial_hf_models_for_task"):
        yield


@pytest.fixture()
def _mock_resolve():
    with mock.patch(
        "lilbee.providers.engine_params.resolve_model_path",
        return_value=cfg.models_dir / "fake.gguf",
    ):
        yield


def _featured_families(count: int):
    """Featured chat families; these land in the ★ Picks section."""
    from lilbee.catalog import ModelFamily, ModelVariant
    from lilbee.catalog.types import ModelCompat

    return [
        ModelFamily(
            slug=f"searchme{i}",
            name=f"SearchMe {i}",
            task="chat",
            description="matching chat model",
            variants=(
                ModelVariant(
                    hf_repo=f"test/searchme-{i}",
                    filename=f"searchme-{i}-Q4.gguf",
                    param_count="7B",
                    quant="Q4_K_M",
                    size_mb=4000,
                    compat=ModelCompat.SUPPORTED,
                ),
            ),
        )
        for i in range(count)
    ]


def _hf_chat_models(count: int):
    """Non-featured chat rows; these land in the task section."""
    from lilbee.catalog.models import CatalogModel
    from lilbee.catalog.types import ModelCompat, ModelTask

    return [
        CatalogModel(
            hf_repo=f"hforg/searchme-hf-{i}",
            gguf_filename=f"searchme-hf-{i}.gguf",
            size_gb=4.0,
            min_ram_gb=8.0,
            description="matching hf model",
            featured=False,
            downloads=1000 - i,
            task=ModelTask.CHAT,
            compat=ModelCompat.SUPPORTED,
        )
        for i in range(count)
    ]


def _mock_catalog_deps(families):
    return mock.patch.multiple(
        "lilbee.cli.tui.screens.catalog",
        get_families=mock.MagicMock(return_value=families),
        get_catalog=mock.MagicMock(return_value=mock.MagicMock(models=[])),
    )


def _mock_remote_models():
    return mock.patch(
        "lilbee.cli.tui.screens.catalog.classify_all_remote_models",
        return_value=[],
    )


async def _open_catalog_chat_grid(app, pilot):
    """Push the catalog, pin the Chat tab, and return the screen."""
    await await_chat(app, pilot)
    await pilot.pause()
    app.switch_view("Catalog")
    await pilot.pause()
    screen = app.screen
    # The 6-tab layout defaults to Discover during the initial mount race;
    # Discover paints rails, not sections.
    screen._active_tab_id_cache = "chat"
    screen._activation_settled = True
    await pilot.pause()
    return screen


async def _type_search(pilot, text: str) -> None:
    """Reveal the filter with / and type *text*, then wait past the debounce."""
    await pilot.press("slash")
    await pilot.pause()
    for char in text:
        await pilot.press("space" if char == " " else char)
    await pilot.pause(0.4)
    await pilot.pause()


def _cards_laid_out_in_viewport(screen) -> int:
    """Count cards whose full height falls inside the scroll container's viewport."""
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    container = screen._grid_container
    viewport = container.region
    total = 0
    for grid in container.query(ModelGrid):
        region = grid.region
        top = max(region.y, viewport.y)
        bottom = min(region.y + region.height, viewport.y + viewport.height)
        visible_lines = max(0, bottom - top)
        total += min((visible_lines // _CARD_HEIGHT) * grid.columns_per_row, len(grid.rows))
    return total


async def test_search_matches_fill_the_viewport(_mock_resolve):
    """Matches spread across Picks/Installed/Chat must not spend the screen on headings.

    Before the flat search section, the three headings plus the two
    one-card sections above the task section left a single clipped row of
    matches: four cards laid out in a 32-line viewport that holds nine.
    """
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    with _mock_catalog_deps(_featured_families(3)), _mock_remote_models():
        app = LilbeeApp()
        async with app.run_test(size=_TERMINAL_SIZE) as pilot:
            screen = await _open_catalog_chat_grid(app, pilot)
            screen._hf_models = _hf_chat_models(30)
            screen._hf_fetched_tasks.add(ModelTask.CHAT)
            screen._installed_names.add("hforg/searchme-hf-0")
            screen._data_version += 1
            screen._refresh_view()
            await pilot.pause()
            await pilot.pause()

            await _type_search(pilot, "searchme")

            container = screen._grid_container
            grids = list(container.query(ModelGrid))
            headings = list(container.query(".section-heading"))
            # One heading is the only chrome the matches pay for, so every card
            # row the rest of the viewport can hold is laid out. Lower bound:
            # when focus lands on the grid the heading scrolls off and one more
            # row fits.
            usable_lines = container.region.height - headings[0].region.height
            capacity = (usable_lines // _CARD_HEIGHT) * grids[0].columns_per_row
            assert _cards_laid_out_in_viewport(screen) >= capacity

            assert len(grids) == 1, "an active search renders one flat result set"
            assert len(grids[0].rows) == 33, "every match belongs to the result set"
            assert len(headings) == 1


async def test_search_after_scrolling_starts_at_the_top(_mock_resolve):
    """A filter pass must not keep the scroll offset of the pre-search dataset.

    Textual clamps the stale offset to the shorter result set's maximum,
    which parked the viewport at the end and rendered the matches above it.
    """
    from lilbee.cli.tui.app import LilbeeApp

    with _mock_catalog_deps(_featured_families(60)), _mock_remote_models():
        app = LilbeeApp()
        async with app.run_test(size=_TERMINAL_SIZE) as pilot:
            screen = await _open_catalog_chat_grid(app, pilot)
            container = screen._grid_container
            container.scroll_to(y=100, animate=False)
            await pilot.pause()
            assert container.scroll_y > 0

            # "searchme 5" matches SearchMe 5 and SearchMe 50..59.
            await _type_search(pilot, "searchme 5")

            assert container.scroll_y == 0
            assert container.max_scroll_y > 0, "the result set is taller than the viewport"


async def test_a_search_matching_nothing_mounts_no_result_section(_mock_resolve):
    """Zero matches must reach the empty-grid CTAs, not an empty "Matches" heading.

    Flattening runs on whatever survived the filter, so the no-match case hands
    it an empty section list; without its own guard it would mount a heading
    over nothing and the CTA branch would never be reached.
    """
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    with _mock_catalog_deps(_featured_families(6)), _mock_remote_models():
        app = LilbeeApp()
        async with app.run_test(size=_TERMINAL_SIZE) as pilot:
            screen = await _open_catalog_chat_grid(app, pilot)

            await _type_search(pilot, "zzzznotamodel")

            container = screen._grid_container
            assert not list(container.query(ModelGrid))
            headings = [str(h.renderable) for h in container.query(".section-heading")]
            assert msg.HEADING_MATCHES not in headings
