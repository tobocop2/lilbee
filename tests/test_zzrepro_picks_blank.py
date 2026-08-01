"""Temporary repro: catalog when picks resolution fails but HF browse works."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from tests._lilbee_app_test_host import LilbeeAppHost


def _dump(screen, label: str) -> None:
    spinner = screen.query_one("#catalog-loading-spinner", Static)
    print(f"--- {label} ---")
    print("FAMILIES:", len(screen._families))
    print("LOADING_MORE:", screen._loading_more, "SEARCH:", screen._search_in_flight)
    print("SPINNER_DISPLAY:", spinner.styles.display)
    print("SPINNER_TEXT:", repr(str(spinner.content)))
    texts = [str(w.content) for w in screen._grid_container.query(Static)]
    print("GRID_STATICS:", texts)
    print("GRID_CHILD_COUNT:", len(screen._grid_container.children))


@pytest.mark.live_picks
async def test_picks_fail_hf_ok(monkeypatch) -> None:
    from lilbee.catalog import picks as picks_mod
    from lilbee.catalog.models import CatalogResult
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    picks_mod.reset_picks()

    import lilbee.cli.tui.screens.catalog as cat_mod

    def _families_boom():
        import time
        time.sleep(0.6)
        raise OSError("offline")

    def _fake_catalog(**kwargs):
        import time
        time.sleep(0.6)
        from tests.conftest import make_test_catalog_model

        return CatalogResult(
            models=[make_test_catalog_model("BrowseOne"), make_test_catalog_model("BrowseTwo")],
            total=2,
            has_more=False,
            limit=20,
            offset=0,
        )

    monkeypatch.setattr(cat_mod, "get_families", _families_boom, raising=True)
    monkeypatch.setattr(cat_mod, "get_catalog", _fake_catalog, raising=True)

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogScreen()

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = pilot.app.query_one(CatalogScreen)
        await pilot.pause()
        _dump(screen, "after 1 pause")
        print("SCREEN_TEXT:", repr(pilot.app.screen.export_text() if hasattr(pilot.app.screen, "export_text") else ""))
        for _ in range(15):
            await pilot.pause()
        _dump(screen, "settled")
    picks_mod.reset_picks()
