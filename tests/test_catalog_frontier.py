"""Catalog Frontier (cloud) section tests.

The catalog renders a Frontier super-section above local sections when
the user has any provider key configured. Rows render as
``FrontierCatalogRow`` and carry provider + key-status pills. Adding
or clearing an API key republishes ``provider_availability_changed_signal``
and the catalog rebuilds.
"""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.cli.tui.screens.catalog_utils import (
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
    matches_search,
)
from lilbee.modelhub.model_manager import RemoteModel
from lilbee.modelhub.models import ModelTask


def _frontier(
    name: str,
    *,
    provider: str = "Gemini",
    status: KeyStatus = KeyStatus.READY,
) -> FrontierCatalogRow:
    return FrontierCatalogRow(
        name=name,
        ref=name,
        task=ModelTask.CHAT,
        provider=provider,
        provider_id=provider.lower(),
        key_status=status,
        is_curated=True,
    )


def _local(name: str, *, installed: bool = False, featured: bool = False) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task=ModelTask.CHAT,
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=featured,
        installed=installed,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
    )


class TestRowTypes:
    def test_local_and_frontier_are_disjoint(self) -> None:
        local = _local("Qwen3 0.6B")
        frontier = _frontier("gemini-2.0-flash")
        assert isinstance(local, LocalCatalogRow)
        assert isinstance(frontier, FrontierCatalogRow)
        assert not isinstance(local, FrontierCatalogRow)

    def test_matches_search_local_uses_task_and_quant(self) -> None:
        row = _local("Qwen3")
        row.quant = "Q4_K_M"
        assert matches_search(row, "qwen") is True
        assert matches_search(row, "q4") is True
        assert matches_search(row, "gemini") is False

    def test_matches_search_frontier_uses_provider(self) -> None:
        row = _frontier("gemini-2.0-flash", provider="Gemini")
        # Provider name and the model id both match.
        assert matches_search(row, "gemini") is True
        assert matches_search(row, "flash") is True
        # Frontier rows never read the local-only fields, so an
        # imaginary local-side filter must not match a frontier row.
        assert matches_search(row, "q4_k_m") is False


class TestGroupRowsForGrid:
    def test_frontier_super_section_appears_first(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_rows_for_grid

        local = [_local("Qwen3", featured=True)]
        frontier = [_frontier("gemini-2.0-flash"), _frontier("gpt-4o", provider="OpenAI")]
        sections = _group_rows_for_grid(local, frontier)

        non_empty = [s for s in sections if s.rows]
        assert non_empty[0].is_frontier is True
        # Both providers get their own subsection, sorted alphabetically.
        provider_headings = [s.heading for s in non_empty if s.is_frontier]
        assert provider_headings[0].endswith("Gemini")
        assert provider_headings[1].endswith("OpenAI")
        # Local sections follow.
        assert non_empty[-1].is_frontier is False


class TestBuildFrontierRows:
    def test_no_keys_yields_no_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        with mock.patch(
            "lilbee.modelhub.model_manager.discover_api_models",
            return_value={},
        ):
            screen = CatalogScreen.__new__(CatalogScreen)
            assert screen._build_frontier_rows("") == []

    def test_curated_models_get_is_curated_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Models in the curated short list flip is_curated=True."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.core.config import cfg

        snapshot = cfg.gemini_api_key
        cfg.gemini_api_key = "sk-test"
        try:
            with mock.patch(
                "lilbee.modelhub.model_manager.discover_api_models",
                return_value={
                    "Gemini": [
                        RemoteModel(
                            name="gemini-2.0-flash",
                            task=ModelTask.CHAT,
                            family="",
                            parameter_size="",
                            provider="Gemini",
                        ),
                        RemoteModel(
                            name="gemini-experimental-9000",
                            task=ModelTask.CHAT,
                            family="",
                            parameter_size="",
                            provider="Gemini",
                        ),
                    ]
                },
            ):
                screen = CatalogScreen.__new__(CatalogScreen)
                rows = screen._build_frontier_rows("")
        finally:
            cfg.gemini_api_key = snapshot

        by_name = {r.name: r for r in rows}
        assert by_name["gemini-2.0-flash"].is_curated is True
        assert by_name["gemini-experimental-9000"].is_curated is False
        # All rows from a key-configured provider are READY.
        assert all(r.key_status == KeyStatus.READY for r in rows)
