"""Content flag: safety-stripped models are derived from upstream HF tags."""

from __future__ import annotations

import pytest

from lilbee.catalog.content import is_safety_stripped


@pytest.mark.parametrize(
    "tags",
    [
        ["gguf", "uncensored", "text-generation"],
        ["gguf", "abliterated"],
        ["decensored"],
        ["nsfw", "erp"],
        ["heretic", "gguf"],
        ["UNCENSORED"],
        ["Abliterated"],
        ["gguf", "base_model:mondk/MiniCPM5-2B-Abliterated-Uncensored-Safetensors"],
    ],
)
def test_stripped_tags_are_flagged(tags: list[str]) -> None:
    """Every surveyed stem flags the row, including one carried by a base_model tag."""
    assert is_safety_stripped(tags) is True


@pytest.mark.parametrize(
    "tags",
    [
        ["gguf", "llama.cpp", "conversational"],
        [],
    ],
)
def test_ordinary_tags_are_clean(tags: list[str]) -> None:
    assert is_safety_stripped(tags) is False


def test_stripped_flag_flows_from_catalog_row_to_every_surface() -> None:
    """One flag on CatalogModel reaches the TUI row, server, MCP, and app rows."""
    from unittest.mock import patch

    from lilbee.app.models import CatalogEntryData
    from lilbee.catalog.families import _catalog_to_variant
    from lilbee.catalog.formatting import enrich_catalog
    from lilbee.catalog.models import CatalogModel, CatalogResult, ModelFamily
    from lilbee.catalog.types import ModelCompat, ModelTask
    from lilbee.cli.tui.screens.catalog_grouping import for_you_by_role
    from lilbee.cli.tui.screens.catalog_utils import catalog_to_row, variant_to_row
    from lilbee.mcp_server import catalog_browse
    from lilbee.server.handlers.models import _build_catalog_entry

    model = CatalogModel(
        hf_repo="x/Qwen3-8B-Uncensored-GGUF",
        gguf_filename="*Q4_K_M.gguf",
        size_gb=4.6,
        min_ram_gb=6.9,
        description="",
        featured=False,
        downloads=10,
        task=ModelTask.CHAT,
        compat=ModelCompat.SUPPORTED,
        safety_stripped=True,
    )
    family = ModelFamily(
        slug="qwen3",
        name="Qwen3",
        task=ModelTask.CHAT,
        description="",
        variants=(_catalog_to_variant(model),),
    )
    assert family.variants[0].safety_stripped is True
    assert catalog_to_row(model, installed=False).safety_stripped is True
    assert variant_to_row(family.variants[0], family, installed=False).safety_stripped is True

    (enriched,) = enrich_catalog(
        CatalogResult(total=1, limit=20, offset=0, models=[model], has_more=False), set()
    )
    assert enriched.safety_stripped is True
    entry = _build_catalog_entry(enriched, available_bytes=None, families_by_repo={})
    assert entry.safety_stripped is True
    assert CatalogEntryData.from_catalog_model(model).safety_stripped is True

    with patch(
        "lilbee.catalog.query.get_catalog",
        return_value=CatalogResult(total=1, limit=20, offset=0, models=[model], has_more=False),
    ):
        (payload,) = catalog_browse()["models"]
    assert payload["safety_stripped"] is True
    assert for_you_by_role([catalog_to_row(model, installed=False)]) == []
