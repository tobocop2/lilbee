"""Group featured models into display families."""

import re

from lilbee.catalog.featured import (
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_RERANK,
    FEATURED_VISION,
)
from lilbee.catalog.formatting import clean_display_name, extract_quant
from lilbee.catalog.models import CatalogModel, ModelFamily, ModelVariant
from lilbee.modelhub.models import ModelTask

_FAMILY_NAME_RE = re.compile(r"^(.+?)\s+\d")


def _extract_family_name(model_name: str) -> str:
    """Extract the family name by stripping the trailing parameter count.
    Applies clean_display_name first to strip -GGUF, -Instruct, etc.

    "Qwen3 8B" -> "Qwen3", "Qwen3-Coder 30B A3B" -> "Qwen3-Coder",
    "Nomic Embed Text v1.5" -> "Nomic Embed Text v1.5" (no trailing number pattern).
    """
    cleaned = clean_display_name(model_name)
    m = _FAMILY_NAME_RE.match(cleaned)
    return m.group(1) if m else cleaned


def _catalog_to_variant(model: CatalogModel) -> ModelVariant:
    """Convert a CatalogModel to a ModelVariant."""
    # Local import to avoid pulling formatting helpers into hf_client/featured.
    from lilbee.catalog.formatting import derive_param_count

    return ModelVariant(
        hf_repo=model.hf_repo,
        filename=model.gguf_filename,
        param_count=derive_param_count(model),
        quant=extract_quant(model.gguf_filename),
        size_mb=int(model.size_gb * 1024),
        recommended=model.recommended,
    )


def _family_slug(display_name: str) -> str:
    """Stable slug for a family, derived from its display name."""
    return _extract_family_name(display_name).lower().replace(" ", "-")


def _build_families(models: tuple[CatalogModel, ...], task: str) -> list[ModelFamily]:
    """Group CatalogModels into families by display-derived family name."""
    groups: dict[str, list[CatalogModel]] = {}
    order: list[str] = []
    for m in models:
        family = _extract_family_name(m.display_name)
        if family not in groups:
            order.append(family)
        groups.setdefault(family, []).append(m)

    families: list[ModelFamily] = []
    for family_name in order:
        members = groups[family_name]
        representative = next((m for m in members if m.recommended), members[0])
        variants = [_catalog_to_variant(m) for m in members]
        families.append(
            ModelFamily(
                slug=_family_slug(representative.display_name),
                name=family_name,
                task=task,
                description=representative.description,
                variants=tuple(variants),
            )
        )
    return families


def get_families() -> list[ModelFamily]:
    """Get all featured models grouped into families.
    Returns families ordered: chat, then embedding, then vision, then reranker.
    Within each family, variants are ordered smallest to largest, with
    the largest marked as recommended (for multi-variant families).
    """
    return (
        _build_families(FEATURED_CHAT, ModelTask.CHAT)
        + _build_families(FEATURED_EMBEDDING, ModelTask.EMBEDDING)
        + _build_families(FEATURED_VISION, ModelTask.VISION)
        + _build_families(FEATURED_RERANK, ModelTask.RERANK)
    )
