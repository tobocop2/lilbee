"""Curated featured-model loader, sourced from featured_models.toml."""

from pathlib import Path

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask


def _load_featured() -> tuple[
    tuple[CatalogModel, ...],
    tuple[CatalogModel, ...],
    tuple[CatalogModel, ...],
    tuple[CatalogModel, ...],
]:
    """Load featured models from the TOML file, cached after first call."""
    import tomllib

    toml_path = Path(__file__).parent.parent / "featured_models.toml"
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    def _build(task: ModelTask) -> tuple[CatalogModel, ...]:
        return tuple(
            CatalogModel(
                hf_repo=m["hf_repo"],
                gguf_filename=m["gguf_filename"],
                size_gb=m["size_gb"],
                min_ram_gb=m["min_ram_gb"],
                description=m["description"],
                featured=True,
                downloads=0,
                task=task,
                recommended=m.get("recommended", False),
            )
            for m in data.get(task, [])
        )

    return (
        _build(ModelTask.CHAT),
        _build(ModelTask.EMBEDDING),
        _build(ModelTask.VISION),
        _build(ModelTask.RERANK),
    )


FEATURED_CHAT, FEATURED_EMBEDDING, FEATURED_VISION, FEATURED_RERANK = _load_featured()

# Maps vision catalog entries to their mmproj (CLIP projection) filenames.
# Vision models need both the main GGUF and the mmproj file to work.
# Keys are hf_repo identifiers; values are glob patterns resolved at download time.
# Every FEATURED_VISION entry MUST have a corresponding key here.
DEFAULT_MMPROJ_PATTERN = "*mmproj*.gguf"

VISION_MMPROJ_FILES: dict[str, str] = {
    "noctrex/LightOnOCR-2-1B-GGUF": DEFAULT_MMPROJ_PATTERN,
}

FEATURED_ALL: tuple[CatalogModel, ...] = (
    FEATURED_CHAT + FEATURED_EMBEDDING + FEATURED_VISION + FEATURED_RERANK
)
