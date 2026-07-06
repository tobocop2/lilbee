"""Model catalog. Discovers available GGUF models from HuggingFace.

Three levels:
1. Featured models. Curated favorites (hardcoded, always available)
2. HF API models. Fetched from HuggingFace API, paginated and filterable
3. Combined catalog. Featured first, then HF results
"""

from lilbee.catalog.download import (
    DownloadConfig,
    download_mmproj,
    download_model,
    find_mmproj_file,
    resolve_filename,
)
from lilbee.catalog.download_progress import ProgressCallback, make_download_callback
from lilbee.catalog.families import get_families
from lilbee.catalog.featured import (
    FEATURED_ALL,
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_RERANK,
    FEATURED_VISION,
    VISION_MMPROJ_FILES,
)
from lilbee.catalog.formatting import (
    PARAM_COUNT_RE,
    QUANT_TIERS,
    EnrichedModel,
    clean_display_name,
    display_label_for_ref,
    download_task_name,
    enrich_catalog,
    extract_quant,
    quant_tier,
)
from lilbee.catalog.models import (
    CatalogModel,
    CatalogResult,
    DownloadProgress,
    ModelFamily,
    ModelVariant,
)
from lilbee.catalog.query import (
    CatalogIndex,
    build_adhoc_entry,
    find_catalog_entry,
    get_catalog,
    is_rerank_ref,
    resolve_pull_target,
)

__all__ = [
    "FEATURED_ALL",
    "FEATURED_CHAT",
    "FEATURED_EMBEDDING",
    "FEATURED_RERANK",
    "FEATURED_VISION",
    "PARAM_COUNT_RE",
    "QUANT_TIERS",
    "VISION_MMPROJ_FILES",
    "CatalogIndex",
    "CatalogModel",
    "CatalogResult",
    "DownloadConfig",
    "DownloadProgress",
    "EnrichedModel",
    "ModelFamily",
    "ModelVariant",
    "ProgressCallback",
    "build_adhoc_entry",
    "clean_display_name",
    "display_label_for_ref",
    "download_mmproj",
    "download_model",
    "download_task_name",
    "enrich_catalog",
    "extract_quant",
    "find_catalog_entry",
    "find_mmproj_file",
    "get_catalog",
    "get_families",
    "is_rerank_ref",
    "make_download_callback",
    "quant_tier",
    "resolve_filename",
    "resolve_pull_target",
]
