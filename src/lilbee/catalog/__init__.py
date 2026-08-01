"""Model catalog. Discovers available GGUF models from HuggingFace.

Two levels:
1. Picks. The most popular models of each parameter tier, fetched live
2. HF API models. Fetched from HuggingFace API, paginated and filterable

The combined catalog puts picks first, then HF results.
"""

from lilbee.catalog.download import (
    DownloadConfig,
    download_mmproj,
    download_model,
    resolve_filename,
)
from lilbee.catalog.download_progress import ProgressCallback, make_download_callback
from lilbee.catalog.families import get_families
from lilbee.catalog.formatting import (
    PARAM_COUNT_RE,
    QUANT_TIERS,
    EnrichedModel,
    agent_model_id,
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
from lilbee.catalog.picks import (
    DEFAULT_MMPROJ_PATTERN,
    find_pick,
    get_picks,
    picks_for,
    reset_picks,
)
from lilbee.catalog.query import (
    build_adhoc_entry,
    get_catalog,
    is_rerank_ref,
    resolve_pull_target,
    size_bucket,
)

__all__ = [
    "DEFAULT_MMPROJ_PATTERN",
    "PARAM_COUNT_RE",
    "QUANT_TIERS",
    "CatalogModel",
    "CatalogResult",
    "DownloadConfig",
    "DownloadProgress",
    "EnrichedModel",
    "ModelFamily",
    "ModelVariant",
    "ProgressCallback",
    "agent_model_id",
    "build_adhoc_entry",
    "clean_display_name",
    "display_label_for_ref",
    "download_mmproj",
    "download_model",
    "download_task_name",
    "enrich_catalog",
    "extract_quant",
    "find_pick",
    "get_catalog",
    "get_families",
    "get_picks",
    "is_rerank_ref",
    "make_download_callback",
    "picks_for",
    "quant_tier",
    "reset_picks",
    "resolve_filename",
    "resolve_pull_target",
    "size_bucket",
]
