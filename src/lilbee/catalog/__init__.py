"""Model catalog — discovers available GGUF models from HuggingFace.

Three levels:
1. Featured models — curated favorites (hardcoded, always available)
2. HF API models — fetched from HuggingFace API, paginated and filterable
3. Combined catalog — featured first, then HF results

This package preserves the historical ``lilbee.catalog`` public API by
re-exporting from sibling submodules.
"""

from typing import Any

from lilbee.catalog.download import _QUANT_PREFERENCE as _QUANT_PREFERENCE
from lilbee.catalog.download import (
    DownloadConfig,
    download_model,
    fetch_model_file_size,
    find_mmproj_file,
    resolve_filename,
)
from lilbee.catalog.download import _download_mmproj as _download_mmproj
from lilbee.catalog.download import _finalize_download as _finalize_download
from lilbee.catalog.download import (
    _mmproj_in_models_dir_matching as _mmproj_in_models_dir_matching,
)
from lilbee.catalog.download import _pick_best_gguf as _pick_best_gguf
from lilbee.catalog.download import _register_model as _register_model
from lilbee.catalog.download import _resolve_mmproj_filename as _resolve_mmproj_filename
from lilbee.catalog.families import _FAMILY_NAME_RE as _FAMILY_NAME_RE
from lilbee.catalog.families import _build_families as _build_families
from lilbee.catalog.families import _catalog_to_variant as _catalog_to_variant
from lilbee.catalog.families import _extract_family_name as _extract_family_name
from lilbee.catalog.families import _family_slug as _family_slug
from lilbee.catalog.families import get_families
from lilbee.catalog.featured import _DEFAULT_MMPROJ_PATTERN as _DEFAULT_MMPROJ_PATTERN
from lilbee.catalog.featured import (
    FEATURED_ALL,
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_RERANK,
    FEATURED_VISION,
    VISION_MMPROJ_FILES,
)
from lilbee.catalog.featured import _load_featured as _load_featured
from lilbee.catalog.formatting import _DISPLAY_NAME_DATE_SUFFIX as _DISPLAY_NAME_DATE_SUFFIX
from lilbee.catalog.formatting import _DISPLAY_NAME_META_PREFIX as _DISPLAY_NAME_META_PREFIX
from lilbee.catalog.formatting import _DISPLAY_NAME_SUFFIXES as _DISPLAY_NAME_SUFFIXES
from lilbee.catalog.formatting import (
    PARAM_COUNT_RE,
    QUANT_TIERS,
    EnrichedModel,
    clean_display_name,
    display_label_for_ref,
    enrich_catalog,
    extract_quant,
    quant_tier,
)
from lilbee.catalog.formatting import _derive_param_count as _derive_param_count
from lilbee.catalog.hf_client import _BYTES_PER_MB as _BYTES_PER_MB
from lilbee.catalog.hf_client import _DEFAULT_TIMEOUT as _DEFAULT_TIMEOUT
from lilbee.catalog.hf_client import _EMPTY_HF_PAGE as _EMPTY_HF_PAGE
from lilbee.catalog.hf_client import _HF_CACHE_MAX_ENTRIES as _HF_CACHE_MAX_ENTRIES
from lilbee.catalog.hf_client import _HF_CACHE_TTL as _HF_CACHE_TTL
from lilbee.catalog.hf_client import _HF_EXPAND_FIELDS as _HF_EXPAND_FIELDS
from lilbee.catalog.hf_client import _HF_GGUF_SEARCH_TERM as _HF_GGUF_SEARCH_TERM
from lilbee.catalog.hf_client import HF_API_URL, ProgressCallback, make_download_callback
from lilbee.catalog.hf_client import _CallbackProgressBar as _CallbackProgressBar
from lilbee.catalog.hf_client import _estimate_size_from_siblings as _estimate_size_from_siblings
from lilbee.catalog.hf_client import _fetch_hf_models as _fetch_hf_models
from lilbee.catalog.hf_client import _has_gguf_siblings as _has_gguf_siblings
from lilbee.catalog.hf_client import _hf_cache as _hf_cache
from lilbee.catalog.hf_client import _hf_cache_lock as _hf_cache_lock
from lilbee.catalog.hf_client import _hf_headers as _hf_headers
from lilbee.catalog.hf_client import _hf_search_value as _hf_search_value
from lilbee.catalog.hf_client import _hf_token as _hf_token
from lilbee.catalog.hf_client import _ProgressTracker as _ProgressTracker
from lilbee.catalog.models import (
    CatalogModel,
    CatalogResult,
    DownloadProgress,
    ModelFamily,
    ModelVariant,
)
from lilbee.catalog.models import _HfGgufMeta as _HfGgufMeta
from lilbee.catalog.models import _HfPage as _HfPage
from lilbee.catalog.query import _PIPELINE_TO_TASK as _PIPELINE_TO_TASK
from lilbee.catalog.query import _SIZE_RANGES as _SIZE_RANGES
from lilbee.catalog.query import _SORT_KEYS as _SORT_KEYS
from lilbee.catalog.query import (
    CatalogIndex,
    build_adhoc_entry,
    find_catalog_entry,
    get_catalog,
    is_rerank_ref,
    resolve_pull_target,
)
from lilbee.catalog.query import _build_catalog_index as _build_catalog_index
from lilbee.catalog.query import _get_installed_models as _get_installed_models
from lilbee.catalog.query import _is_hf_repo_id as _is_hf_repo_id
from lilbee.catalog.query import _pipeline_to_task as _pipeline_to_task
from lilbee.catalog.query import _search_blob as _search_blob
from lilbee.catalog.query import _sort_models as _sort_models
from lilbee.catalog.query import _task_to_pipeline as _task_to_pipeline


def __getattr__(name: str) -> Any:
    """Expose ``catalog.cfg`` lazily so ``monkeypatch.setattr(catalog.cfg, ...)`` still works."""
    if name == "cfg":
        # circular: lilbee.catalog -> lilbee.config via cfg
        from lilbee.core.config import cfg

        return cfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FEATURED_ALL",
    "FEATURED_CHAT",
    "FEATURED_EMBEDDING",
    "FEATURED_RERANK",
    "FEATURED_VISION",
    "HF_API_URL",
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
    "download_model",
    "enrich_catalog",
    "extract_quant",
    "fetch_model_file_size",
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
