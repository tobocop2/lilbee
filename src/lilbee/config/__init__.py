"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
Uses pydantic-settings for automatic env var loading with TOML config file support.

The package is split into:

- :mod:`lilbee.config.enums` — :class:`ClustererBackend`, :class:`WikiEntityMode`
- :mod:`lilbee.config.parsing` — boolean parsing helpers
- :mod:`lilbee.config.defaults` — frozen literal defaults and table-name constants
- :mod:`lilbee.config.validators` — catalog/role validators and :func:`ConfigField`
- :mod:`lilbee.config.model` — the :class:`Config` dataclass and the ``cfg`` singleton

Every public symbol the rest of the codebase imported from the old
``lilbee.config`` module is re-exported here so existing imports keep working.
The ``as``-aliased imports below are intentional re-exports; ruff treats them
as explicit and stops flagging them as unused (F401).
"""

# ruff: noqa: I001
from .defaults import (
    CHUNK_CONCEPTS_TABLE as CHUNK_CONCEPTS_TABLE,
    CHUNKS_TABLE as CHUNKS_TABLE,
    CITATIONS_TABLE as CITATIONS_TABLE,
    CONCEPT_EDGES_TABLE as CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE as CONCEPT_NODES_TABLE,
    DEFAULT_ALLOWED_NER_LABELS as DEFAULT_ALLOWED_NER_LABELS,
    DEFAULT_CRAWL_EXCLUDE_PATTERNS as DEFAULT_CRAWL_EXCLUDE_PATTERNS,
    DEFAULT_HTTP_TIMEOUT as DEFAULT_HTTP_TIMEOUT,
    DEFAULT_IGNORE_DIRS as DEFAULT_IGNORE_DIRS,
    DEFAULT_NUM_CTX as DEFAULT_NUM_CTX,
    META_TABLE as META_TABLE,
    SOURCES_TABLE as SOURCES_TABLE,
    _ARCHIVE_EXCLUDE as _ARCHIVE_EXCLUDE,
    _ATTACHMENT_EXCLUDE as _ATTACHMENT_EXCLUDE,
    _AUTH_EXCLUDE as _AUTH_EXCLUDE,
    _DEFAULT_CORS_ORIGIN_REGEX as _DEFAULT_CORS_ORIGIN_REGEX,
    _DEFAULT_SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
    _DUPLICATE_VIEW_EXCLUDE as _DUPLICATE_VIEW_EXCLUDE,
    _ECOMMERCE_EXCLUDE as _ECOMMERCE_EXCLUDE,
    _FEED_EXCLUDE as _FEED_EXCLUDE,
    _MEDIAWIKI_EXCLUDE as _MEDIAWIKI_EXCLUDE,
    _META_EXCLUDE as _META_EXCLUDE,
    _TRACKING_EXCLUDE as _TRACKING_EXCLUDE,
    _WP_EXCLUDE as _WP_EXCLUDE,
)
from .enums import (
    ClustererBackend as ClustererBackend,
    WikiEntityMode as WikiEntityMode,
)
from .model import (
    Config as Config,
    _PlainEnvSource as _PlainEnvSource,
    _TomlSource as _TomlSource,
    _build_cfg as _build_cfg,
    _model_defaults_dict as _model_defaults_dict,
    cfg as cfg,
    config_load_error as config_load_error,
)
from .parsing import (
    _BOOL_FALSE as _BOOL_FALSE,
    _BOOL_TRUE as _BOOL_TRUE,
    _parse_bool as _parse_bool,
)
from .validators import (
    ConfigField as ConfigField,
    _MODEL_FIELD_TO_TASK as _MODEL_FIELD_TO_TASK,
    _SKIP_MODEL_TASK_VALIDATION_ENV as _SKIP_MODEL_TASK_VALIDATION_ENV,
    _enforce_role_match as _enforce_role_match,
    _find_model_catalog_entry as _find_model_catalog_entry,
    _model_task_validation_bypassed as _model_task_validation_bypassed,
    _skips_catalog_check as _skips_catalog_check,
    validate_model_task_assignment as validate_model_task_assignment,
)

__all__ = [
    "CHUNKS_TABLE",
    "CHUNK_CONCEPTS_TABLE",
    "CITATIONS_TABLE",
    "CONCEPT_EDGES_TABLE",
    "CONCEPT_NODES_TABLE",
    "DEFAULT_ALLOWED_NER_LABELS",
    "DEFAULT_CRAWL_EXCLUDE_PATTERNS",
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_IGNORE_DIRS",
    "DEFAULT_NUM_CTX",
    "META_TABLE",
    "SOURCES_TABLE",
    "ClustererBackend",
    "Config",
    "ConfigField",
    "WikiEntityMode",
    "cfg",
    "config_load_error",
    "validate_model_task_assignment",
]
