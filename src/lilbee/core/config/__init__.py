"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
Uses pydantic-settings for automatic env var loading with TOML config file support.
"""

# ruff: noqa: I001
from .defaults import (
    CHUNK_CONCEPTS_TABLE as CHUNK_CONCEPTS_TABLE,
    CHUNKS_TABLE as CHUNKS_TABLE,
    CITATIONS_TABLE as CITATIONS_TABLE,
    CONCEPT_EDGES_TABLE as CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE as CONCEPT_NODES_TABLE,
    DEFAULT_CRAWL_EXCLUDE_PATTERNS as DEFAULT_CRAWL_EXCLUDE_PATTERNS,
    DEFAULT_HTTP_TIMEOUT as DEFAULT_HTTP_TIMEOUT,
    DEFAULT_IGNORE_DIRS as DEFAULT_IGNORE_DIRS,
    DEFAULT_NUM_CTX as DEFAULT_NUM_CTX,
    MEMORIES_TABLE as MEMORIES_TABLE,
    META_TABLE as META_TABLE,
    PAGE_TEXTS_TABLE as PAGE_TEXTS_TABLE,
    SOURCES_TABLE as SOURCES_TABLE,
)
from .enums import (
    ClustererBackend as ClustererBackend,
    WikiEntityMode as WikiEntityMode,
)
from .model import (
    Config as Config,
    cfg as cfg,
    config_load_error as config_load_error,
)
from .validators import (
    ConfigField as ConfigField,
)

__all__ = [
    "CHUNKS_TABLE",
    "CHUNK_CONCEPTS_TABLE",
    "CITATIONS_TABLE",
    "CONCEPT_EDGES_TABLE",
    "CONCEPT_NODES_TABLE",
    "DEFAULT_CRAWL_EXCLUDE_PATTERNS",
    "DEFAULT_HTTP_TIMEOUT",
    "DEFAULT_IGNORE_DIRS",
    "DEFAULT_NUM_CTX",
    "MEMORIES_TABLE",
    "META_TABLE",
    "PAGE_TEXTS_TABLE",
    "SOURCES_TABLE",
    "ClustererBackend",
    "Config",
    "ConfigField",
    "WikiEntityMode",
    "cfg",
    "config_load_error",
]
