"""Constants and configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
"""

import os
import sys
from pathlib import Path


def _default_data_dir() -> Path:
    """Return platform-appropriate data directory.

    - Linux:   ~/.local/share/lilbee  (XDG_DATA_HOME)
    - macOS:   ~/Library/Application Support/lilbee
    - Windows: %LOCALAPPDATA%/lilbee
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "lilbee"


def _env(key: str, default: str) -> str:
    """Read a LILBEE_ env var with fallback."""
    return os.environ.get(f"LILBEE_{key}", default)


def _env_int(key: str, default: int) -> int:
    """Read a LILBEE_ env var as int with fallback."""
    raw = os.environ.get(f"LILBEE_{key}")
    if raw is None:
        return default
    return int(raw)


# Paths — LILBEE_DATA overrides the platform default
_data_env = _env("DATA", "")
_data_root = Path(_data_env) if _data_env else _default_data_dir()

DOCUMENTS_DIR = _data_root / "documents"
DATA_DIR = _data_root / "data"
LANCEDB_DIR = DATA_DIR / "lancedb"

# Ollama models — configurable via LILBEE_CHAT_MODEL / LILBEE_EMBEDDING_MODEL
CHAT_MODEL = _env("CHAT_MODEL", "mistral")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 768)

# Chunking — configurable via LILBEE_CHUNK_SIZE / LILBEE_CHUNK_OVERLAP
CHUNK_SIZE = _env_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)

# Embedding limits
MAX_EMBED_CHARS = _env_int("MAX_EMBED_CHARS", 2000)

# Retrieval
TOP_K = _env_int("TOP_K", 10)
MAX_DISTANCE = float(_env("MAX_DISTANCE", "1.5"))

# System prompt for RAG answers
SYSTEM_PROMPT = _env(
    "SYSTEM_PROMPT",
    "You are a helpful technical assistant. Answer questions using "
    "the provided context. Be specific — prefer exact numbers, part numbers, "
    "and measurements over vague references. Cite facts directly from the context. "
    "Do not make up information.",
)

# Directory ignore patterns for file discovery and copy
_DEFAULT_IGNORE_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        "venv",
        "build",
        "dist",
        "target",
        "vendor",
        "_build",
        "coverage",
        "htmlcov",
    }
)

_extra = _env("IGNORE", "")
IGNORE_DIRS = _DEFAULT_IGNORE_DIRS | frozenset(
    name.strip() for name in _extra.split(",") if name.strip()
)


def is_ignored_dir(name: str) -> bool:
    """Return True if a directory name should be skipped during traversal."""
    return name.startswith(".") or name in IGNORE_DIRS or name.endswith(".egg-info")


# LanceDB table names
CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"
