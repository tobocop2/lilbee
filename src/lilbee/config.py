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

# Retrieval
TOP_K = _env_int("TOP_K", 10)

# LanceDB table names
CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"
