"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
"""

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lilbee import settings
from lilbee.platform import default_data_dir, env, env_float, env_int, env_int_optional

log = logging.getLogger(__name__)

DEFAULT_IGNORE_DIRS = frozenset(
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

CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"


class Config(BaseModel):
    """Runtime configuration — one singleton instance, mutated by CLI overrides."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_root: Path
    documents_dir: Path
    data_dir: Path
    lancedb_dir: Path
    chat_model: str
    embedding_model: str
    embedding_dim: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)
    max_embed_chars: int = Field(ge=1)
    top_k: int = Field(ge=1)
    max_distance: float = Field(ge=0.0)
    system_prompt: str
    ignore_dirs: frozenset[str]
    vision_model: str = ""
    vision_timeout: float = Field(default=120.0, ge=0.0)
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=7433, ge=1, le=65535)
    json_mode: bool = False
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k_sampling: int | None = Field(default=None, ge=1)
    repeat_penalty: float | None = Field(default=None, ge=0.0)
    num_ctx: int | None = Field(default=None, ge=1)
    seed: int | None = None

    def generation_options(self, **overrides: Any) -> dict[str, Any]:
        """Build Ollama generation options from config fields and overrides.

        Remaps ``top_k_sampling`` to Ollama's ``top_k`` key.
        Filters out ``None`` values so Ollama uses its model defaults.
        """
        mapping: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k_sampling,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
        }
        mapping.update(overrides)
        return {k: v for k, v in mapping.items() if v is not None}

    @classmethod
    def from_env(cls) -> "Config":
        """Build config from environment variables and settings file."""
        data_root = _resolve_data_root()
        chat_model = _load_chat_model(data_root)
        vision_model = _load_vision_model(data_root)
        vision_timeout = _parse_vision_timeout()

        extra = env("IGNORE", "")
        ignore_dirs = DEFAULT_IGNORE_DIRS | frozenset(
            name.strip() for name in extra.split(",") if name.strip()
        )

        return cls(
            data_root=data_root,
            documents_dir=data_root / "documents",
            data_dir=data_root / "data",
            lancedb_dir=data_root / "data" / "lancedb",
            chat_model=chat_model,
            embedding_model=env("EMBEDDING_MODEL", "nomic-embed-text"),
            embedding_dim=env_int("EMBEDDING_DIM", 768),
            chunk_size=env_int("CHUNK_SIZE", 512),
            chunk_overlap=env_int("CHUNK_OVERLAP", 100),
            max_embed_chars=env_int("MAX_EMBED_CHARS", 2000),
            top_k=env_int("TOP_K", 10),
            max_distance=float(env("MAX_DISTANCE", "0.7")),
            system_prompt=env(
                "SYSTEM_PROMPT",
                "You are a helpful technical assistant. Answer questions using "
                "the provided context. Be specific — prefer exact numbers, part numbers, "
                "and measurements over vague references. Cite facts directly from the context. "
                "Do not make up information.",
            ),
            ignore_dirs=ignore_dirs,
            vision_model=vision_model,
            vision_timeout=vision_timeout,
            server_host=env("SERVER_HOST", "127.0.0.1"),
            server_port=env_int("SERVER_PORT", 7433),
            temperature=env_float("TEMPERATURE"),
            top_p=env_float("TOP_P"),
            top_k_sampling=env_int_optional("TOP_K_SAMPLING"),
            repeat_penalty=env_float("REPEAT_PENALTY"),
            num_ctx=env_int_optional("NUM_CTX"),
            seed=env_int_optional("SEED"),
        )


def _resolve_data_root() -> Path:
    """Determine the data root: LILBEE_DATA env > local .lilbee/ > platform default."""
    data_env = env("DATA", "")
    if data_env:
        return Path(data_env)

    from lilbee.platform import find_local_root

    local = find_local_root()
    if local is not None:
        return local

    return default_data_dir()


def _load_chat_model(data_root: Path) -> str:
    """Resolve chat model: LILBEE_CHAT_MODEL env > persisted setting > default."""
    chat_model = env("CHAT_MODEL", "qwen3:8b")
    if "LILBEE_CHAT_MODEL" not in os.environ:
        try:
            saved = settings.get(data_root, "chat_model")
        except (ValueError, OSError):
            saved = None
        if saved:
            chat_model = saved
    return chat_model


def _load_vision_model(data_root: Path) -> str:
    """Resolve vision model: LILBEE_VISION_MODEL env > persisted setting > empty."""
    vision_model_env = os.environ.get("LILBEE_VISION_MODEL", "").strip()
    if vision_model_env:
        return vision_model_env
    try:
        return settings.get(data_root, "vision_model") or ""
    except (ValueError, OSError):
        return ""


def _parse_vision_timeout() -> float:
    """Parse LILBEE_VISION_TIMEOUT env var, returning default on invalid input."""
    raw = os.environ.get("LILBEE_VISION_TIMEOUT", "").strip()
    if not raw:
        return 120.0
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid LILBEE_VISION_TIMEOUT=%r, ignoring", raw)
        return 120.0


cfg = Config.from_env()
