"""Application configuration for lilbee.

All settings can be overridden via environment variables prefixed with LILBEE_.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lilbee import settings
from lilbee.platform import default_data_dir, env, env_float, env_int, env_int_optional

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


@dataclass
class Config:
    """Runtime configuration — one singleton instance, mutated by CLI overrides."""

    data_root: Path
    documents_dir: Path
    data_dir: Path
    lancedb_dir: Path
    chat_model: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    max_embed_chars: int
    top_k: int
    max_distance: float
    system_prompt: str
    ignore_dirs: frozenset[str]
    vision_model: str = ""
    vision_timeout: float = 120.0  # seconds per page
    server_host: str = "127.0.0.1"
    server_port: int = 7433
    json_mode: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k_sampling: int | None = None
    repeat_penalty: float | None = None
    num_ctx: int | None = None
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
        data_env = env("DATA", "")
        data_root = Path(data_env) if data_env else default_data_dir()

        if not data_env:
            from lilbee.platform import find_local_root

            local = find_local_root()
            if local is not None:
                data_root = local

        chat_model = env("CHAT_MODEL", "qwen3:8b")
        if "LILBEE_CHAT_MODEL" not in os.environ:
            try:
                saved = settings.get(data_root, "chat_model")
            except Exception:
                saved = None
            if saved:
                chat_model = saved

        vision_model_env = os.environ.get("LILBEE_VISION_MODEL", "").strip()
        if vision_model_env:
            vision_model: str = vision_model_env
        else:
            try:
                vision_model = settings.get(data_root, "vision_model") or ""
            except Exception:
                vision_model = ""

        vision_timeout_raw = os.environ.get("LILBEE_VISION_TIMEOUT", "").strip()
        try:
            vision_timeout: float = float(vision_timeout_raw) if vision_timeout_raw else 120.0
        except ValueError:
            import logging

            logging.getLogger(__name__).warning(
                "Invalid LILBEE_VISION_TIMEOUT=%r, ignoring", vision_timeout_raw
            )
            vision_timeout = 120.0

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


cfg = Config.from_env()
