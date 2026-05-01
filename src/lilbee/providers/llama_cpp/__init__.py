"""Llama.cpp provider package.

The public surface is :class:`LlamaCppProvider`. Internal helpers
(``load_llama``, ``resolve_model_path``, ``read_gguf_metadata``, the
log dispatcher, batching primitives) live in their host submodules and
must be imported from there directly.
"""

from __future__ import annotations

from lilbee.providers.llama_cpp.provider import LlamaCppProvider

__all__ = ["LlamaCppProvider"]
