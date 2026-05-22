"""Llama.cpp provider package.

The public surface is :class:`LlamaCppProvider`. Internal helpers
(``load_llama``, ``resolve_model_path``, ``read_gguf_metadata``, the
log dispatcher, batching primitives) live in their host submodules and
must be imported from there directly.
"""

from __future__ import annotations

# Importing this submodule installs a soft-fail patch on
# llama_cpp.llama_chat_format.Jinja2ChatFormatter so that GGUFs whose
# embedded chat template uses Jinja tags jinja2 can't compile (e.g.
# SmolLM3's `{% generation %}`) load successfully whenever the caller
# provides an explicit chat_format override.
from lilbee.providers.llama_cpp import jinja_resilience  # noqa: F401
from lilbee.providers.llama_cpp.provider import LlamaCppProvider

__all__ = ["LlamaCppProvider"]
