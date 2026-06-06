"""HuggingFace ref helpers: parse and format ``<org>/<repo>/<file>.gguf`` strings."""

from __future__ import annotations

# A native GGUF ref ``<org>/<repo>/<file>.gguf`` has at least two ``/`` separators;
# the filename may add more when a quant lives in a repo subdir (``Q4_K_M/...``).
NATIVE_GGUF_REF_MIN_SLASHES = 2


def hf_repo_from_ref(ref: str) -> str:
    """Return the ``<org>/<repo>`` portion of a native GGUF ref.

    Native GGUF refs have the form ``<org>/<repo>/<filename>.gguf``, where the
    filename may itself include repo subdirectories (unsloth stores quants under
    e.g. ``Q4_K_M/...gguf``). The repo is always the first two segments.
    Provider-prefixed refs (``openai/gpt-4``, ``ollama/llama3:8b``) and bare
    repos lack the ``.gguf`` suffix and are returned unchanged.
    """
    if ref.endswith(".gguf") and ref.count("/") >= NATIVE_GGUF_REF_MIN_SLASHES:
        return "/".join(ref.split("/")[:NATIVE_GGUF_REF_MIN_SLASHES])
    return ref


def format_native_gguf_ref(hf_repo: str, gguf_filename: str) -> str:
    """Render the canonical ``<hf_repo>/<gguf_filename>`` native GGUF ref."""
    return f"{hf_repo}/{gguf_filename}"
