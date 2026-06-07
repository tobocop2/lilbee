"""HuggingFace ref helpers: parse and format ``<org>/<repo>/<file>.gguf`` strings."""

from __future__ import annotations

GGUF_GLOB = "*.gguf"

_QUANT_PREFERENCE = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K", "Q3_K_M")


def pick_best_gguf(filenames: list[str]) -> str:
    """Pick the best GGUF file by quantization preference."""
    for quant in _QUANT_PREFERENCE:
        for f in filenames:
            if quant in f:
                return f
    return filenames[0]


def is_bare_hf_repo(ref: str) -> bool:
    """True if *ref* has the bare ``<org>/<repo>`` shape (no filename segment)."""
    return ref.count("/") == 1 and not ref.endswith(".gguf")


def hf_repo_from_ref(ref: str) -> str:
    """Return the ``<org>/<repo>`` portion of a native GGUF ref.

    Native GGUF refs have the form ``<org>/<repo>/<filename>.gguf``; the
    repo is the prefix before the final slash. Provider-prefixed refs
    (``openai/gpt-4``, ``ollama/llama3:8b``) and bare repos lack the
    ``.gguf`` suffix and are returned unchanged.
    """
    if ref.endswith(".gguf") and "/" in ref:
        return ref.rsplit("/", 1)[0]
    return ref


def format_native_gguf_ref(hf_repo: str, gguf_filename: str) -> str:
    """Render the canonical ``<hf_repo>/<gguf_filename>`` native GGUF ref."""
    return f"{hf_repo}/{gguf_filename}"
