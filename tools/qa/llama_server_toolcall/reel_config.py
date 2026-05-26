"""Per-model demo config: real display names + the prompt each model is asked.

The prompts are about lilbee's *retrieval* stack (chunking, embedding, search,
reranking, query expansion, concept graph) -- the stable value-prop code, not
the in-flux model-families parser. They are scaled to the model: small models
get one focused, specific question; the big/giant models get open-ended
practical dev questions. Each prompt is validated by reading the actual
`opencode run` transcript before it is recorded, and iterated until the answer
is genuinely good.
"""

from __future__ import annotations

import re
from pathlib import Path

# Quant / shard markers; everything from the first one on is noise, not the name.
_QUANT = re.compile(r"[.\-](?:Q\d+_[K0-9](?:_[A-Z])?|IQ\d+_[A-Z]+|mxfp4|BF16|F16|F32)\b")


def display_name(gguf: str) -> str:
    """Real model name for a GGUF path, e.g. Qwen3-4B-Q4_K_M.gguf -> Qwen3-4B."""
    return _QUANT.split(Path(gguf).name)[0]


# family -> prompt. Reel order is defined in record_reel.py.
PROMPTS: dict[str, str] = {
    # --- small models: one focused, specific retrieval question ---
    "qwen3": (
        "How does lilbee split a document into chunks before indexing it? "
        "Look it up in the code, explain it briefly, and cite the file."
    ),
    "gemma4": (
        "Search lilbee's code for how it creates text embeddings, then name the "
        "file in one sentence."
    ),
    "llama3": (
        "When I search lilbee, how does it find the chunks that match my query? "
        "Explain the search step and cite the file."
    ),
    "mistral-nemo": (
        "Does lilbee rerank its search results? Find the reranking code, explain "
        "what it does, and cite the file."
    ),
    "hermes": (
        "How does lilbee expand or rewrite my query before searching? "
        "Look it up, explain it, and cite the file."
    ),
    "functionary": (
        "How does lilbee remove duplicate results from a search? "
        "Find the code and summarize how it works."
    ),
    # --- mid: code-flavored for the coder model ---
    "qwen3-coder": (
        "Search lilbee for how it chunks source code, then explain in two or three "
        "sentences how code chunking differs from prose chunking, citing the file."
    ),
    # --- large single-GPU: practical, multi-stage dev questions ---
    "gpt-oss": (
        "I'm evaluating lilbee for retrieval over a mixed code-and-docs corpus. "
        "Walk me through its pipeline from a raw file to a search result -- "
        "chunking, embedding, vector search, and reranking -- and cite the key "
        "file for each stage."
    ),
    "glm-air": (
        "Explain how lilbee builds a concept graph from an indexed corpus and how "
        "it uses it at query time. Reference the code."
    ),
    # --- giants: open-ended, impressive practical tasks ---
    "qwen3-235b": (
        "I'm evaluating lilbee as the retrieval layer for an internal dev "
        "assistant. Give me an end-to-end walkthrough of its retrieval pipeline "
        "-- chunking, embedding, vector search, reranking, and query expansion -- "
        "with file citations."
    ),
    "minimax-m2": (
        "Give me a technical overview of how lilbee turns a question into a "
        "ranked, cited answer over an indexed corpus -- the full retrieval path "
        "-- and point out where I'd extend it to add a new ranking signal. Cite "
        "the relevant files."
    ),
    "glm-4.6": (
        "Walk me through lilbee's retrieval architecture end to end and explain "
        "how query expansion and reranking improve result quality, with file "
        "citations. Then suggest where I'd add a caching layer."
    ),
}
