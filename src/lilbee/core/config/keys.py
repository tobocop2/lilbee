"""Config-field key sets shared by the settings boundary and worker pool."""

from __future__ import annotations

# Keep in sync with ``providers.sdk_backend.API_KEY_FIELDS``.
PROVIDER_API_KEYS: frozenset[str] = frozenset(
    {
        "llm_api_key",
        "openrouter_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "mistral_api_key",
        "deepseek_api_key",
    }
)

# Settings whose value is baked into a loaded model and only takes effect
# after the worker reloads.
LOAD_AFFECTING_KEYS: frozenset[str] = frozenset(
    {
        "num_ctx",
        "num_ctx_max",
        "chat_n_ctx_target",
        "kv_cache_type",
        "chat_model",
        "embedding_model",
        "vision_model",
        "reranker_model",
    }
)

# Subset of LOAD_AFFECTING_KEYS the worker can swap in place on the next call.
PER_CALL_RELOADABLE_KEYS: frozenset[str] = frozenset({"chat_model", "vision_model"})

# Writes here require reconstructing the services singleton.
PROVIDER_SWITCHING_KEYS: frozenset[str] = frozenset({"llm_provider"})
