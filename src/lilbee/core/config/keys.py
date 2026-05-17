"""Config-field key sets shared by the settings boundary and worker pool."""

from __future__ import annotations

# Per-provider entries mirror ``providers.sdk_backend.API_KEY_FIELDS``;
# ``llm_api_key`` is the extra generic-provider key the SDK fan-out doesn't
# consume but the settings boundary still publishes availability for.
# Core cannot import from providers, so the per-provider half lives in
# parallel; add new providers here AND in ``API_KEY_FIELDS`` together.
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

# Settings baked into Llama() at load time, or whose change picks a
# different model file. Sampling params are read per-call and excluded.
LOAD_AFFECTING_KEYS: frozenset[str] = frozenset(
    {
        "num_ctx",
        "chat_model",
        "embedding_model",
        "vision_model",
        "reranker_model",
    }
)

# Subset of LOAD_AFFECTING_KEYS whose change is observed by the worker on
# the next per-call ``request.model`` (chat / vision workers check the path
# in ``_ensure_loaded`` and reload in place). For these the pool does not
# need to drop the role; the next call swaps the model inside the live
# worker, saving the 1-3 s spawn cost.
PER_CALL_RELOADABLE_KEYS: frozenset[str] = frozenset({"chat_model", "vision_model"})
