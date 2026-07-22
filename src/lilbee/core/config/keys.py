"""Config-field key sets shared by the settings boundary and the engine."""

from __future__ import annotations

# API-key cfg field names: keep in sync with ``providers.sdk_backend.PROVIDER_KEYS``.
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
        # Baked into the llama-server argv (--n-gpu-layers, --cpu-moe, --n-cpu-moe,
        # --flash-attn), so a change must rebuild the engine; the bind contract
        # (role/model + pin) would otherwise adopt a running engine's old offload,
        # cache, or flash-attention flags unchanged.
        "n_gpu_layers",
        "cpu_moe",
        "n_cpu_moe",
        "flash_attention",
        "chat_model",
        "embedding_model",
        "vision_model",
        "reranker_model",
        "reranker_type",
    }
)

# Writes here require reconstructing the services singleton.
PROVIDER_SWITCHING_KEYS: frozenset[str] = frozenset({"llm_provider"})

# Settings that change which physical devices an engine launches on. Part of the
# cross-process engine pin, so a process with a different placement binds its own
# engine rather than silently adopting the incumbent's GPUs. Kept out of
# LOAD_AFFECTING_KEYS so they do not change the settings reload path (gpu_devices
# takes effect on restart), only which engines are shareable.
PLACEMENT_PIN_KEYS: frozenset[str] = frozenset({"placement", "gpu_devices"})
