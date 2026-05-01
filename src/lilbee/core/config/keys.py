"""Names of the per-provider API key fields on the Config model.

Anything that wants to react to provider availability (e.g. the catalog
and model picker refreshing their API-model rows after a key is added)
keys off this set so the list does not drift from the canonical Config
declaration.
"""

from __future__ import annotations

PROVIDER_API_KEYS: frozenset[str] = frozenset(
    {
        "llm_api_key",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
    }
)
