"""Local OpenAI-compatible model servers (Ollama, LM Studio).

One :class:`LocalServerSpec` per server (``ollama``, ``lm_studio``) describes
its routing key, litellm wire prefix, default base URL, identifying URL
substrings, model source, and which catalog operations it supports. Imports
stay light for the routing layer; the HTTP behaviour (discovery, listing)
lives in the modelhub layer, dispatched off the spec.
"""

from lilbee.providers.local_servers.lm_studio import LM_STUDIO
from lilbee.providers.local_servers.ollama import OLLAMA
from lilbee.providers.local_servers.registry import (
    LOCAL_SERVER_KEYS,
    LOCAL_SERVERS,
    canonical_local_ref,
    detect_local_server,
    local_server_for_key,
    local_server_for_label,
    openai_models_url,
)
from lilbee.providers.local_servers.spec import LocalServerSpec

__all__ = [
    "LM_STUDIO",
    "LOCAL_SERVERS",
    "LOCAL_SERVER_KEYS",
    "OLLAMA",
    "LocalServerSpec",
    "canonical_local_ref",
    "detect_local_server",
    "local_server_for_key",
    "local_server_for_label",
    "openai_models_url",
]
