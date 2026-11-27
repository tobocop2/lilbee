"""Config keys whose change makes the fleet owner rebuild."""

from __future__ import annotations

from lilbee.core.config.keys import LOAD_AFFECTING_KEYS
from lilbee.providers.roles import ROLE_GATE_FIELD_TO_ROLE

FLEET_RELOAD_KEYS: frozenset[str] = LOAD_AFFECTING_KEYS | frozenset(ROLE_GATE_FIELD_TO_ROLE)
"""Settings whose change on a config reload makes the fleet owner rebuild.

A peer's plan view holds exactly these keys, so it waits only for a rebuild that comes.
This module sits above ``roles`` because ``roles`` must not import the config package,
which imports ``roles`` back through the model-reference chain.
"""

ENGINE_TUNING_KEYS: frozenset[str] = frozenset(
    {
        "num_ctx",
        "num_ctx_max",
        "chat_n_ctx_target",
        "kv_cache_type",
        "n_gpu_layers",
        "cpu_moe",
        "n_cpu_moe",
        "flash_attention",
    }
)
"""The fleet reload keys that change how a server runs, not what a role answers with.

A one-off command takes only these from config.toml at its retry, so it binds the
owner's restarted fleet. The other fleet reload keys pick a model, gate a role or
shape the library, and a one-off keeps them at their start value.
"""
