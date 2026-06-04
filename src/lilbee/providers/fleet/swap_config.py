"""Generate a llama-swap config that keeps every fleet role co-resident.

See docs/architecture.md (llama-swap) for the supervisor/proxy design.
"""

from __future__ import annotations

import json
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.providers.fleet.fleet import InstanceLaunch

# One group holds every role with swap disabled, so llama-swap brings them all up
# and never evicts one to load another (the co-residency the fleet needs).
_GROUP_NAME = "lilbee"
_START_PORT = 5800
# Generous cold-load ceiling: a multi-hundred-GB giant can take minutes to map.
_HEALTH_CHECK_TIMEOUT_S = 600
_LOG_LEVEL = "info"
# llama-swap substitutes ${PORT} per member before running the command.
_PORT_MACRO = "${PORT}"
_PROXY_URL = "http://localhost:${PORT}"
_PORT_FLAG = "--port"
_TTL_KEEP = 0  # never time a member out; the group keeps it resident

# llama-swap config keys.
_KEY_START_PORT = "startPort"
_KEY_HEALTH_TIMEOUT = "healthCheckTimeout"
_KEY_LOG_LEVEL = "logLevel"
_KEY_MODELS = "models"
_KEY_CMD = "cmd"
_KEY_PROXY = "proxy"
_KEY_TTL = "ttl"
_KEY_ENV = "env"
_KEY_GROUPS = "groups"
_KEY_SWAP = "swap"
_KEY_EXCLUSIVE = "exclusive"
_KEY_PERSISTENT = "persistent"
_KEY_MEMBERS = "members"


def build_swap_config(launches: list[InstanceLaunch]) -> str:
    """Render a llama-swap config (JSON, which is valid YAML) for *launches*.

    Each role becomes a model whose id is the role name and whose command is the
    role's llama-server argv plus llama-swap's ${PORT} macro; one ``swap: false``
    group holds them all co-resident behind the single OpenAI endpoint.
    """
    models: dict[str, object] = {}
    for launch in launches:
        entry: dict[str, object] = {
            _KEY_CMD: _command_line(launch.argv),
            _KEY_PROXY: _PROXY_URL,
            _KEY_TTL: _TTL_KEEP,
        }
        if launch.env_overrides:
            entry[_KEY_ENV] = [f"{key}={value}" for key, value in launch.env_overrides.items()]
        models[launch.role.value] = entry
    config: dict[str, object] = {
        _KEY_START_PORT: _START_PORT,
        _KEY_HEALTH_TIMEOUT: _HEALTH_CHECK_TIMEOUT_S,
        _KEY_LOG_LEVEL: _LOG_LEVEL,
        _KEY_MODELS: models,
        _KEY_GROUPS: {
            _GROUP_NAME: {
                _KEY_SWAP: False,
                _KEY_EXCLUSIVE: False,
                _KEY_PERSISTENT: True,
                _KEY_MEMBERS: [launch.role.value for launch in launches],
            }
        },
    }
    return json.dumps(config, indent=2)


def _command_line(argv: list[str]) -> str:
    """Shell command for a member: the role argv plus the ${PORT} macro.

    Argv is shell-quoted so a spaced model path survives; the macro is appended
    literally so llama-swap substitutes the claimed port.
    """
    return f"{shlex.join(argv)} {_PORT_FLAG} {_PORT_MACRO}"
