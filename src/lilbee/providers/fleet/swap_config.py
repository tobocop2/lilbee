"""Generate a llama-swap config that keeps every fleet role co-resident.

See docs/architecture.md (llama-swap) for the supervisor/proxy design.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lilbee.providers.fleet.launch import InstanceLaunch

# One group holds every role with swap disabled, so llama-swap brings them all up
# and never evicts one to load another (the co-residency the fleet needs).
_GROUP_NAME = "lilbee"
# Cold-load ceiling floor; the heaviest member's weights scale it up from here.
_HEALTH_CHECK_TIMEOUT_FLOOR_S = 600
# Conservative cold-load disk rate; a slow network volume streams well under this.
_COLD_LOAD_BYTES_PER_S = 150 * 1024 * 1024
_LOG_LEVEL = "info"
# Matches the --host every server argv binds (adapters); "localhost" would have
# llama-swap dial [::1] first, where another process could hold the same port.
_PROXY_URL_TEMPLATE = "http://127.0.0.1:{port}"
# Shared with the swap manager, whose orphan-server sweep matches this flag's
# value in survivor cmdlines.
PORT_FLAG = "--port"
_TTL_KEEP = 0  # never time a member out; the group keeps it resident

# llama-swap config keys.
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


def build_swap_config(launches: list[InstanceLaunch], member_ports: Mapping[str, int]) -> str:
    """Render a llama-swap config (JSON, which is valid YAML) for *launches*.

    Each role becomes a model whose id is the role name and whose command is the
    role's llama-server argv plus the explicit port from *member_ports*; one
    ``swap: false`` group holds them all co-resident behind the single OpenAI
    endpoint. Ports are allocated fresh per start (never llama-swap's fixed
    ``startPort`` range) so a previous instance's lingering server can't collide
    with the new fleet's bind.
    """
    models: dict[str, object] = {}
    for launch in launches:
        port = member_ports[launch.model_id]
        entry: dict[str, object] = {
            _KEY_CMD: _command_line(launch.argv, port),
            _KEY_PROXY: _PROXY_URL_TEMPLATE.format(port=port),
            _KEY_TTL: _TTL_KEEP,
        }
        if launch.env_overrides:
            entry[_KEY_ENV] = [f"{key}={value}" for key, value in launch.env_overrides.items()]
        models[launch.model_id] = entry
    config: dict[str, object] = {
        _KEY_HEALTH_TIMEOUT: _health_check_timeout_s(launches),
        _KEY_LOG_LEVEL: _LOG_LEVEL,
        _KEY_MODELS: models,
        _KEY_GROUPS: {
            _GROUP_NAME: {
                _KEY_SWAP: False,
                _KEY_EXCLUSIVE: False,
                _KEY_PERSISTENT: True,
                _KEY_MEMBERS: [launch.model_id for launch in launches],
            }
        },
    }
    return json.dumps(config, indent=2)


def cold_load_timeout_s(weights_bytes: int) -> int:
    """Cold-load ceiling for one member's weights at a conservative disk rate, floored.

    The single source of the scaling formula: llama-swap's health-check timeout
    and the provider's per-client request timeout both derive from it, so a model
    whose load llama-swap would wait out can never time out the client first.
    """
    return max(_HEALTH_CHECK_TIMEOUT_FLOOR_S, weights_bytes // _COLD_LOAD_BYTES_PER_S)


def _health_check_timeout_s(launches: list[InstanceLaunch]) -> int:
    """Cold-load ceiling of the heaviest member; the timeout is proxy-global in
    llama-swap, so the slowest possible load sets it."""
    heaviest = max((launch.weights_bytes for launch in launches), default=0)
    return cold_load_timeout_s(heaviest)


def _command_line(argv: list[str], port: int) -> str:
    """Shell command for a member: the role argv plus its explicit port.

    Quoting must match how llama-swap splits the command back into argv: MS
    rules on Windows (POSIX single quotes would stay literal in the paths and
    the spawn fails with "file does not exist"), POSIX everywhere else.
    """
    if sys.platform == "win32":
        rendered = subprocess.list2cmdline(argv)
    else:
        rendered = shlex.join(argv)
    return f"{rendered} {PORT_FLAG} {port}"
