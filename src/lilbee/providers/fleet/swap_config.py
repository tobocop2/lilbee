"""Generate one swap group's llama-swap config.

Each group runs behind its own llama-swap process with its own config, so a
reload of one group never touches another's loaded servers. See
docs/internals/engine.md (llama-swap) for the supervisor/proxy design.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

from lilbee.data.offload import max_workers
from lilbee.providers.fleet.engines.base import PORT_SLOT
from lilbee.providers.fleet.readback import MEMORY_FLAG, engine_log_env
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lilbee.providers.fleet.launch import InstanceLaunch

# Prefix of every group name. Without swap one group holds the process's members
# co-resident (a role's replicas); with swap each residency class is a group.
_GROUP_NAME = "lilbee"
# Cold-load ceiling floor per role; the member's weights scale it up from here.
# An embedder's weights stream in seconds, so its floor covers only process start
# and backend init: waiting out a chat-sized ten minutes on a crash-looping or
# wedged embed server holds every ingest caller for that long.
_COLD_LOAD_FLOOR_S: dict[WorkerRole, int] = {
    WorkerRole.CHAT: 600,
    WorkerRole.VISION: 600,
    WorkerRole.RERANK: 600,
    WorkerRole.EMBED: 120,
}
# Conservative cold-load disk rate; a slow network volume streams well under this.
_COLD_LOAD_BYTES_PER_S = 150 * 1024 * 1024
_LOG_LEVEL = "info"
# The streams llama-swap copies to its stdout, which lilbee writes to the group
# log. The upstream stream is the engines' own output, and it is one setting for
# the whole llama-swap process. It goes in only when a member's engine writes no
# log file of its own (TEI, MLX, OVMS and the other engines that read no log
# variables), and only for a router that reads a list (ROUTER_FLOOR). A
# llama-server also prints every line of its own log to its console, so in a group
# that also holds such an engine, each llama member's output is in both files, at
# trace verbosity for a member that reports its load. A group of llama-servers
# alone keeps upstream off. The http stream (one line per request) never goes in.
_STDOUT_PROXY = "proxy"
_STDOUT_PROXY_AND_UPSTREAM = "proxy,upstream"
# The first llama-swap release that reads a list of streams in logToStdout
# (ParseLogToStdout in internal/config/config.go). v258 and older accept only
# proxy, upstream, both or none, and refuse a list at load, so an older router
# gets proxy alone. No single value copies upstream on both sides: from v259 on,
# "both" also turns on the http stream.
ROUTER_FLOOR = 259
# Matches the --host every server argv binds (adapters); "localhost" would have
# llama-swap dial [::1] first, where another process could hold the same port.
_PROXY_URL_TEMPLATE = "http://127.0.0.1:{port}"
# Shared with the swap manager, whose orphan-server sweep matches this flag's
# value in survivor cmdlines.
PORT_FLAG = "--port"

# llama-swap refuses a request with HTTP 429 once this many are in flight for
# one model (its default is 10), and a request held through a cold load counts.
# One engine serves every lilbee process the user runs (the TUI, serve with its
# HTTP clients, MCP, an ingest run), and llama-server queues requests past its
# slots by itself, so the limit only has to stop runaway load. It covers this
# many processes, each at its own ceiling: the larger of the slots and the
# ingest pool, whose every thread can hold one embed or OCR request.
_CONCURRENCY_PROCESSES = 4
# Part of the engine pin, so a peer never binds an engine launched under another
# contract. Raise it when a launch rule changes. 1: every model carries
# concurrencyLimit. 2: an equal split across cards runs tensor parallel. 3: a swap
# process keeps a residency class loaded together. 4: the llama-swap log holds the
# output of every engine that writes no log of its own, and a --memory launch
# writes its own log at the default verbosity.
SWAP_CONFIG_REVISION = 4

# llama-swap config keys.
_KEY_HEALTH_TIMEOUT = "healthCheckTimeout"
_KEY_LOG_LEVEL = "logLevel"
_KEY_LOG_TO_STDOUT = "logToStdout"
_KEY_MODELS = "models"
_KEY_CMD = "cmd"
_KEY_PROXY = "proxy"
_KEY_TTL = "ttl"
_KEY_CONCURRENCY_LIMIT = "concurrencyLimit"
_KEY_ENV = "env"
_KEY_GROUPS = "groups"
_KEY_SWAP = "swap"
_KEY_EXCLUSIVE = "exclusive"
_KEY_PERSISTENT = "persistent"
_KEY_MEMBERS = "members"


def build_swap_config(
    launches: list[InstanceLaunch],
    member_ports: Mapping[str, int],
    *,
    swap: bool = False,
    ttl_seconds: int = 0,
    engine_logs: Mapping[str, Path],
    router_reads_streams: bool = True,
) -> str:
    """Render a llama-swap config (JSON, which is valid YAML) for *launches*.

    Each launch becomes a model whose id is its replica model id and whose
    command is the llama-server argv plus the explicit port from *member_ports*;
    one group holds them behind this group's proxy endpoint. ``swap`` makes one
    group of each residency class (``InstanceLaunch.swap_class``): members of a
    class stay loaded together, and loading a member evicts every other class;
    the default keeps all members co-resident. ``ttl_seconds`` is llama-swap's idle unload
    timer per member; 0 keeps weights loaded forever. Ports are allocated fresh per start (never
    llama-swap's fixed ``startPort`` range) so a previous instance's lingering
    server can't collide with the new fleet's bind.

    ``engine_logs`` maps a model id to the file its engine writes its own log to
    (:func:`lilbee.providers.fleet.readback.engine_log_of`); the caller passes it,
    so a config never comes from a log map that production does not build. A
    launch without ``--memory`` logs at the verbosity that reports what it
    allocated: for an engine without ``GET /memory`` that file is the only
    place those numbers exist, and reading them back is what tells the planner whether its estimate
    held. A ``--memory`` launch serves the numbers over HTTP and logs at the
    default verbosity. A member with no entry writes no file, so llama-swap
    copies its output into the group log instead; without that copy the output
    of a member that fails to start is lost when the group stops. llama-swap
    copies that output from :data:`ROUTER_FLOOR` on. ``router_reads_streams``
    says the router is that release or later; when it is false, the config
    writes the proxy stream alone, which every release reads.
    """
    models: dict[str, object] = {}
    for launch in launches:
        port = member_ports[launch.model_id]
        entry: dict[str, object] = {
            _KEY_CMD: _command_line(launch.argv, port),
            _KEY_PROXY: _PROXY_URL_TEMPLATE.format(port=port),
            _KEY_TTL: ttl_seconds,
            _KEY_CONCURRENCY_LIMIT: _concurrency_limit(launch),
        }
        env = dict(launch.env_overrides)
        log_path = engine_logs.get(launch.model_id)
        if log_path is not None:
            env.update(engine_log_env(log_path, load_report=MEMORY_FLAG not in launch.argv))
        if env:
            entry[_KEY_ENV] = [f"{key}={value}" for key, value in env.items()]
        entry.update(launch.swap_extras)
        models[launch.model_id] = entry
    config: dict[str, object] = {
        _KEY_HEALTH_TIMEOUT: health_check_timeout_s(launches),
        _KEY_LOG_LEVEL: _LOG_LEVEL,
        _KEY_LOG_TO_STDOUT: (
            _STDOUT_PROXY_AND_UPSTREAM
            if router_reads_streams
            and any(launch.model_id not in engine_logs for launch in launches)
            else _STDOUT_PROXY
        ),
        _KEY_MODELS: models,
        _KEY_GROUPS: _evicting_groups(launches) if swap else _resident_group(launches),
    }
    return json.dumps(config, indent=2)


def _resident_group(launches: list[InstanceLaunch]) -> dict[str, object]:
    """One group whose members stay loaded together."""
    return {
        _GROUP_NAME: {
            _KEY_SWAP: False,
            _KEY_EXCLUSIVE: False,
            _KEY_PERSISTENT: True,
            _KEY_MEMBERS: [launch.model_id for launch in launches],
        }
    }


def _evicting_groups(launches: list[InstanceLaunch]) -> dict[str, object]:
    """One group per residency class; loading a member unloads every other group."""
    members: dict[str, list[str]] = {}
    for launch in launches:
        members.setdefault(launch.swap_class or launch.model_id, []).append(launch.model_id)
    return {
        f"{_GROUP_NAME}-{name}": {
            _KEY_SWAP: False,
            _KEY_EXCLUSIVE: True,
            _KEY_PERSISTENT: False,
            _KEY_MEMBERS: ids,
        }
        for name, ids in members.items()
    }


def _concurrency_limit(launch: InstanceLaunch) -> int:
    """Requests llama-swap lets in flight for *launch* before it answers 429."""
    return _CONCURRENCY_PROCESSES * max(launch.slots, max_workers())


def cold_load_timeout_s(weights_bytes: int, role: WorkerRole) -> int:
    """Cold-load ceiling for one member's weights at a conservative disk rate, floored per role.

    The single source of the scaling formula: llama-swap's health-check timeout
    and the provider's per-client request timeout both derive from it, so a model
    whose load llama-swap would wait out can never time out the client first.
    """
    return max(_COLD_LOAD_FLOOR_S[role], weights_bytes // _COLD_LOAD_BYTES_PER_S)


def health_check_timeout_s(launches: list[InstanceLaunch]) -> int:
    """Slowest member's cold-load ceiling; the timeout is proxy-global in
    llama-swap, so the slowest possible load sets it."""
    return max(
        (cold_load_timeout_s(launch.weights_bytes, launch.role) for launch in launches),
        default=max(_COLD_LOAD_FLOOR_S.values()),
    )


def _command_line(argv: list[str], port: int) -> str:
    """Shell command for a member: the role argv plus its explicit port.

    Quoting must match how llama-swap splits the command back into argv: MS
    rules on Windows (POSIX single quotes would stay literal in the paths and
    the spawn fails with "file does not exist"), POSIX everywhere else.
    """
    slotted = PORT_SLOT in argv
    full = [str(port) if arg == PORT_SLOT else arg for arg in argv]
    if not slotted:
        full += [PORT_FLAG, str(port)]
    return subprocess.list2cmdline(full) if sys.platform == "win32" else shlex.join(full)
