"""Supervise the single llama-swap process that fronts every fleet role.

llama-swap owns each role's llama-server lifecycle; this manages the one proxy
process and exposes its endpoint and readiness. See docs/internals/engine.md.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import httpx
import psutil

from lilbee.core.health_warnings import HealthWarning, WarningCode
from lilbee.parent_monitor import boot_identity, linux_age_s, linux_start_stamp, macos_start_stamp
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.binary import (
    ROUTER_GO_INSTALL,
    engine_pin,
    resolve_llama_swap,
    router_release,
)
from lilbee.providers.fleet.build_use import BuildHolds, builds_run_by
from lilbee.providers.fleet.child_guard import release_death_pipe, spawn_bound_child
from lilbee.providers.fleet.engine_store import engines_root
from lilbee.providers.fleet.gpu_env import engine_environ
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import InstanceLaunch, role_model_prefix
from lilbee.providers.fleet.planning import clear_ctx_downshift, log_role_launches
from lilbee.providers.fleet.readback import (
    MEMORY_FLAG,
    check_launch,
    check_memory_report,
    engine_log_of,
    report_missing_log,
)
from lilbee.providers.fleet.swap_config import (
    CONCURRENCY_LIMIT,
    PORT_FLAG,
    ROUTER_FLOOR,
    build_swap_config,
    health_check_timeout_s,
)
from lilbee.runtime.engine_lock import clear_keep_warm, reaper_running

if TYPE_CHECKING:
    from lilbee.providers.fleet.engines.base import EngineAdapter, ServerProcess
    from lilbee.providers.roles import WorkerRole

if sys.platform == "win32":  # pragma: no cover - Windows has no resource module
    resource = None
else:
    import resource

log = logging.getLogger(__name__)

# llama-swap passes its open-file limit to the engines it spawns, and llama-server
# holds one socket per admitted request. The extra 64 cover the files each process
# opens for itself. macOS starts a process with a soft limit of 256.
_OPEN_FILE_FLOOR = CONCURRENCY_LIMIT + 64


def _raise_open_file_limit() -> None:
    """Raise this process's soft open-file limit to the floor, never past the hard limit.

    llama-swap and its engines inherit the limit. A hard limit under the floor is
    used as it is, with one warning. A platform with no limit to set is left alone.
    A refused change logs one warning and the start goes on.
    """
    if resource is None:
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= _OPEN_FILE_FLOOR:
        return
    target = _OPEN_FILE_FLOOR
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target < _OPEN_FILE_FLOOR:
        log.warning(
            "The open-file limit is capped at %d, under the %d the engine can use at full load.",
            hard,
            _OPEN_FILE_FLOOR,
        )
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (OSError, ValueError) as exc:
            log.warning("The open-file limit stays at %d, not the %d wanted: %s", soft, target, exc)


_HOST = "127.0.0.1"
# Where a router older than ROUTER_FLOOR is replaced from, beside ROUTER_GO_INSTALL.
_ROUTER_RELEASES_URL = "https://github.com/mostlygeek/llama-swap/releases"
# One llama-swap per swap group: the group name lands in the config filename so
# each group's processes are identified (and stopped) by their own config path,
# and a placement change can restart one group without touching the others.
# The writer pid segment is uniqueness, not ownership: the build lock ensures
# one builder per engine dir, and reaping cleans dead writers' leftovers.
_CONFIG_FILENAME_TEMPLATE = "llama-swap-{group}.{pid}.json"
_CONFIG_FILE_GLOB = "llama-swap-*.json"
# llama-swap's own stdout/stderr (its HTTP access log) is captured to a file in a
# ``logs/`` dir inside the engine dir, which is the machine slot rather than any
# one lilbee's data root, so the log sits beside the engine it belongs to instead
# of beside server.log. Capturing it at all, rather than inheriting the parent's
# fd, is because a TUI or CLI parent owns the terminal and an inherited fd would bleed
# llama-swap's request log onto the screen and corrupt the render. The output of an
# engine that writes no log file of its own is copied in too (swap_config).
_LOGS_SUBDIR = "logs"
_LOG_FILENAME_TEMPLATE = "llama-swap-{group}.log"
# Each writer's state file records its swap's pid/pgid so a later start can
# stop a dead or unhealthy engine. Health, not ownership, decides sparing.
_STATE_FILENAME_PREFIX = "llama-swap.state."
_STATE_FILENAME_SUFFIX = ".json"
# Also matches the legacy single shared state file ("llama-swap.state.json").
_STATE_FILE_GLOB = f"{_STATE_FILENAME_PREFIX}*"
_STATE_KEY_PID = "pid"
_STATE_KEY_PGID = "pgid"
_STATE_KEY_CREATED_AT = "created_at"
_STATE_KEY_BOOT_ID = "boot_id"
_STATE_KEY_START_TICKS = "start_ticks"
# macOS only: {"boot": boot session, "abstime": start on the mach clock}. A new key, so a
# lilbee from before it reads such a record by its create time.
_STATE_KEY_MACH_START = "mach_start"
_MACH_START_BOOT = "boot"
_MACH_START_ABSTIME = "abstime"
_STATE_KEY_NAME = "name"
_STATE_KEY_MEMBER_PORTS = "member_ports"
_STATE_KEY_PROXY_PORT = "proxy_port"
_STATE_KEY_LAUNCHES = "launches"
_STATE_KEY_ENGINE_PIN = "engine_pin"
_STATE_KEY_CONFIG_ROOT = "config_root"
_STATE_KEY_FLEET_VIEW = "fleet_view"
_STATE_KEY_PLANNED_VIEW = "planned_view"
_STATE_KEY_FOLLOWED_KEYS = "followed_keys"
_STATE_KEY_READY_AT = "ready_at"
_STATE_KEY_STARTED_MONO = "started_mono"
_STATE_KEY_STARTED_BOOT = "started_boot"
_STATE_KEY_SWAP_BINARY = "swap_binary"
# Atomic state writes: the dot prefix keeps half-written tmp files out of the
# reap scan's glob.
_STATE_TMP_PREFIX = "."
_STATE_TMP_SUFFIX = ".tmp"
# Pid reuse guard: a live process at a recorded pid whose create time differs
# from the recorded one by more than this is a different process.
_CREATE_TIME_TOLERANCE_S = 1.0
_LLAMA_SWAP_PROCESS_NAME = "llama-swap"
_EXE_SUFFIX = ".exe"
# A script server shows the interpreter's name; this prefix finds python and python3.12.
_INTERPRETER_PREFIX = "python"
_CONFIG_FLAG = "-config"
_LISTEN_FLAG = "-listen"
_HEALTH_PATH = "/health"
RUNNING_PATH = "/running"
# Stops every upstream llama-server; llama-swap itself keeps listening. A model id
# after a "/" stops that upstream alone.
UNLOAD_PATH = "/api/models/unload"
_HTTP_TIMEOUT_S = 10.0
# llama-swap's own proxy answers within a second; upstream model loads have their
# own (longer) budget inside llama-swap, so this only covers the proxy coming up.
_BOOT_TIMEOUT_S = 30.0
_BOOT_POLL_S = 0.25
# Cap on the captured llama-swap output a boot-failure error carries.
_BOOT_LOG_TAIL_CHARS = 2000
# Per-group SIGTERM grace before SIGKILL on the manager shutdown/reload path. A
# hard kill is safe (llama-server holds no persistent state). Note this is NOT
# the constant the serve handoff waits on: that path goes through stop_engine ->
# _stop_stale_swap and spends _ORPHAN_STOP_TIMEOUT_S plus the kill/reap waits, so
# SERVER_LOCK_TIMEOUT budgets only a teardown whose SIGTERMs are honored.
_STOP_TIMEOUT_S = 2.5
# Grace for a llama-server that outlived llama-swap before it is force-killed.
_ORPHAN_STOP_TIMEOUT_S = 5.0
# Grace for a SIGKILLed process to exit (and release its VRAM) before the next
# free-memory probe runs.
_KILL_WAIT_TIMEOUT_S = 5.0
_PROBE_TIMEOUT_S = 5.0
# Liveness probes talk to a loopback proxy, so they get their own short budget
# rather than the module's 10 s general HTTP one. The ladder runs this probe for
# every group while holding the cross-process build lock, so one wedged port
# (SYN-accepted but unresponsive) would otherwise stall every other lilbee start
# for tens of seconds. A local proxy that cannot answer /running this fast is
# not usable for inference either.
_LIVENESS_TIMEOUT = httpx.Timeout(connect=0.5, read=2.0, write=2.0, pool=2.0)
# Run as `python -m`; the frozen build dispatches `-m lilbee.*` to the module's main().
_REAPER_MODULE = "lilbee.providers.fleet.engine_reaper"
_REAPER_LOG_FILENAME = "engine-reaper.log"
# A reaper reports running within one interpreter start; a frozen onefile start
# on a slow disk is the long case.
_REAPER_READY_TIMEOUT_S = 10.0
_REAPER_READY_POLL_S = 0.05


@lru_cache(maxsize=1)
def _probe_client() -> httpx.Client:
    """One shared client for the localhost engine probes.

    ``httpx.get`` builds a fresh ``Client`` per call, and every ``Client``
    construction creates an SSL context, which loads the system CA bundle. These
    probes are plain HTTP to 127.0.0.1, so none of that TLS setup is ever used --
    and the readiness probe runs on the task bar's timer (up to 10 Hz), which made
    ``ssl.create_default_context`` 23% of TUI CPU in a py-spy profile. One client
    builds that at most once and keeps the connection alive between polls.
    ``trust_env`` is off so a proxy env var cannot redirect a loopback probe.
    """
    return httpx.Client(trust_env=False)


# /running JSON shape: {"running": [{"model": <id>, "state": "ready", ...}, ...]}.
_KEY_RUNNING = "running"
_KEY_MODEL = "model"
_KEY_STATE = "state"
_STATE_READY = "ready"
_STATE_STARTING = "starting"


def _platform_const(module: object, name: str, default: int) -> int:
    """A platform-conditional stdlib constant (absent on some OSes -> default)."""
    return getattr(module, name, default)


_CREATE_NEW_PROCESS_GROUP: int = _platform_const(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
_DETACHED_PROCESS: int = _platform_const(subprocess, "DETACHED_PROCESS", 0)
_SIGKILL: int = _platform_const(signal, "SIGKILL", signal.SIGTERM)


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* via a temp file in the same dir, then rename over it.

    A plain write truncates the destination first, so a process dying mid-write
    (OOM kill, SIGKILL, disk full) leaves an empty or half-written file behind.
    For the llama-swap config that means the next spawn hands the engine a file
    it cannot start from; for the state file it means a sibling's reap scan
    reads a torn record.

    The temp name carries the destination's name, and both config and state
    filenames embed the writing process's pid, so a crash leftover can be told
    from a live writer's file in flight -- see ``_clean_stale_tmp_files``.
    """
    tmp_path = path.with_name(f"{_STATE_TMP_PREFIX}{path.name}{_STATE_TMP_SUFFIX}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _state_filename(owner_pid: int, group: str) -> str:
    """The per-owner, per-group state filename for the lilbee process *owner_pid*."""
    return f"{_STATE_FILENAME_PREFIX}{group}.{owner_pid}{_STATE_FILENAME_SUFFIX}"


class EngineLifetime(StrEnum):
    """What ends a spawned engine."""

    BUILDER = "builder"  # the process that spawned it exits or dies
    USERS = "users"  # the dir's reaper, once no user lock and no keep-warm opt-in is left
    DETACHED = "detached"  # nothing; the dir's reaper only unloads its weights


@dataclass(frozen=True)
class SwapState:
    """A running llama-swap's recorded identity and serving contract.

    Read back from the engine dir's state file, so it describes engines this
    process did not start. The currency the bind/build ladder is written in:
    swap_manager records it, provider reads it to decide what a slot is
    serving, and contract matches it against what this process wants.
    """

    pid: int
    pgid: int | None
    created_at: float | None = None
    boot_id: str | None = None  # with start_ticks, the Linux identity a clock step cannot move
    start_ticks: float | None = None
    mach_start: tuple[str, int] | None = None  # (boot session, mach start), the macOS identity
    started_mono: float | None = None  # the system monotonic clock at spawn, on every platform
    started_boot: str | None = None  # the boot that clock reading belongs to
    member_ports: tuple[int, ...] = ()
    proxy_port: int | None = None
    launches: tuple[dict, ...] = ()
    engine_pin: str | None = None
    origin: PlanOrigin | None = None
    ready_at: float | None = None
    swap_binary: str = ""  # the llama-swap executable; empty in a record written before the field


@dataclass(frozen=True)
class PlanOrigin:
    """The config root of a fleet, the values its plan used, and the keys its builder follows.

    *fleet_view* holds each fleet reload key config.toml set at plan time, with the file's
    value; every version writes and reads it the same way. *planned_view* holds the same
    keys with the value the plan used, and *followed* names the reload keys the builder
    takes from config.toml. A record of an older version has neither. Two origins are
    equal when their root and their file view are.
    """

    config_root: str
    fleet_view: str | None
    planned_view: str | None = field(default=None, compare=False)
    followed: frozenset[str] | None = field(default=None, compare=False)


class EngineReadiness(StrEnum):
    """What an answering engine's /running listing says about binding to it."""

    READY = "ready"  # a member passed its health check
    LOADING = "loading"  # a member is starting; llama-swap queues requests behind it
    IDLE = "idle"  # nothing loaded, but a load was proved or the first one may not have begun
    FAILED = "failed"  # nothing loaded, no load ever proved, and the cold-load window is over


class SwapManager:
    """Owns one llama-swap process fronting one role group's servers.

    The provider runs one manager per role, so restarting a group (a placement
    or model change) never touches another group's loaded servers.
    """

    def __init__(self, data_dir: Path, group: SwapGroup) -> None:
        self._data_dir = data_dir
        self._group = group
        self._config_path = data_dir / _config_filename(os.getpid(), group.value)
        self._log_path = data_dir / _LOGS_SUBDIR / _LOG_FILENAME_TEMPLATE.format(group=group.value)
        # Instances whose engine report has already been compared to the estimate,
        # so the check runs once per start rather than on every readiness poll.
        self._estimate_checked: set[str] = set()
        self._launch_by_model: dict[str, InstanceLaunch] = {}
        # The file each member's engine writes its own log to; a member with none has
        # its output in this group's llama-swap log.
        self._engine_logs: dict[str, Path] = {}
        # Placement divergences by model id, surfaced on health. Guarded because
        # role_ready probes off the provider lock while health reads anytime.
        self._placement_warnings: dict[str, HealthWarning] = {}
        # Host spills by model id; a spill and a divergence can both hold for one model.
        self._spill_warnings: dict[str, HealthWarning] = {}
        # Set by a start that finds a router too old to copy a member's output.
        self._router_warning: HealthWarning | None = None
        self._warnings_lock = threading.Lock()
        self._state_path = data_dir / _state_filename(os.getpid(), group.value)
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: BinaryIO | None = None
        # Where this boot's output starts in the append-mode log file.
        self._log_offset = 0
        self._port: int | None = None
        self._member_ports: list[int] = []
        # Which member port serves which model id, for the direct GET /memory
        # readback. Only launches this manager spawned are in here: a bound
        # manager's state record carries the ports but not the mapping, and it
        # has no launches to check either.
        self._member_port_by_model: dict[str, int] = {}
        # The serving contract (per-role model/ctx/slots) persisted in every
        # state write, so a guest lilbee can bind to this live fleet.
        self._launches_payload: list[dict] = []
        self._swap_binary = ""
        self._origin: PlanOrigin | None = None
        # True when this manager uses an engine another process built: it then
        # never writes state, never reaps, and never signals engine processes.
        self._bound = False
        # When this builder first saw a member ready; persisted so a binder can
        # tell an idle-unloaded engine from one whose first load never finished.
        self._ready_at: float | None = None
        # The system monotonic clock at spawn. Every process of one boot shares it, so a
        # binder ages the engine by it and a wall-clock step cannot move the age.
        self._started_mono: float | None = None
        self._started_boot: str | None = None

    def start(
        self,
        launches: list[InstanceLaunch],
        *,
        engines: Mapping[WorkerRole, EngineAdapter],
        ttl_seconds: int = 0,
        lifetime: EngineLifetime = EngineLifetime.BUILDER,
        origin: PlanOrigin | None = None,
    ) -> None:
        """Write the config and spawn llama-swap, waiting for its proxy to answer.

        *engines* is the mapping the plan of these launches was made with; the
        start logs each launch with the engine of its role from it.

        The proxy and every member get a freshly allocated free port, which is
        why llama-swap's own startPort is not used: that assigns a fixed
        sequential range at config load, so it would collide with a previous
        instance's server still shutting down (the new llama-server then fails
        its bind and llama-swap reports it only as "exited prematurely").

        This narrows that collision rather than removing a race. The ports are
        picked by binding and closing ephemeral sockets, while llama-swap starts
        each upstream lazily on its first request, so a member port can sit
        unbound for as long as it takes that request to arrive and anything else
        on the box may take it in between. Nothing in llama-swap offers a
        spawn-time probe to close that window; warming the roles up front
        shortens it for the roles that are warmed.

        *lifetime* says what ends the engine. ``USERS`` needs the caller to hold
        the dir's build lock, and falls back to ``BUILDER`` when no reaper reports
        running for the dir, so the engine is never left unbound and unwatched.
        *origin* goes into the state, so a peer can see a fleet planned before a config write.
        """
        self._origin = origin
        # Found first, so a missing router fails here, before anything is stopped
        # or written.
        router = resolve_llama_swap()
        self._swap_binary = str(router)
        # Idempotent safety net; the provider reaps before planning so the GPU
        # probe already saw the real free memory.
        self.reap_stale()
        # Singleton guard: one llama-swap per data_dir for this lilbee. Reap any
        # llama-swap we already started against this config (a leaked duplicate
        # from a prior race/reload) before spawning, so they cannot accumulate
        # and double-book a GPU.
        _stop_own_fleet(self._config_path, tuple(self._member_ports))
        ports = _pick_free_ports(1 + len(launches))
        member_ports = dict(zip([launch.model_id for launch in launches], ports[1:], strict=True))
        self._member_ports = sorted(member_ports.values())
        self._member_port_by_model = dict(member_ports)
        self._launches_payload = [launch.to_state() for launch in launches]
        self._launch_by_model = {launch.model_id: launch for launch in launches}
        self._engine_logs = {}
        for launch in launches:
            log_path = engine_log_of(engines[launch.role], launch, self._log_path.parent)
            if log_path is not None:
                self._engine_logs[launch.model_id] = log_path
        reads_streams = self._router_reads_streams(router, launches)
        self._estimate_checked.clear()
        self._ready_at = None
        self._clear_placement_warnings()
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self._config_path,
            build_swap_config(
                launches,
                member_ports,
                swap=self._group.swaps,
                ttl_seconds=ttl_seconds,
                engine_logs=self._engine_logs,
                router_reads_streams=reads_streams,
            ),
        )
        self._port = ports[0]
        # Capture llama-swap's stdout/stderr to a file so its access log never
        # reaches an inherited terminal (a TUI/CLI parent) and garbles the screen.
        self._close_log()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("ab")
        self._log_offset = self._log_path.stat().st_size
        if lifetime is EngineLifetime.USERS and not ensure_engine_reaper(self._data_dir):
            log.warning("The engine at %s stops with this process.", self._data_dir)
            lifetime = EngineLifetime.BUILDER
        _raise_open_file_limit()
        # The router outlives this process under a shared lifetime, and the servers it starts
        # die with it, so the router holds the in-use file of every store build the fleet runs.
        from lilbee.core.config import cfg

        holds = BuildHolds(
            builds_run_by(
                engines_root(cfg.models_dir),
                [self._swap_binary, *(launch.server_binary for launch in launches)],
            )
        )
        try:
            self._proc = spawn_bound_child(
                [
                    self._swap_binary,
                    _CONFIG_FLAG,
                    str(self._config_path),
                    _LISTEN_FLAG,
                    f"{_HOST}:{self._port}",
                ],
                bind_lifetime=lifetime is EngineLifetime.BUILDER,
                env=engine_environ(),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                creationflags=_CREATE_NEW_PROCESS_GROUP,
                **holds.spawn_kwargs,
            )
        finally:
            holds.close()
        self._started_mono = time.monotonic()
        self._started_boot = boot_identity()
        self._write_state()
        self._await_health()
        log_role_launches(launches, engines=engines)

    def _router_reads_streams(self, router: Path, launches: list[InstanceLaunch]) -> bool:
        """Whether *router* reads a list of streams; warns when a member's output is lost.

        A release with no number (a source build, or a router that does not answer
        the probe) counts as current. Call after ``_engine_logs`` is filled. The
        warning goes to the log and to :meth:`health_warnings`, and each start
        replaces the one before it.
        """
        with self._warnings_lock:
            self._router_warning = None
        release = router_release(router)
        if release is None or release >= ROUTER_FLOOR:
            return True
        lost = [launch.model_id for launch in launches if launch.model_id not in self._engine_logs]
        if lost:
            warning = HealthWarning(
                code=WarningCode.ROUTER_OUTDATED,
                message=(
                    f"The llama-swap router at {router} is v{release}, so the output of "
                    f"{', '.join(lost)} is not in {self._log_path}."
                ),
                remedy=(
                    f"Replace the file at {router} with llama-swap v{ROUTER_FLOOR} or later. "
                    f"Download it from {_ROUTER_RELEASES_URL}, or run '{ROUTER_GO_INSTALL}' "
                    f"and copy the llama-swap file from your Go bin folder to {router}. "
                    "Then restart lilbee."
                ),
            )
            log.warning("%s", warning.line)
            with self._warnings_lock:
                self._router_warning = warning
        return False

    def reap_stale(self) -> None:
        """Kill every dead or unhealthy recorded engine; see :func:`reap_stale`."""
        reap_stale(self._data_dir)

    def _process_identity(self) -> tuple[int, int | None, float | None] | None:
        """(pid, pgid, create time) of the swap this manager runs, or None."""
        if self._proc is not None:
            pid = self._proc.pid
            pgid: int | None = None
            if sys.platform != "win32":
                with contextlib.suppress(ProcessLookupError):
                    pgid = os.getpgid(pid)
            created_at: float | None = None
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                created_at = psutil.Process(pid).create_time()
            return pid, pgid, created_at
        return None

    def _write_state(self) -> None:
        """Record the swap's pid/pgid/create time, member ports, and our identity.

        The write is atomic (tmp file then ``os.replace``) so a sibling's reap
        scan can never read a torn file and mistake this live record for junk.
        """
        identity = self._process_identity()
        if identity is None:
            return
        swap_pid, pgid, created_at = identity
        state: dict[str, object] = {
            _STATE_KEY_PID: swap_pid,
            _STATE_KEY_PGID: pgid,
            _STATE_KEY_CREATED_AT: created_at,
            _STATE_KEY_NAME: _LLAMA_SWAP_PROCESS_NAME,
            _STATE_KEY_MEMBER_PORTS: self._member_ports,
            _STATE_KEY_PROXY_PORT: self._port,
            _STATE_KEY_LAUNCHES: self._launches_payload,
            _STATE_KEY_ENGINE_PIN: engine_pin(),
            _STATE_KEY_SWAP_BINARY: self._swap_binary,
        }
        state.update(_start_stamp_fields(swap_pid))
        if self._origin is not None:
            state[_STATE_KEY_CONFIG_ROOT] = self._origin.config_root
            state[_STATE_KEY_FLEET_VIEW] = self._origin.fleet_view
            if self._origin.planned_view is not None:
                state[_STATE_KEY_PLANNED_VIEW] = self._origin.planned_view
            if self._origin.followed is not None:
                state[_STATE_KEY_FOLLOWED_KEYS] = sorted(self._origin.followed)
        if self._ready_at is not None:
            state[_STATE_KEY_READY_AT] = self._ready_at
        if self._started_mono is not None and self._started_boot is not None:
            state[_STATE_KEY_STARTED_MONO] = self._started_mono
            state[_STATE_KEY_STARTED_BOOT] = self._started_boot
        _atomic_write(self._state_path, json.dumps(state))

    def restamp(self, origin: PlanOrigin) -> None:
        """Record *origin* for the running swap: its plan still matches config.toml."""
        self._origin = origin
        self._write_state()

    def record_ready(self) -> None:
        """Stamp the state file once when a load first succeeds; a binder writes nothing."""
        if self._bound or self._ready_at is not None:
            return
        self._ready_at = time.time()
        self._write_state()

    @property
    def log_path(self) -> Path:
        """The llama-swap log for this group, which records each member's exit."""
        return self._log_path

    def endpoint(self) -> str:
        """Base URL of the llama-swap OpenAI-compatible proxy."""
        if self._port is None:
            raise ProviderError(
                "The local model engine is not running.",
                kind=ProviderErrorKind.SERVER,
            )
        return f"http://{_HOST}:{self._port}"

    def role_ready(self, role: WorkerRole) -> bool:
        """Whether at least one of *role*'s replica servers is loaded and ready."""
        prefix = role_model_prefix(role)
        ready = self._ready_models()
        if ready is None:
            return False  # a failed read says nothing about the engines, so no finding changes
        self._check_estimates(ready)
        if ready:
            self.record_ready()
        return any(model.startswith(prefix) for model in ready)

    def _check_estimates(self, ready: set[str]) -> None:
        """Compare each newly-ready engine's own report against what it was planned for.

        The plan is otherwise open-loop, and a wrong estimate only ever surfaces
        as a failed request much later. Once per load: readiness is polled, and the
        answer does not change while the engine stays loaded. An engine that left the
        ready set has unloaded, so its findings go and its next load is read afresh.
        """
        for model_id in self._estimate_checked - ready:
            self._estimate_checked.discard(model_id)
            with self._warnings_lock:
                self._placement_warnings.pop(model_id, None)
                self._spill_warnings.pop(model_id, None)
        for model_id in ready - self._estimate_checked:
            launch = self._launch_by_model.get(model_id)
            self._estimate_checked.add(model_id)
            if launch is None:
                continue
            # Ready means this role's context loaded, so any reduction taken to
            # get here has done its job and must not follow the role into the
            # next plan, a freed machine, or a model the user switched to.
            clear_ctx_downshift(launch.role)
            # A launch carrying --memory serves its own report on GET /memory
            # and writes a log at the default verbosity, with no load report
            # (swap_config), so the log-side checks would misfire on it by construction.
            if MEMORY_FLAG in launch.argv:
                self._check_memory_estimate(model_id, launch)
                continue
            # An engine that writes no log of its own has no file to read back.
            if model_id not in self._engine_logs:
                continue
            # The engine is ready, so a missing log is not "too early" any more.
            if report_missing_log(self._log_path.parent, model_id, launch.role):
                continue
            if launch.est_vram_bytes <= 0:
                continue
            spills: list[HealthWarning] = []
            warning = check_launch(
                self._log_path.parent,
                model_id,
                launch.role,
                launch.model,
                launch.est_vram_bytes,
                launch.est_vram_by_device,
                launch.est_unreported_bytes,
                launch.est_host_bytes,
                spills,
            )
            self._record_findings(model_id, warning, spills)

    def _check_memory_estimate(self, model_id: str, launch: InstanceLaunch) -> None:
        """Run the estimate check off the engine's own ``GET /memory``.

        Straight to the member's port rather than through llama-swap: the model
        is ready, so the server owns its port, and the proxy adds only a routing
        layer that has nothing to route on for a bare GET. An endpoint that does
        not answer after the engine took the flag is reported, not swallowed --
        it is this mode's analog of a ready engine that wrote no log.
        """
        port = self._member_port_by_model.get(model_id)
        if port is None:  # bound to another process's engine; nothing was planned here
            return
        try:
            resp = _probe_client().get(f"http://{_HOST}:{port}/memory", timeout=_PROBE_TIMEOUT_S)
            payload = resp.json() if resp.status_code == httpx.codes.OK else None
        except (httpx.HTTPError, ValueError):
            payload = None
        if payload is None:
            log.warning(
                "The %s engine was launched with %s but its /memory endpoint did not "
                "answer, so its memory use could not be checked against the estimate. "
                "Placement estimates for this model are unverified.",
                launch.role.value,
                MEMORY_FLAG,
            )
            return
        if launch.est_vram_bytes <= 0:
            return
        spills: list[HealthWarning] = []
        warning = check_memory_report(
            launch.role,
            launch.model,
            launch.est_vram_bytes,
            launch.est_vram_by_device,
            payload,
            launch.est_host_bytes,
            spills,
        )
        self._record_findings(model_id, warning, spills)

    def _record_findings(
        self, model_id: str, placement: HealthWarning | None, spills: list[HealthWarning]
    ) -> None:
        """Keep what a readback found for *model_id*, for :meth:`health_warnings`."""
        with self._warnings_lock:
            if placement is not None:
                self._placement_warnings[model_id] = placement
            if spills:
                self._spill_warnings[model_id] = spills[-1]

    def _clear_placement_warnings(self) -> None:
        """Forget every readback finding; the engines they describe are gone or restarting."""
        with self._warnings_lock:
            self._placement_warnings.clear()
            self._spill_warnings.clear()

    def health_warnings(self) -> list[HealthWarning]:
        """Placement divergences and host spills of ready engines, and an outdated router."""
        with self._warnings_lock:
            warnings = [*self._placement_warnings.values(), *self._spill_warnings.values()]
            if self._router_warning is not None:
                warnings.append(self._router_warning)
            return warnings

    def is_live(self) -> bool:
        """Whether the swap process is up and its proxy answers ``/running``."""
        if self._proc is None or self._proc.poll() is not None:
            return False
        if self._port is None:
            return False
        return self._proxy_answers()

    @property
    def running(self) -> bool:
        """Whether this manager currently has a spawned llama-swap process."""
        return self._proc is not None

    @property
    def bound(self) -> bool:
        """Whether this manager rides an engine built by another process."""
        return self._bound

    def bind(self, state: SwapState) -> bool:
        """Use a running engine's proxy without taking any ownership of it.

        The engine's own state record stays untouched: the binder writes
        nothing, and shutdown() merely drops the binding.
        """
        if state.proxy_port is None:
            return False
        self._port = state.proxy_port
        self._member_ports = list(state.member_ports)
        if not self._proxy_answers():
            self._port = None
            self._member_ports = []
            return False
        self._launches_payload = [dict(launch) for launch in state.launches]
        self._bound = True
        return True

    def _proxy_answers(self) -> bool:
        """Whether the bound proxy port serves llama-swap's running endpoint.

        Shares state_is_healthy's identity check via _running_endpoint_answers, so
        bind and reap agree on what "answering" means by construction rather than by
        two hand-kept-identical copies.
        """
        return _running_endpoint_answers(self.endpoint())

    def shutdown(self) -> None:
        """Stop every llama-swap this lilbee owns at our config and reap servers.

        Authoritative teardown keyed on config-path identity, not the single
        tracked ``Popen``: a warm-up/reset race or a reload can leave several
        llama-swap processes this lilbee spawned, any of them reparented to init
        (still holding the engine binary open) -- trusting one handle would leak
        them. Every llama-swap running against our config is reaped. Unlinks only
        this owner's state file; another instance's record stays.
        """
        if self._bound:
            # Not ours to stop: drop the binding and leave the engine serving.
            self._bound = False
            self._port = None
            self._member_ports = []
            self._launches_payload = []
            return
        _stop_own_fleet(self._config_path, tuple(self._member_ports))
        self._clear_placement_warnings()
        # Nothing is coming back to bind these, so the picker can offer them again.
        release_reserved_ports([*self._member_ports, *([self._port] if self._port else [])])
        self._state_path.unlink(missing_ok=True)
        if self._proc is not None:
            # Free this engine's death pipe so its watcher exits now, not at our death.
            release_death_pipe(self._proc.pid)
        self._proc = None
        self._port = None
        self._close_log()

    def _close_log(self) -> None:
        """Close the captured llama-swap log handle, if one is open."""
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None

    def _await_health(self) -> None:
        """Poll the proxy's /health until it answers, or fail with a clear error."""
        url = f"{self.endpoint()}{_HEALTH_PATH}"
        deadline = time.monotonic() + _BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                self._fail("The local model engine exited before it was ready.")
            with contextlib.suppress(httpx.HTTPError):
                if _probe_client().get(url, timeout=_PROBE_TIMEOUT_S).status_code == httpx.codes.OK:
                    return
            time.sleep(_BOOT_POLL_S)
        self._fail("The local model engine did not start in time.")

    def _ready_models(self) -> set[str] | None:
        """Model ids whose upstream is loaded and ready, per llama-swap's /running.

        A read-only probe: a concurrent shutdown can clear ``_port`` between the
        caller's check and ``endpoint()``, raising ProviderError, so that is
        suppressed too and the probe reports None rather than throwing. None means
        the read failed; an empty set means llama-swap answered with nothing ready.
        """
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError, TypeError, ProviderError):
            payload = (
                _probe_client()
                .get(f"{self.endpoint()}{RUNNING_PATH}", timeout=_PROBE_TIMEOUT_S)
                .json()
            )
            return ready_model_ids(payload)
        return None

    def _boot_log_tail(self) -> str:
        """The current boot's captured llama-swap output, capped for an error message."""
        try:
            with self._log_path.open("rb") as handle:
                handle.seek(self._log_offset)
                data = handle.read()
        except OSError:
            return ""
        return data.decode(errors="replace").strip()[-_BOOT_LOG_TAIL_CHARS:]

    def _fail(self, message: str) -> None:
        """Tear down and raise a user-facing engine-start error carrying the boot log."""
        self.shutdown()
        tail = self._boot_log_tail()
        if tail:
            message = f"{message} Engine log ({self._log_path}):\n{tail}"
        raise ProviderError(message, kind=ProviderErrorKind.SERVER)


# Linux publishes the range here; every other platform is asked via sysctl.
_PROC_PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")


def _port_range_from(path: Path) -> tuple[int, int] | None:
    """The two integers in *path*, or ``None`` when it is absent or unreadable."""
    try:
        low, high = path.read_text(encoding="utf-8").split()[:2]
        return int(low), int(high)
    except (OSError, ValueError):
        return None


def _ephemeral_range() -> tuple[int, int] | None:
    """The port range the kernel hands out for unbound sockets, if it says.

    ``None`` when neither source answers, which is the signal to fall back to
    letting the OS choose.
    """
    from_proc = _port_range_from(_PROC_PORT_RANGE)
    if from_proc is not None:
        return from_proc
    try:  # macOS and the BSDs, which have no procfs entry for this
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "net.inet.ip.portrange.first", "net.inet.ip.portrange.last"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        low, high = out.stdout.split()[:2]
        return int(low), int(high)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# Where lilbee looks for engine ports when the kernel's ephemeral range is known.
# Above the registered-service crowd, below every default ephemeral range.
_PORT_SEARCH_FLOOR = 20000
_PORT_WINDOW_SPAN = 8192
# Block width. Each process searches one block, so concurrent lilbees hold
# disjoint ranges. A fleet takes one proxy port plus one per member, and embed
# and vision replicate per GPU, so 64 covers a 30-GPU host; a wider fleet spills
# into the next block.
_PORT_BLOCK = 64

# Ports handed to a child that has not bound them yet. llama-swap binds a member
# port only on that member's first request, so the probe socket is long closed
# by then and the port looks free to every later probe. Without this the picker
# hands the next group exactly what it gave the last one, every time.
_reserved_ports: set[int] = set()
_reserved_lock = threading.Lock()


def release_reserved_ports(ports: Iterable[int]) -> None:
    """Give *ports* back to the picker, once nothing is expected to bind them."""
    with _reserved_lock:
        _reserved_ports.difference_update(ports)


def _window_span(ceiling: tuple[int, int]) -> int:
    """How many ports below the ephemeral floor this host leaves to search."""
    return max(1, min(ceiling[0], _PORT_SEARCH_FLOOR + _PORT_WINDOW_SPAN) - _PORT_SEARCH_FLOOR)


def _search_start(ceiling: tuple[int, int]) -> int:
    """First port of the block this process owns.

    The pid selects a whole block, not an offset: reservation is per-process, and
    a fleet takes its ports contiguously, so pid-offset starts one apart overlap
    on all but one port.
    """
    blocks = max(1, _window_span(ceiling) // _PORT_BLOCK)
    return _PORT_SEARCH_FLOOR + (os.getpid() % blocks) * _PORT_BLOCK


def _pick_free_ports(count: int) -> list[int]:
    """Bind *count* free localhost ports at once and return them.

    All sockets stay open until every port is claimed so the OS cannot hand the
    same port out twice within one allocation.

    Picked from below the kernel's ephemeral range rather than inside it. The
    gap between lilbee picking a port and llama-server binding it spans the whole
    lazy-spawn wait, and a port inside the ephemeral range can be handed to any
    passing outbound connection during that gap; one below it cannot be handed to
    anybody, so the only way to lose it is another server binding that exact port
    on purpose. Falls back to letting the OS choose when the range is unknown.
    """
    ceiling = _ephemeral_range()
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in sockets:
            _bind_below_ephemeral(sock, ceiling)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _bind_below_ephemeral(sock: socket.socket, ceiling: tuple[int, int] | None) -> None:
    """Bind *sock* to a free, unreserved port under the ephemeral floor.

    Falls back to letting the OS choose when the range is unknown or the window
    is used up, which keeps a fleet start working at the cost of returning to the
    ephemeral range for those ports.
    """
    if ceiling is not None and ceiling[0] > _PORT_SEARCH_FLOOR:
        span = _window_span(ceiling)
        start = _search_start(ceiling)
        for offset in range(span):
            port = _PORT_SEARCH_FLOOR + (start - _PORT_SEARCH_FLOOR + offset) % span
            with _reserved_lock:
                if port in _reserved_ports:
                    continue
                try:
                    sock.bind((_HOST, port))
                except OSError:
                    continue
                _reserved_ports.add(port)
                return
    sock.bind((_HOST, 0))


def _live_children(pid: int) -> list[psutil.Process]:
    """The process's current descendants, or none when it already exited."""
    try:
        children: list[psutil.Process] = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    return children


def _reap_survivors(children: list[psutil.Process]) -> None:
    """Terminate then kill any captured child that is still running."""
    survivors = [child for child in children if child.is_running()]
    for child in survivors:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.terminate()
    _, alive = psutil.wait_procs(survivors, timeout=_ORPHAN_STOP_TIMEOUT_S)
    for child in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            child.kill()
    _await_killed(alive)


def _await_killed(procs: list[psutil.Process]) -> None:
    """Wait for SIGKILLed processes to exit so their VRAM is free before any probe."""
    if not procs:
        return
    _, alive = psutil.wait_procs(procs, timeout=_KILL_WAIT_TIMEOUT_S)
    for proc in alive:
        log.warning("Process %s survived SIGKILL; its VRAM may still be held.", proc.pid)


def _processes_named(needle: str) -> Iterator[psutil.Process]:
    """Live processes whose executable name contains *needle*.

    ``name()`` is a cheap field (comm/proc_name); ``cmdline()`` reads the full
    argument vector and on macOS blocks on entitlement-protected binaries. So the
    name is the pre-filter and callers pay for ``cmdline()`` only on a match,
    which keeps a full-process-table scan from stalling on an unrelated process.
    """
    for proc in psutil.process_iter(["name"]):
        # process_iter already skips processes that vanish mid-scan and, per its
        # ad_value contract, leaves ``name`` as None where it could not be read.
        name = proc.info["name"] or ""
        if needle in name:
            yield proc


def _swaps_for_config(config_path: Path) -> list[psutil.Process]:
    """Every live llama-swap (any owner) running against *config_path*.

    Identity is the ``-config <path>`` argument, which every llama-swap this
    lilbee starts carries and which survives reparenting to init -- so this finds
    a leaked duplicate or a swap reparented away from us, neither of which a
    tracked Popen handle nor a ``children()`` scan would catch.
    """
    target = str(config_path)
    swaps: list[psutil.Process] = []
    for proc in _processes_named(_LLAMA_SWAP_PROCESS_NAME):
        try:
            cmdline = proc.cmdline()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
            SystemError,
        ):
            # OSError/SystemError: macOS psutil mishandles entitlement-protected
            # binaries (sysctl KERN_PROCARGS2), leaking a raw PermissionError or a
            # C-extension SystemError instead of an AccessDenied.
            continue
        # Identity is the -config path; _processes_named already gated on comm.
        if target in cmdline:
            swaps.append(proc)
    return swaps


def find_live_state(data_dir: Path, group: SwapGroup) -> SwapState | None:
    """The newest recorded state for *group* at *data_dir* (no liveness check).

    A record's presence does not prove the engine is up; callers that need that
    probe it with ``state_is_healthy``. The name reflects that a record is written
    only for a running engine, not that this function verifies it.
    """
    best: SwapState | None = None
    for state_path in sorted(data_dir.glob(_STATE_FILE_GLOB)):
        if f".{group.value}." not in f".{state_path.name}":
            continue
        state = _load_state(state_path)
        if state is None:
            continue
        if best is None or (state.created_at or 0) > (best.created_at or 0):
            best = state
    return best


def ready_model_ids(payload: Any) -> set[str]:
    """The model ids that a decoded ``/running`` body lists as ready.

    Raises KeyError, TypeError or AttributeError for a body of another shape.
    """
    return {
        entry[_KEY_MODEL]
        for entry in payload[_KEY_RUNNING]
        if entry.get(_KEY_STATE) == _STATE_READY
    }


def _running_listing(base_url: str) -> list[dict] | None:
    """The ``running`` entries at *base_url*, or ``None`` when it is not llama-swap.

    Proxy ports are ephemeral: after an engine dies, any unrelated local service
    that later binds the recorded port and returns a 2xx/3xx to an unknown path
    would pass a bare status check, so a dead record would look healthy forever and
    inference clients would bind to a non-engine endpoint. Requiring the ``running``
    JSON payload shape that only llama-swap produces makes the probe identity-checked.
    Total: any transport error or non-conforming body reads as "not our engine".
    """
    try:
        resp = _probe_client().get(f"{base_url}{RUNNING_PATH}", timeout=_LIVENESS_TIMEOUT)
    except (OSError, httpx.HTTPError):
        return None
    if resp.status_code >= httpx.codes.BAD_REQUEST:
        return None
    try:
        listing = resp.json().get(_KEY_RUNNING)
    except (ValueError, AttributeError):
        return None
    return listing if isinstance(listing, list) else None


def _running_endpoint_answers(base_url: str) -> bool:
    """Whether *base_url* serves llama-swap's ``/running`` endpoint (identity, not liveness)."""
    return _running_listing(base_url) is not None


def state_is_healthy(state: SwapState) -> bool:
    """Whether the engine behind *state* answers on its recorded proxy port."""
    if state.proxy_port is None:
        return False
    return _running_endpoint_answers(f"http://{_HOST}:{state.proxy_port}")


def _bind_deadline_s(state: SwapState) -> float:
    """Seconds after spawn at which a record that never loaded is called failed.

    The slowest member's cold-load ceiling (llama-swap's own health-check timeout)
    plus the window the builder gets to bring the proxy up before it warms.
    """
    try:
        launches = [InstanceLaunch.from_state(item) for item in state.launches]
    except (KeyError, TypeError, ValueError):
        launches = []
    return health_check_timeout_s(launches) + _BOOT_TIMEOUT_S


def _engine_age_s(state: SwapState, now: float) -> float | None:
    """Seconds the engine behind *state* has run, or None when the record gives no start.

    A record with the spawn reading of the system monotonic clock and the identity of the
    boot it was read in is aged by that clock, on every platform, when the identity is the
    current boot's; a wall-clock step does not move it. A Linux record without them but
    with the boot id and start ticks is aged by the boot clock. Any other record (a file
    written before both, or one from another boot) is aged by the wall clock *now* against
    its recorded create time, so a forward step there can still pass the deadline early.
    """
    if (
        state.started_mono is not None
        and state.started_boot is not None
        and state.started_boot == boot_identity()
    ):
        return time.monotonic() - state.started_mono
    if state.boot_id is not None and state.start_ticks is not None:
        age = linux_age_s(state.boot_id, state.start_ticks)
        if age is not None:
            return age
    return None if state.created_at is None else now - state.created_at


def _unloaded_readiness(state: SwapState, now: float) -> EngineReadiness:
    """IDLE or FAILED for an answering engine with no member loading or ready."""
    if state.ready_at is not None:
        return EngineReadiness.IDLE
    age = _engine_age_s(state, now)
    if age is None or age < _bind_deadline_s(state):
        return EngineReadiness.IDLE
    return EngineReadiness.FAILED


def engine_readiness(state: SwapState, *, now: float) -> EngineReadiness | None:
    """The bind verdict from one ``/running`` read, ``None`` when the engine does not answer.

    *now* is wall-clock time; it ages only a record without the monotonic spawn reading or the
    Linux boot id and start ticks.
    """
    listing = (
        None if state.proxy_port is None else _running_listing(f"http://{_HOST}:{state.proxy_port}")
    )
    if listing is None:
        return None
    # /running is an untyped body: a state that is not a string cannot name a member state.
    members = [entry.get(_KEY_STATE) for entry in listing if isinstance(entry, dict)]
    if not all(member is None or isinstance(member, str) for member in members):
        return None
    if _STATE_READY in members:
        return EngineReadiness.READY
    if _STATE_STARTING in members:
        return EngineReadiness.LOADING
    return _unloaded_readiness(state, now)


def engine_record_exists(data_dir: Path) -> bool:
    """Whether any engine state file is present, without probing proxy health.

    A filesystem fact, unlike a proxy HTTP probe: it is true for an engine that
    is live but momentarily unprobeable (fd exhaustion, host thrash), so the
    ladder can clear a recorded engine before building rather than double-build
    beside one an HTTP probe failed to see.
    """
    return any(data_dir.glob(_STATE_FILE_GLOB))


def stop_engine(data_dir: Path) -> list[str]:
    """Stop every engine the dir's state files record, regardless of liveness.

    The unconditional off switch behind ``lilbee engine stop`` and the
    last-user-out path: each recorded swap is terminated through its state
    record (never a Popen handle, so it works on engines this process did
    not build) and its file removed. A record whose llama-swap is already dead
    still has its llama-servers (each in its own process group) reaped by
    recorded port, exactly as reap_stale does -- otherwise the off switch would
    leave those orphans holding VRAM and delete the ports needed to find them.
    Stale config files for dead owners are cleaned too, and the persistence
    opt-in is dropped with the engine it described, so the dir is left as
    clean as a reap leaves it. Unparseable files are left alone, as in
    reap_stale: they may be a sibling's in-flight write. Returns the group tokens
    whose engine was actually alive, so a caller reports only real stops.
    """
    return _stop_recorded_engines(data_dir, _stop_recorded_engine)


def kill_engine(data_dir: Path) -> list[str]:
    """Kill every engine the dir's state files record, with no SIGTERM grace.

    The exit's stop of an engine whose start never finished: it holds no state
    worth a graceful stop, and a server that traps SIGTERM would hold the exit
    for the whole grace. Records and the persistence opt-in go as in
    ``stop_engine``. Returns the group tokens whose engine was alive.
    """
    return _stop_recorded_engines(data_dir, _kill_recorded_engine)


def _stop_recorded_engines(data_dir: Path, stop: Callable[[SwapState], bool]) -> list[str]:
    """Run *stop* on each recorded engine and drop its record; the groups *stop* found alive."""
    _clean_stale_configs(data_dir)
    # The persistence opt-in describes the engine instance being stopped, so it
    # dies with it. Cleared here rather than at each call site so no stop path
    # can leave a mark that makes the next engine sticky-warm.
    clear_keep_warm(data_dir)
    stopped: list[str] = []
    for state_path in sorted(data_dir.glob(_STATE_FILE_GLOB)):
        state = _load_state(state_path)
        if state is None:
            continue
        if stop(state):
            group = _state_group(state_path.name)
            if group is not None:
                stopped.append(group)
        state_path.unlink(missing_ok=True)
    return stopped


def unload_engine_models(data_dir: Path) -> bool:
    """Unload the models of every engine *data_dir* records; False when any proxy refused.

    The llama-swap processes stay up, so the next request reloads on demand.
    """
    states = (_load_state(path) for path in sorted(data_dir.glob(_STATE_FILE_GLOB)))
    ports = [state.proxy_port for state in states if state and state.proxy_port is not None]
    return all([_unload_models(port) for port in ports])


def engine_answers(data_dir: Path) -> bool:
    """Whether any engine *data_dir* records answers on its proxy port."""
    states = (_load_state(path) for path in sorted(data_dir.glob(_STATE_FILE_GLOB)))
    return any(state_is_healthy(state) for state in states if state)


def _unload_models(proxy_port: int) -> bool:
    url = f"http://{_HOST}:{proxy_port}{UNLOAD_PATH}"
    try:
        resp = _probe_client().post(url, timeout=_HTTP_TIMEOUT_S)
    except (OSError, httpx.HTTPError) as exc:
        log.warning("Could not unload the models at %s: %s", url, exc)
        return False
    if resp.status_code >= httpx.codes.BAD_REQUEST:
        log.warning("The engine at %s refused the unload: HTTP %d", url, resp.status_code)
        return False
    return True


def ensure_engine_reaper(data_dir: Path) -> bool:
    """Make sure a live reaper watches *data_dir*; False when none reports running.

    A reaper reports running by holding the dir's reaper lock. Waiting for that,
    not only for the launch, also catches a reaper that dies on import.
    """
    if reaper_running(data_dir):
        return True
    reaper_log = data_dir / _LOGS_SUBDIR / _REAPER_LOG_FILENAME
    try:
        reaper_log.parent.mkdir(parents=True, exist_ok=True)
        log_offset = reaper_log.stat().st_size if reaper_log.exists() else 0
        # The child keeps its own copy of the handle; the parent closes its copy at once.
        with reaper_log.open("ab") as reaper_err:
            reaper = subprocess.Popen(  # noqa: S603 - argv is this interpreter and a fixed module
                [sys.executable, "-m", _REAPER_MODULE, str(data_dir)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=reaper_err,
                start_new_session=True,
                creationflags=_CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS,
            )
    except OSError:
        log.warning("Could not launch the engine reaper for %s.", data_dir)
        return False
    deadline = time.monotonic() + _REAPER_READY_TIMEOUT_S
    while time.monotonic() < deadline and reaper.poll() is None:
        if reaper_running(data_dir):
            return True
        time.sleep(_REAPER_READY_POLL_S)
    # An exited reaper either lost the claim to a live one or failed to start.
    if reaper_running(data_dir):
        return True
    log.warning(
        "The engine reaper for %s did not report running. Its log is %s.%s",
        data_dir,
        reaper_log,
        _log_tail_since(reaper_log, log_offset),
    )
    return False


def _log_tail_since(path: Path, offset: int) -> str:
    """The text written to *path* after *offset*, capped, as a message suffix ('' when none)."""
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return ""
    text = data.decode(errors="replace").strip()[-_BOOT_LOG_TAIL_CHARS:]
    return f"\n{text}" if text else ""


def reap_stale(data_dir: Path) -> bool:
    """Kill every dead or unhealthy recorded engine at *data_dir*.

    An OOM-killed lilbee leaves llama-swap (and its servers) holding VRAM,
    so planning would otherwise see artificially reduced free memory; the
    ladder calls this before its GPU probe. Every state file is scanned
    (all groups, including legacy names): an engine that is alive AND
    answering on its proxy is spared regardless of who started it (a
    reload's own healthy groups, or a bindable engine the ladder skipped);
    everything else is stopped through its record and its file removed. An
    unparseable file is skipped, never deleted: it may be a sibling's
    in-flight write. When the swap itself is dead, its servers (each in
    its own process group) may still be alive holding VRAM; they are
    matched by name plus recorded member port and stopped before the file
    is removed.

    Returns whether it stopped any engine, so a caller can drop a device reading
    taken while that engine held the card.

    Module-level (not a method) because it must run before planning decides
    which role groups exist, when no per-group manager has been built yet.
    """
    _clean_stale_tmp_files(data_dir)
    _clean_stale_configs(data_dir)
    stopped = False
    for state_path in sorted(data_dir.glob(_STATE_FILE_GLOB)):
        state = _load_state(state_path)
        if state is None:
            continue
        if state_is_healthy(state):
            # An answering engine is in use (bind accepts on exactly this
            # test); reaping must never disagree with binding.
            continue
        stopped = _stop_recorded_engine(state) or stopped
        state_path.unlink(missing_ok=True)
    return stopped


def _clean_stale_tmp_files(data_dir: Path) -> None:
    """Remove crash-leftover state and config tmp files whose writer is dead."""
    tmp_glob = f"{_STATE_TMP_PREFIX}*{_STATE_TMP_SUFFIX}"
    for tmp_path in data_dir.glob(tmp_glob):
        writer_pid = _state_owner_pid(tmp_path.name)
        if writer_pid is not None and not psutil.pid_exists(writer_pid):
            tmp_path.unlink(missing_ok=True)


def _clean_stale_configs(data_dir: Path) -> None:
    """Remove per-owner config files whose owner lilbee is gone.

    The swaps themselves are reaped from the state files; these leftover config
    files are just clutter once their writer pid is dead. A live owner's config
    (pid still exists) and a pid-less legacy name are left untouched; skipping on
    pid reuse only leaves harmless clutter, never deletes a live owner's config.
    """
    for config_path in data_dir.glob(_CONFIG_FILE_GLOB):
        owner = _config_owner_pid(config_path.name)
        if owner is not None and not psutil.pid_exists(owner):
            config_path.unlink(missing_ok=True)


def _stop_own_fleet(config_path: Path, member_ports: tuple[int, ...]) -> None:
    """Stop every llama-swap this lilbee owns at *config_path* and reap upstreams.

    Keyed on config-path identity rather than a tracked Popen or the live process
    tree: a warm-up/reload race can leave several llama-swap processes this lilbee
    started, any of which may be reparented to init, so no single handle or child
    scan finds them all. Every llama-swap running against our config is reaped:
    the build lock guarantees one builder per engine dir, so no sibling sparing
    applies. Each swap runs each llama-server in its own process group, so the
    upstreams are swept separately: captured descendants plus any llama-server
    still bound to one of our member ports (a respawned upstream the descendant
    snapshot missed), then confirmed gone.
    """
    swaps = list(_swaps_for_config(config_path))
    children: list[psutil.Process] = []
    for swap in swaps:
        children.extend(_live_children(swap.pid))
    for swap in swaps:
        if sys.platform == "win32":
            _hard_stop_proc(swap)
        else:
            _terminate_proc_group(swap)
    _reap_survivors(children + _find_orphan_servers(member_ports))


def _terminate_proc_group(proc: psutil.Process) -> None:
    """SIGTERM a process's group, escalating to SIGKILL on timeout."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):  # pragma: no cover - exited between checks
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except psutil.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, _SIGKILL)
        _await_killed([proc])


def _hard_stop_proc(proc: psutil.Process) -> None:
    """Terminate a process, escalating to a hard kill on timeout (Windows path)."""
    with contextlib.suppress(psutil.NoSuchProcess):
        proc.terminate()
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except psutil.TimeoutExpired:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.kill()


def _load_state(path: Path) -> SwapState | None:
    """Parse a state file into a :class:`SwapState`; ``None`` when absent/corrupt."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_pgid = payload.get(_STATE_KEY_PGID)
        raw_created = payload.get(_STATE_KEY_CREATED_AT)
        raw_boot_id = payload.get(_STATE_KEY_BOOT_ID)
        raw_ticks = payload.get(_STATE_KEY_START_TICKS)
        raw_ports = payload.get(_STATE_KEY_MEMBER_PORTS) or []
        raw_proxy = payload.get(_STATE_KEY_PROXY_PORT)
        raw_root = payload.get(_STATE_KEY_CONFIG_ROOT)
        raw_followed = payload.get(_STATE_KEY_FOLLOWED_KEYS)
        origin = (
            None
            if raw_root is None
            else PlanOrigin(
                str(raw_root),
                payload.get(_STATE_KEY_FLEET_VIEW),
                payload.get(_STATE_KEY_PLANNED_VIEW),
                None if raw_followed is None else frozenset(str(key) for key in raw_followed),
            )
        )
        raw_ready = payload.get(_STATE_KEY_READY_AT)
        raw_mono = payload.get(_STATE_KEY_STARTED_MONO)
        raw_started_boot = payload.get(_STATE_KEY_STARTED_BOOT)
        has_mono = raw_mono is not None and raw_started_boot is not None
        return SwapState(
            pid=int(payload[_STATE_KEY_PID]),
            pgid=int(raw_pgid) if raw_pgid is not None else None,
            created_at=float(raw_created) if raw_created is not None else None,
            boot_id=str(raw_boot_id) if raw_boot_id and raw_ticks is not None else None,
            start_ticks=float(raw_ticks) if raw_boot_id and raw_ticks is not None else None,
            mach_start=_load_mach_start(payload.get(_STATE_KEY_MACH_START)),
            started_mono=float(raw_mono) if has_mono else None,
            started_boot=str(raw_started_boot) if has_mono else None,
            member_ports=tuple(int(port) for port in raw_ports),
            proxy_port=int(raw_proxy) if raw_proxy is not None else None,
            launches=tuple(payload.get(_STATE_KEY_LAUNCHES) or ()),
            engine_pin=payload.get(_STATE_KEY_ENGINE_PIN),
            origin=origin,
            ready_at=float(raw_ready) if raw_ready is not None else None,
            swap_binary=str(payload.get(_STATE_KEY_SWAP_BINARY) or ""),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _start_stamp_fields(pid: int) -> dict[str, object]:
    """The record keys that name *pid* across a wall-clock step, for this platform.

    Linux: boot id and start ticks. macOS: the boot session and the mach start, under
    their own key. Empty on Windows, where the create time is not moved by a step, and
    when the read fails; the record then names the process by its create time alone.
    """
    linux = linux_start_stamp(pid)
    if linux is not None:
        return {_STATE_KEY_BOOT_ID: linux[0], _STATE_KEY_START_TICKS: linux[1]}
    mach = macos_start_stamp(pid)
    if mach is not None:
        return {_STATE_KEY_MACH_START: {_MACH_START_BOOT: mach[0], _MACH_START_ABSTIME: mach[1]}}
    return {}


def _load_mach_start(raw: object) -> tuple[str, int] | None:
    """The (boot session, mach start) of a ``mach_start`` value, or None when it is absent."""
    if not isinstance(raw, dict):
        return None
    return str(raw[_MACH_START_BOOT]), int(raw[_MACH_START_ABSTIME])


def _is_live_llama_swap(state: SwapState) -> bool:
    """True when the recorded pid is alive and is the recorded llama-swap.

    A recorded identity that differs from the live process's is pid reuse,
    even when the recycled pid runs another instance's llama-swap. On Linux
    the identity is the boot id plus start ticks; on macOS it is the boot session
    plus the start on the mach clock. Both are compared exactly: a wall-clock step
    moves psutil's create time but not these. A record without either (Windows,
    or a macOS record from before the mach start) compares the create time within
    a tolerance. A legacy state file without a create time falls back to the
    cmdline match alone.
    """
    try:
        proc = psutil.Process(state.pid)
        cmdline = proc.cmdline()
        create_time = proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    if state.boot_id is not None and state.start_ticks is not None:
        if linux_start_stamp(state.pid) != (state.boot_id, state.start_ticks):
            return False
    elif state.mach_start is not None:
        if macos_start_stamp(state.pid) != state.mach_start:
            return False
    elif (
        state.created_at is not None
        and abs(create_time - state.created_at) > _CREATE_TIME_TOLERANCE_S
    ):
        return False
    binary = Path(next(iter(cmdline), "")).name
    return _LLAMA_SWAP_PROCESS_NAME in binary


def _stop_stale_swap(state: SwapState) -> None:
    """TERM-then-KILL a stale llama-swap's group and reap the servers it spawned.

    Swept as wide as ``_stop_own_fleet``: a reparented or respawned server is no
    longer a descendant, and every caller unlinks the record next, so the member
    ports are the last thing that can match it.
    """
    children = _live_children(state.pid)
    try:
        proc = psutil.Process(state.pid)
    except psutil.NoSuchProcess:
        proc = None
    if proc is not None:
        _signal_stale(state, signal.SIGTERM)
        try:
            proc.wait(timeout=_ORPHAN_STOP_TIMEOUT_S)
        except psutil.TimeoutExpired:
            _signal_stale(state, _SIGKILL)
            _await_killed([proc])
    _reap_survivors(children + _find_orphan_servers(state.member_ports))


def _signal_stale(state: SwapState, sig: int) -> None:
    """Signal the stale swap's process group, or the pid where groups don't apply."""
    if state.pgid is not None and sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(state.pgid, sig)
        return
    with contextlib.suppress(psutil.NoSuchProcess):
        psutil.Process(state.pid).send_signal(sig)


def _stop_recorded_engine(state: SwapState) -> bool:
    """Terminate a live llama-swap and its servers, or reap the servers a dead one
    orphaned (matched by recorded port, since they run in their own process groups
    and outlive the swap). Returns whether anything was actually alive to stop, so
    the off switch reports a stale record as "nothing stopped" rather than a false
    success.
    """
    if _is_live_llama_swap(state):
        _stop_stale_swap(state)
        return True
    orphans = _find_orphan_servers(state.member_ports)
    _reap_survivors(orphans)
    return bool(orphans)


def _kill_recorded_engine(state: SwapState) -> bool:
    """SIGKILL a live llama-swap's group and its servers at once; whether any was alive.

    The recorded pid's children are read only once it is still llama-swap, as the
    graceful stop does: a reused pid's children belong to another program.
    """
    servers: list[psutil.Process] = []
    swaps: list[psutil.Process] = []
    if _is_live_llama_swap(state):
        servers = _live_children(state.pid)
        with contextlib.suppress(psutil.NoSuchProcess):
            swaps.append(psutil.Process(state.pid))
        _signal_stale(state, _SIGKILL)
    servers.extend(_find_orphan_servers(state.member_ports))
    for server in servers:
        with contextlib.suppress(psutil.NoSuchProcess):
            server.kill()
    _await_killed(swaps + servers)
    return bool(swaps or servers)


def _find_orphan_servers(ports: tuple[int, ...]) -> list[psutil.Process]:
    """Live servers of any engine that outlived their swap.

    Each engine adapter declares its server's executable name and, for an engine
    whose command line points into a folder of its own, that folder. A process
    matches on the name plus a member ``--port`` value or that folder, so an
    unrelated process of the same name on another port is never killed. A server
    below a live llama-swap belongs to a current run and is spared.
    """
    if not ports:
        return []
    targets = {str(port) for port in ports}
    return [proc for server in _server_processes() for proc in _orphans_of(server, targets)]


def _server_processes() -> tuple[ServerProcess, ...]:
    """The distinct server identities that the shipped engine adapters declare."""
    from lilbee.providers.fleet.engines.choice import ADAPTERS

    return tuple(dict.fromkeys(adapter.server_process for adapter in ADAPTERS.values()))


def _orphans_of(server: ServerProcess, targets: set[str]) -> list[psutil.Process]:
    """Live *server* processes on a *targets* port or in its folder, with no swap above."""
    root = None if server.folder is None else Path(server.folder())
    orphans: list[psutil.Process] = []
    for proc in _server_candidates(server):
        try:
            cmdline = proc.cmdline()
            name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not _runs_server(name, cmdline, server):
            continue
        on_port = _port_argument(cmdline) in targets
        in_folder = root is not None and any(_is_under(arg, root) for arg in cmdline)
        if (on_port or in_folder) and not _has_live_swap_ancestor(proc):
            orphans.append(proc)
    return orphans


def _server_candidates(server: ServerProcess) -> Iterator[psutil.Process]:
    """Live processes whose name is *server*'s, or an interpreter's for a script server."""
    for proc in psutil.process_iter(["name"]):
        name = proc.info["name"] or ""
        if _is_server_name(name, server) or (server.interpreted and _is_interpreter(name)):
            yield proc


def _is_server_name(name: str, server: ServerProcess) -> bool:
    """True when *name* is the executable name of *server*, whole."""
    return name in (server.name, f"{server.name}{_EXE_SUFFIX}")


def _is_interpreter(name: str) -> bool:
    return name.lower().startswith(_INTERPRETER_PREFIX)


def _runs_server(name: str, cmdline: list[str], server: ServerProcess) -> bool:
    """True when a candidate of *server* called *name* with *cmdline* runs it.

    A candidate is named for the server or, for a script server, an interpreter.
    """
    if _is_server_name(name, server):
        return True
    script = Path(next((arg for arg in cmdline[1:] if not arg.startswith("-")), "")).name
    return _is_server_name(script, server) or script in server.scripts


def _is_under(arg: str, root: Path) -> bool:
    """True when *arg* is the folder *root* or a path below it."""
    path = Path(arg)
    return path == root or root in path.parents


def _has_live_swap_ancestor(proc: psutil.Process) -> bool:
    """True when a parent above *proc* is a live llama-swap.

    The walk goes one parent at a time and stops at the first llama-swap, so an
    ancestor that vanishes or cannot be read above it never hides it. A hop that
    cannot be read before any llama-swap ends the walk as not spared: a server
    reparented to init is the orphan this sweep exists for.
    """
    current = proc
    while True:
        try:
            parent = current.parent()
            if parent is None:
                return False
            if _LLAMA_SWAP_PROCESS_NAME in parent.name():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        current = parent


def _state_owner_pid(name: str) -> int | None:
    """Owner pid embedded in a state or state-tmp filename, ``None`` when absent.

    Handles both the group-qualified form (``llama-swap.state.chat.123.json``)
    and the legacy pre-group form (``llama-swap.state.123.json``): the pid is
    always the last dotted segment of the stem.
    """
    stem = name.removeprefix(_STATE_TMP_PREFIX).removesuffix(_STATE_TMP_SUFFIX)
    stem = stem.removeprefix(_STATE_FILENAME_PREFIX).removesuffix(_STATE_FILENAME_SUFFIX)
    try:
        return int(stem.rsplit(".", 1)[-1])
    except ValueError:
        return None


def _state_group(name: str) -> str | None:
    """Group token from a group-qualified state filename, ``None`` for legacy names.

    ``llama-swap.state.chat.123.json`` -> ``chat``; the legacy pre-group form
    ``llama-swap.state.123.json`` has no group token.
    """
    stem = name.removeprefix(_STATE_FILENAME_PREFIX).removesuffix(_STATE_FILENAME_SUFFIX)
    head, _, _ = stem.rpartition(".")  # drop the trailing pid; group is what remains
    return head or None


def _config_filename(pid: int, group: str) -> str:
    """This owner's config filename for *group* (``llama-swap-<group>.<pid>.json``)."""
    return _CONFIG_FILENAME_TEMPLATE.format(group=group, pid=pid)


def _config_owner_pid(name: str) -> int | None:
    """Owner pid embedded in a config filename, ``None`` for a legacy pid-less name."""
    stem = name.removeprefix("llama-swap-").removesuffix(".json")
    try:
        return int(stem.rsplit(".", 1)[-1])
    except ValueError:
        return None


def _port_argument(cmdline: list[str]) -> str | None:
    """The value following the port flag in *cmdline*, or ``None``."""
    for flag, value in itertools.pairwise(cmdline):
        if flag == PORT_FLAG:
            return value
    return None
