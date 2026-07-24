"""Supervise the single llama-swap process that fronts every fleet role.

llama-swap owns each role's llama-server lifecycle; this manages the one proxy
process and exposes its endpoint and readiness. See docs/architecture.md.
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
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import httpx
import psutil

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.binary import engine_pin, resolve_llama_swap
from lilbee.providers.fleet.child_guard import spawn_llama_swap
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import role_model_prefix
from lilbee.providers.fleet.swap_config import PORT_FLAG, build_swap_config
from lilbee.runtime.engine_lock import clear_keep_warm

if TYPE_CHECKING:
    from lilbee.providers.fleet.launch import InstanceLaunch
    from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

_HOST = "127.0.0.1"
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
# llama-swap's request log onto the screen and corrupt the render. Per-model
# upstream logs are unaffected (those go to llama-swap's /logs API).
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
_STATE_KEY_NAME = "name"
_STATE_KEY_MEMBER_PORTS = "member_ports"
_STATE_KEY_PROXY_PORT = "proxy_port"
_STATE_KEY_LAUNCHES = "launches"
_STATE_KEY_ENGINE_PIN = "engine_pin"
# Atomic state writes: the dot prefix keeps half-written tmp files out of the
# reap scan's glob.
_STATE_TMP_PREFIX = "."
_STATE_TMP_SUFFIX = ".tmp"
# Pid reuse guard: a live process at a recorded pid whose create time differs
# from the recorded one by more than this is a different process.
_CREATE_TIME_TOLERANCE_S = 1.0
_LLAMA_SWAP_PROCESS_NAME = "llama-swap"
_LLAMA_SERVER_PROCESS_NAME = "llama-server"
_CONFIG_FLAG = "-config"
_LISTEN_FLAG = "-listen"
_HEALTH_PATH = "/health"
_RUNNING_PATH = "/running"
_HTTP_TIMEOUT_S = 10.0
# llama-swap's own proxy answers within a second; upstream model loads have their
# own (longer) budget inside llama-swap, so this only covers the proxy coming up.
_BOOT_TIMEOUT_S = 30.0
_BOOT_POLL_S = 0.25
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


_PROVIDER = "llama-server"
# /running JSON shape: {"running": [{"model": <id>, "state": "ready", ...}, ...]}.
_KEY_RUNNING = "running"
_KEY_MODEL = "model"
_KEY_STATE = "state"
_STATE_READY = "ready"


def _platform_const(module: object, name: str, default: int) -> int:
    """A platform-conditional stdlib constant (absent on some OSes -> default)."""
    return getattr(module, name, default)


_CREATE_NEW_PROCESS_GROUP: int = _platform_const(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
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
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _state_filename(owner_pid: int, group: str) -> str:
    """The per-owner, per-group state filename for the lilbee process *owner_pid*."""
    return f"{_STATE_FILENAME_PREFIX}{group}.{owner_pid}{_STATE_FILENAME_SUFFIX}"


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
    member_ports: tuple[int, ...] = ()
    proxy_port: int | None = None
    launches: tuple[dict, ...] = ()
    engine_pin: str | None = None


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
        self._state_path = data_dir / _state_filename(os.getpid(), group.value)
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: BinaryIO | None = None
        self._port: int | None = None
        self._member_ports: list[int] = []
        # The serving contract (per-role model/ctx/slots) persisted in every
        # state write, so a guest lilbee can bind to this live fleet.
        self._launches_payload: list[dict] = []
        # True when this manager uses an engine another process built: it then
        # never writes state, never reaps, and never signals engine processes.
        self._bound = False

    def start(
        self, launches: list[InstanceLaunch], *, ttl_seconds: int = 0, bind_lifetime: bool = True
    ) -> None:
        """Write the config and spawn llama-swap, waiting for its proxy to answer.

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

        ``bind_lifetime`` binds the engine to this process so a crash cannot orphan
        it; it is False for a keep-warm fleet that is meant to outlive lilbee.
        """
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
        self._launches_payload = [launch.to_state() for launch in launches]
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self._config_path,
            build_swap_config(
                launches, member_ports, swap=self._group.swaps, ttl_seconds=ttl_seconds
            ),
        )
        self._port = ports[0]
        # Capture llama-swap's stdout/stderr to a file so its access log never
        # reaches an inherited terminal (a TUI/CLI parent) and garbles the screen.
        self._close_log()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("ab")
        self._proc = spawn_llama_swap(
            [
                str(resolve_llama_swap()),
                _CONFIG_FLAG,
                str(self._config_path),
                _LISTEN_FLAG,
                f"{_HOST}:{self._port}",
            ],
            bind_lifetime=bind_lifetime,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            creationflags=_CREATE_NEW_PROCESS_GROUP,
        )
        self._write_state()
        self._await_health()

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
        state = {
            _STATE_KEY_PID: swap_pid,
            _STATE_KEY_PGID: pgid,
            _STATE_KEY_CREATED_AT: created_at,
            _STATE_KEY_NAME: _LLAMA_SWAP_PROCESS_NAME,
            _STATE_KEY_MEMBER_PORTS: self._member_ports,
            _STATE_KEY_PROXY_PORT: self._port,
            _STATE_KEY_LAUNCHES: self._launches_payload,
            _STATE_KEY_ENGINE_PIN: engine_pin(),
        }
        _atomic_write(self._state_path, json.dumps(state))

    def endpoint(self) -> str:
        """Base URL of the llama-swap OpenAI-compatible proxy."""
        if self._port is None:
            raise ProviderError(
                "The local model engine is not running.",
                provider=_PROVIDER,
                kind=ProviderErrorKind.SERVER,
            )
        return f"http://{_HOST}:{self._port}"

    def role_ready(self, role: WorkerRole) -> bool:
        """Whether at least one of *role*'s replica servers is loaded and ready."""
        prefix = role_model_prefix(role)
        return any(model.startswith(prefix) for model in self._ready_models())

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
        self._state_path.unlink(missing_ok=True)
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

    def _ready_models(self) -> set[str]:
        """Model ids whose upstream is loaded and ready, per llama-swap's /running.

        A read-only probe: a concurrent shutdown can clear ``_port`` between the
        caller's check and ``endpoint()``, raising ProviderError, so that is
        suppressed too and the probe reports "nothing ready" rather than throwing.
        """
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError, TypeError, ProviderError):
            payload = (
                _probe_client()
                .get(f"{self.endpoint()}{_RUNNING_PATH}", timeout=_PROBE_TIMEOUT_S)
                .json()
            )
            return {
                entry[_KEY_MODEL]
                for entry in payload[_KEY_RUNNING]
                if entry.get(_KEY_STATE) == _STATE_READY
            }
        return set()

    def _fail(self, message: str) -> None:
        """Tear down and raise a user-facing engine-start error."""
        self.shutdown()
        raise ProviderError(message, provider=_PROVIDER, kind=ProviderErrorKind.SERVER)


def _pick_free_ports(count: int) -> list[int]:
    """Bind *count* ephemeral localhost ports at once and return them.

    All sockets stay open until every port is claimed so the OS cannot hand the
    same port out twice within one allocation.
    """
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind((_HOST, 0))
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


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


def _running_endpoint_answers(base_url: str) -> bool:
    """Whether *base_url* serves llama-swap's ``/running`` endpoint (identity, not
    just liveness).

    Proxy ports are ephemeral: after an engine dies, any unrelated local service
    that later binds the recorded port and returns a 2xx/3xx to an unknown path
    would pass a bare status check, so a dead record would look healthy forever and
    inference clients would bind to a non-engine endpoint. Requiring the ``running``
    JSON payload shape that only llama-swap produces makes the probe identity-checked.
    Total: any transport error or non-conforming body reads as "not our engine".
    """
    try:
        resp = _probe_client().get(f"{base_url}{_RUNNING_PATH}", timeout=_LIVENESS_TIMEOUT)
    except (OSError, httpx.HTTPError):
        return False
    if resp.status_code >= httpx.codes.BAD_REQUEST:
        return False
    try:
        return isinstance(resp.json().get(_KEY_RUNNING), list)
    except (ValueError, AttributeError):
        return False


def state_is_healthy(state: SwapState) -> bool:
    """Whether the engine behind *state* answers on its recorded proxy port."""
    if state.proxy_port is None:
        return False
    return _running_endpoint_answers(f"http://{_HOST}:{state.proxy_port}")


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
        if _stop_recorded_engine(state):
            group = _state_group(state_path.name)
            if group is not None:
                stopped.append(group)
        state_path.unlink(missing_ok=True)
    return stopped


def reap_stale(data_dir: Path) -> None:
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

    Module-level (not a method) because it must run before planning decides
    which role groups exist, when no per-group manager has been built yet.
    """
    _clean_stale_tmp_files(data_dir)
    _clean_stale_configs(data_dir)
    for state_path in sorted(data_dir.glob(_STATE_FILE_GLOB)):
        state = _load_state(state_path)
        if state is None:
            continue
        if state_is_healthy(state):
            # An answering engine is in use (bind accepts on exactly this
            # test); reaping must never disagree with binding.
            continue
        _stop_recorded_engine(state)
        state_path.unlink(missing_ok=True)


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
        payload = json.loads(path.read_text())
        raw_pgid = payload.get(_STATE_KEY_PGID)
        raw_created = payload.get(_STATE_KEY_CREATED_AT)
        raw_ports = payload.get(_STATE_KEY_MEMBER_PORTS) or []
        raw_proxy = payload.get(_STATE_KEY_PROXY_PORT)
        return SwapState(
            pid=int(payload[_STATE_KEY_PID]),
            pgid=int(raw_pgid) if raw_pgid is not None else None,
            created_at=float(raw_created) if raw_created is not None else None,
            member_ports=tuple(int(port) for port in raw_ports),
            proxy_port=int(raw_proxy) if raw_proxy is not None else None,
            launches=tuple(payload.get(_STATE_KEY_LAUNCHES) or ()),
            engine_pin=payload.get(_STATE_KEY_ENGINE_PIN),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _is_live_llama_swap(state: SwapState) -> bool:
    """True when the recorded pid is alive and is the recorded llama-swap.

    A recorded create time that differs from the live process's is pid reuse,
    even when the recycled pid runs another instance's llama-swap; a legacy
    state file without one falls back to the cmdline match alone.
    """
    try:
        proc = psutil.Process(state.pid)
        cmdline = proc.cmdline()
        create_time = proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    if state.created_at is not None and abs(create_time - state.created_at) > (
        _CREATE_TIME_TOLERANCE_S
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


def _find_orphan_servers(ports: tuple[int, ...]) -> list[psutil.Process]:
    """Live llama-server processes serving one of *ports*.

    Both the binary name and the ``--port`` value must match, so an unrelated
    process on a recycled port is never killed; a server whose parent is a
    live llama-swap belongs to a current run on a reused port and is spared.
    """
    if not ports:
        return []
    targets = {str(port) for port in ports}
    orphans: list[psutil.Process] = []
    for proc in _processes_named(_LLAMA_SERVER_PROCESS_NAME):
        try:
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # Identity is the --port value plus the absence of a live swap parent;
        # _processes_named already gated on the executable name (comm).
        if _port_argument(cmdline) in targets and not _has_live_swap_parent(proc):
            orphans.append(proc)
    return orphans


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


def _has_live_swap_parent(proc: psutil.Process) -> bool:
    """True when *proc*'s parent is a live llama-swap (the server is not orphaned)."""
    try:
        parent = proc.parent()
        if parent is None:
            return False
        return _LLAMA_SWAP_PROCESS_NAME in parent.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _port_argument(cmdline: list[str]) -> str | None:
    """The value following the port flag in *cmdline*, or ``None``."""
    for flag, value in itertools.pairwise(cmdline):
        if flag == PORT_FLAG:
            return value
    return None
