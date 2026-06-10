"""Supervise the single llama-swap process that fronts every fleet role.

llama-swap owns each role's llama-server lifecycle; this manages the one proxy
process and exposes its endpoint and readiness. See docs/architecture.md.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import psutil

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.binary import resolve_llama_swap
from lilbee.providers.fleet.launch import role_model_prefix
from lilbee.providers.fleet.swap_config import build_swap_config

if TYPE_CHECKING:
    from lilbee.providers.fleet.launch import InstanceLaunch
    from lilbee.providers.roles import WorkerRole

_HOST = "127.0.0.1"
_CONFIG_FILENAME = "llama-swap.json"
# Cross-run reaping: each owner lilbee writes its own state file (named with its
# pid) recording its swap's pid/pgid plus the owner's pid and create time, so the
# next start can kill a dead owner's surviving llama-swap (it holds VRAM
# otherwise) while leaving a live owner's swap and file alone.
_STATE_FILENAME_PREFIX = "llama-swap.state."
_STATE_FILENAME_SUFFIX = ".json"
# Also matches the legacy single shared state file ("llama-swap.state.json").
_STATE_FILE_GLOB = f"{_STATE_FILENAME_PREFIX}*"
_STATE_KEY_PID = "pid"
_STATE_KEY_PGID = "pgid"
_STATE_KEY_CREATED_AT = "created_at"
_STATE_KEY_OWNER_PID = "owner_pid"
_STATE_KEY_OWNER_CREATED_AT = "owner_created_at"
_STATE_KEY_NAME = "name"
# Pid reuse guard: a live process at a recorded pid whose create time differs
# from the recorded one by more than this is a different process.
_CREATE_TIME_TOLERANCE_S = 1.0
_LLAMA_SWAP_PROCESS_NAME = "llama-swap"
_CONFIG_FLAG = "-config"
_LISTEN_FLAG = "-listen"
_HEALTH_PATH = "/health"
_RUNNING_PATH = "/running"
# llama-swap's own proxy answers within a second; upstream model loads have their
# own (longer) budget inside llama-swap, so this only covers the proxy coming up.
_BOOT_TIMEOUT_S = 30.0
_BOOT_POLL_S = 0.25
_STOP_TIMEOUT_S = 10.0
# Grace for a llama-server that outlived llama-swap before it is force-killed.
_ORPHAN_STOP_TIMEOUT_S = 5.0
_PROBE_TIMEOUT_S = 5.0
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


def _state_filename(owner_pid: int) -> str:
    """The per-owner state filename for the lilbee process *owner_pid*."""
    return f"{_STATE_FILENAME_PREFIX}{owner_pid}{_STATE_FILENAME_SUFFIX}"


@dataclass(frozen=True)
class _SwapState:
    """A previous run's llama-swap identity, read back for cross-run reaping."""

    pid: int
    pgid: int | None
    owner_pid: int | None
    owner_created_at: float | None
    created_at: float | None = None


class SwapManager:
    """Owns one llama-swap process fronting every configured role co-resident."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._config_path = data_dir / _CONFIG_FILENAME
        self._state_path = data_dir / _state_filename(os.getpid())
        self._proc: subprocess.Popen[bytes] | None = None
        self._port: int | None = None

    def start(self, launches: list[InstanceLaunch]) -> None:
        """Write the config and spawn llama-swap, waiting for its proxy to answer.

        The proxy and every member get a freshly allocated free port; a fixed
        member port range would collide with a previous instance's server that
        is still shutting down (the new llama-server then fails its bind and
        llama-swap reports it only as "exited prematurely").
        """
        # Idempotent safety net; the provider reaps before planning so the GPU
        # probe already saw the real free memory.
        self.reap_stale()
        ports = _pick_free_ports(1 + len(launches))
        member_ports = dict(zip([launch.model_id for launch in launches], ports[1:], strict=True))
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(build_swap_config(launches, member_ports))
        self._port = ports[0]
        self._proc = subprocess.Popen(  # noqa: S603 - argv[0] is the resolved llama-swap
            [
                str(resolve_llama_swap()),
                _CONFIG_FLAG,
                str(self._config_path),
                _LISTEN_FLAG,
                f"{_HOST}:{self._port}",
            ],
            start_new_session=True,
            creationflags=_CREATE_NEW_PROCESS_GROUP,
        )
        self._write_state()
        self._await_health()

    def reap_stale(self) -> None:
        """Kill every dead owner's surviving llama-swap.

        An OOM-killed lilbee leaves llama-swap (and its servers) holding VRAM,
        so planning would otherwise see artificially reduced free memory; the
        provider calls this before its GPU probe. Every state file in the data
        dir is scanned (including the legacy shared-name file): a dead owner's
        swap is reaped and its file removed; a live owner's swap and file are
        left alone, so a second lilbee at the same data_dir (e.g. ``lilbee
        sync`` beside the server) never kills the live owner's healthy swap.
        Recorded create times guard against owner- and swap-pid reuse; the
        cmdline match covers legacy files without a swap create time.
        """
        for state_path in sorted(self._data_dir.glob(_STATE_FILE_GLOB)):
            state = _load_state(state_path)
            if state is not None and _owner_alive(state.owner_pid, state.owner_created_at):
                continue
            state_path.unlink(missing_ok=True)
            if state is not None and _is_live_llama_swap(state):
                _stop_stale_swap(state)

    def _write_state(self) -> None:
        """Record the swap's pid/pgid/create time plus our pid and create time."""
        proc = self._proc
        if proc is None:
            return
        pgid: int | None = None
        if sys.platform != "win32":
            with contextlib.suppress(ProcessLookupError):
                pgid = os.getpgid(proc.pid)
        created_at: float | None = None
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            created_at = psutil.Process(proc.pid).create_time()
        state = {
            _STATE_KEY_PID: proc.pid,
            _STATE_KEY_PGID: pgid,
            _STATE_KEY_CREATED_AT: created_at,
            _STATE_KEY_OWNER_PID: os.getpid(),
            _STATE_KEY_OWNER_CREATED_AT: psutil.Process().create_time(),
            _STATE_KEY_NAME: _LLAMA_SWAP_PROCESS_NAME,
        }
        self._state_path.write_text(json.dumps(state))

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

    def reload(self, launches: list[InstanceLaunch]) -> None:
        """Apply a changed model set by restarting llama-swap with a fresh config."""
        self.shutdown()
        self.start(launches)

    @property
    def running(self) -> bool:
        """Whether this manager currently has a spawned llama-swap process."""
        return self._proc is not None

    def shutdown(self) -> None:
        """Stop llama-swap and every server it spawned (a no-op when not running).

        Unlinks only this owner's state file; another instance's record stays.
        """
        proc = self._proc
        if proc is not None:
            _stop_process_tree(proc)
            self._state_path.unlink(missing_ok=True)
        self._proc = None
        self._port = None

    def _await_health(self) -> None:
        """Poll the proxy's /health until it answers, or fail with a clear error."""
        url = f"{self.endpoint()}{_HEALTH_PATH}"
        deadline = time.monotonic() + _BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                self._fail("The local model engine exited before it was ready.")
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(url, timeout=_PROBE_TIMEOUT_S).status_code == httpx.codes.OK:
                    return
            time.sleep(_BOOT_POLL_S)
        self._fail("The local model engine did not start in time.")

    def _ready_models(self) -> set[str]:
        """Model ids whose upstream is loaded and ready, per llama-swap's /running."""
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError, TypeError):
            payload = httpx.get(
                f"{self.endpoint()}{_RUNNING_PATH}", timeout=_PROBE_TIMEOUT_S
            ).json()
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


def _stop_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Stop llama-swap and reap any llama-server that outlives it.

    llama-swap starts each server in its own process group (its Setpgid), so
    signalling llama-swap's group never reaches the servers; a survivor keeps
    its port bound and that port's next bind fails. The children are captured
    while llama-swap is still alive, then swept after it stops.
    """
    children = _live_children(proc.pid)
    if sys.platform == "win32":
        _hard_stop(proc)
    else:
        _terminate_group(proc)
    _reap_survivors(children)


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


def _terminate_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the process group, escalating to SIGKILL on timeout."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:  # pragma: no cover - process exited between checks
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, _SIGKILL)


def _hard_stop(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the process, escalating to a hard kill on timeout (Windows path)."""
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()


def _load_state(path: Path) -> _SwapState | None:
    """Parse a state file into a :class:`_SwapState`; ``None`` when absent/corrupt."""
    try:
        payload = json.loads(path.read_text())
        raw_pgid = payload.get(_STATE_KEY_PGID)
        raw_owner = payload.get(_STATE_KEY_OWNER_PID)
        raw_owner_created = payload.get(_STATE_KEY_OWNER_CREATED_AT)
        raw_created = payload.get(_STATE_KEY_CREATED_AT)
        return _SwapState(
            pid=int(payload[_STATE_KEY_PID]),
            pgid=int(raw_pgid) if raw_pgid is not None else None,
            owner_pid=int(raw_owner) if raw_owner is not None else None,
            owner_created_at=float(raw_owner_created) if raw_owner_created is not None else None,
            created_at=float(raw_created) if raw_created is not None else None,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _owner_alive(pid: int | None, created_at: float | None) -> bool:
    """True when the recorded owner lilbee process is still running (not a zombie).

    A live process at *pid* whose create time differs from *created_at* is a
    pid-reuse impostor, so the owner counts as dead.
    """
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        status = str(proc.status())
        create_time = proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    if created_at is not None and abs(create_time - created_at) > _CREATE_TIME_TOLERANCE_S:
        return False
    return status != str(psutil.STATUS_ZOMBIE)


def _is_live_llama_swap(state: _SwapState) -> bool:
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


def _stop_stale_swap(state: _SwapState) -> None:
    """TERM-then-KILL a stale llama-swap's group and reap the servers it spawned."""
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
    _reap_survivors(children)


def _signal_stale(state: _SwapState, sig: int) -> None:
    """Signal the stale swap's process group, or the pid where groups don't apply."""
    if state.pgid is not None and sys.platform != "win32":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(state.pgid, sig)
        return
    with contextlib.suppress(psutil.NoSuchProcess):
        psutil.Process(state.pid).send_signal(sig)
