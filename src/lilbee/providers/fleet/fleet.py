"""Supervisor for the managed llama-server fleet.

Each instance runs in its own process group (so a stop/crash kills the whole
group; PID-only kills strand GPU memory), claims its port at spawn time (no
batch up-front allocation that races), and records ``pid``/``port`` to a file so
a crashed parent's orphans are reaped on the next start. A background monitor
restarts a server that dies and tracks health so the router skips a degraded
one. See docs/architecture.md for the device/placement rationale.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

_HOST = "127.0.0.1"
_READY_TIMEOUT_S = 180.0
_READY_POLL_S = 0.5
_STOP_TIMEOUT_S = 10.0
_MONITOR_POLL_S = 2.0
_PORT_BIND_RETRIES = 8
# Restart backoff (seconds), clamped to the last entry for repeated crashes.
_RESTART_BACKOFF_S = (1.0, 2.0, 5.0, 10.0, 30.0)
# Give up restarting a server after this many CONSECUTIVE failed (re)starts; the
# role then stays down (calls raise a user-facing ProviderError) instead of
# crash-looping forever (e.g. a model that OOMs at launch because the VRAM
# estimate under-shot).
_MAX_RESTART_ATTEMPTS = 5
# How much of a failed server's captured stderr to surface in the warning log.
_STDERR_TAIL_CHARS = 2000
# Windows puts the child in a new process group via creationflags; POSIX uses
# start_new_session. The constant is Windows-only in stdlib (0 = no-op on POSIX).
_CREATE_NEW_PROCESS_GROUP: int = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
# SIGKILL is POSIX-only; resolve dynamically so teardown stays importable on Windows.
_SIGKILL: int = getattr(signal, "SIGKILL", signal.SIGTERM)


def pick_free_port() -> int:
    """Bind an ephemeral localhost port and return it (closed before reuse)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


@dataclass
class InstanceLaunch:
    """Everything needed to spawn one server, minus the port (claimed at spawn)."""

    role: WorkerRole
    argv: list[str]  # llama-server command WITHOUT --port; spawn appends it
    env_overrides: dict[str, str]  # backend-specific device-pinning env
    model: str
    port_file: Path
    token_cap: int | None = None  # per-slot ctx for embed/rerank input truncation


class FleetServer:
    """One supervised llama-server process in its own process group."""

    def __init__(self, launch: InstanceLaunch) -> None:
        self._launch = launch
        self._proc: subprocess.Popen[bytes] | None = None
        self.client: LlamaServerClient | None = None
        self.restarts = 0
        # Consecutive failed (re)starts; reset to 0 on a successful ready. Bounds
        # the crash-loop so a server that can't start stops being respawned.
        self.consecutive_failures = 0
        # ``ready`` gates routing: True only after wait_ready passes, so a server
        # mid-(re)start (alive but model still loading) is never routed to.
        self.ready = False

    @property
    def _stderr_log(self) -> Path:
        """Per-instance stderr capture file (sibling of the port file)."""
        return self._launch.port_file.with_suffix(".log")

    @property
    def gave_up(self) -> bool:
        """True once consecutive failures hit the cap; the role then stays down."""
        return self.consecutive_failures >= _MAX_RESTART_ATTEMPTS

    @property
    def role(self) -> WorkerRole:
        return self._launch.role

    def spawn(self) -> LlamaServerClient:
        """Claim a port, launch the process group, record pids/port, return client."""
        port = pick_free_port()
        argv = [*self._launch.argv, "--port", str(port)]
        env = {**os.environ, **self._launch.env_overrides}
        # Capture stderr to a per-instance file (truncated each spawn) so a failed
        # launch is diagnosable; a file (not a pipe) needs no parent drain and
        # cannot deadlock the child. The parent's handle is closed immediately --
        # the child keeps its own dup.
        stderr_fh = self._stderr_log.open("wb")
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv is a fixed template
                argv,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=stderr_fh,
                start_new_session=sys.platform != "win32",
                creationflags=_CREATE_NEW_PROCESS_GROUP,
            )
        finally:
            stderr_fh.close()
        # parent_pid identifies the owning lilbee, so reaping never kills a
        # concurrent instance's servers (only a dead parent's orphans).
        self._launch.port_file.write_text(
            json.dumps({"parent_pid": os.getpid(), "pid": self._proc.pid, "port": port})
        )
        if self.client is not None:
            self.client.close()
        self.client = LlamaServerClient(
            f"http://{_HOST}:{port}", self._launch.model, token_cap=self._launch.token_cap
        )
        return self.client

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait_ready(self, timeout: float = _READY_TIMEOUT_S) -> bool:
        """Poll ``/health`` until ready (200 == model loaded), death, or timeout.

        A few bind retries cover the rare case where the claimed port was taken
        between selection and the child binding it (the process dies at once).
        """
        for _ in range(_PORT_BIND_RETRIES):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self.is_alive():
                    break  # likely a port-bind failure; respawn on a fresh port
                if self.client is not None and self.client.health():
                    return True
                time.sleep(_READY_POLL_S)
            if self.is_alive():
                return False  # alive but never became ready within the timeout
            self.spawn()  # dead before ready -> retry with a new port
        return False

    def stop(self) -> None:
        """Kill the process group (SIGTERM -> SIGKILL), close the client, clean up."""
        if self._proc is not None and self._proc.poll() is None:
            _terminate_group(self._proc)
        if self.client is not None:
            self.client.close()
        self._launch.port_file.unlink(missing_ok=True)
        self._stderr_log.unlink(missing_ok=True)

    def restart(self) -> bool:
        """Stop the dead process and respawn on a fresh port; True if ready.

        Tracks consecutive failures so the monitor can stop respawning a server
        that never comes up (see ``gave_up``); a success resets the count.
        """
        self.restarts += 1
        if self._proc is not None and self._proc.poll() is None:
            _terminate_group(self._proc)
        self.spawn()
        ready = self.wait_ready()
        self.consecutive_failures = 0 if ready else self.consecutive_failures + 1
        return ready

    def failed_start_detail(self) -> str:
        """Tail of the server's captured stderr, for diagnosing a failed launch."""
        try:
            text = self._stderr_log.read_text(errors="replace")
        except OSError:
            return ""
        return text.strip()[-_STDERR_TAIL_CHARS:]


def _terminate_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the whole process group, escalating to SIGKILL on timeout.

    Windows has no process groups via this path, so it hard-stops the process.
    """
    if sys.platform == "win32":
        _hard_stop(proc)
        return
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


def _kill_pid_group(pid: int) -> None:
    """Best-effort kill of a recorded orphan PID's group (reaper on restart)."""
    try:
        if sys.platform == "win32":
            os.kill(pid, signal.SIGTERM)
        else:
            os.killpg(os.getpgid(pid), _SIGKILL)
    except (OSError, ProcessLookupError):
        return  # already gone


def _is_pid_alive(pid: int) -> bool:
    """Whether *pid* is still running. Conservative: 'assume alive' if unsure.

    Windows has no safe ``os.kill(pid, 0)`` liveness probe (a 0 signal can
    terminate), so there we never reap (a leaked orphan beats killing a live one).
    """
    if sys.platform == "win32":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but not signalable (e.g. EPERM) -> alive
    return True


def reap_orphans(data_dir: Path) -> None:
    """Kill llama-server processes a *dead* parent left behind, then clear its files.

    A clean shutdown removes its port files. Any left behind belong either to a
    crashed parent (reap them, they strand GPU memory) or a concurrent live
    instance (leave them alone) -- distinguished by whether the recorded parent
    lilbee pid is still alive.
    """
    for port_file in data_dir.glob("llama-server-*.port"):
        try:
            record = json.loads(port_file.read_text())
            parent_pid = int(record["parent_pid"])
            pid = int(record["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            port_file.unlink(missing_ok=True)
            continue
        if _is_pid_alive(parent_pid):
            continue  # another live lilbee owns this server; do not touch it
        _kill_pid_group(pid)
        port_file.unlink(missing_ok=True)


class Fleet:
    """Owns running server instances; restarts crashes; serves healthy clients."""

    def __init__(
        self,
        *,
        ready_timeout: float = _READY_TIMEOUT_S,
        data_dir: Path | None = None,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        self.ready_timeout = ready_timeout
        self.data_dir = data_dir
        self._servers: list[FleetServer] = []
        self._lock = threading.RLock()
        self._monitor: threading.Thread | None = None
        self._stop_monitor = threading.Event()
        self._on_spawning = on_spawning
        self._on_spawned = on_spawned

    def set_listener(
        self,
        *,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        """Attach spawn-lifecycle callbacks. Used by the TUI to report reloads."""
        self._on_spawning = on_spawning
        self._on_spawned = on_spawned

    def _notify(self, callback: Callable[[WorkerRole], None] | None, role: WorkerRole) -> None:
        """Fire a listener callback, swallowing its errors (UI feedback is best-effort)."""
        if callback is not None:
            with contextlib.suppress(Exception):
                callback(role)

    def _bring_up(self, server: FleetServer) -> None:
        """Spawn one server and wait for readiness, reporting to listeners.

        A server that never becomes ready is left not-ready (routing skips it) and
        its failure counter is bumped; the monitor keeps retrying it up to the cap.
        """
        self._notify(self._on_spawning, server.role)
        server.spawn()
        server.ready = server.wait_ready(timeout=self.ready_timeout)
        if server.ready:
            self._notify(self._on_spawned, server.role)
        else:
            server.consecutive_failures += 1
            log.warning(
                "llama-server for role %s did not become ready; calls to that role "
                "will error until it recovers. stderr tail: %s",
                server.role,
                server.failed_start_detail(),
            )

    def start(self, launches: list[InstanceLaunch]) -> None:
        """Reap prior orphans, spawn every launch, wait for readiness, monitor.

        Each role is independent: a server that fails to become ready is left
        not-ready (its calls error) while the others still serve. The monitor
        keeps retrying a dead one (bounded by the restart cap).
        """
        if self.data_dir is not None:
            reap_orphans(self.data_dir)
        for launch in launches:
            server = FleetServer(launch)
            self._servers.append(server)
            self._bring_up(server)
        if self._servers:
            self._start_monitor()

    def restart_role(self, role: WorkerRole, launches: list[InstanceLaunch]) -> None:
        """Replace *role*'s servers with *launches*; other roles keep serving.

        Stops the role's current servers first (so a model swap never holds two
        copies' VRAM at once), then brings up the replacements outside the routing
        lock so other roles route uninterrupted. An empty *launches* (the role was
        unconfigured) just stops the old servers. A concurrent shutdown is honored:
        fresh servers are stopped rather than stranded.
        """
        with self._lock:
            old = [s for s in self._servers if s.role == role]
            self._servers = [s for s in self._servers if s.role != role]
        for server in old:
            server.stop()
        fresh: list[FleetServer] = []
        for launch in launches:
            if self._stop_monitor.is_set():
                break
            server = FleetServer(launch)
            self._bring_up(server)
            fresh.append(server)
        with self._lock:
            if self._stop_monitor.is_set():
                for server in fresh:
                    server.stop()
                return
            self._servers.extend(fresh)
            if self._servers and self._monitor is None:
                self._start_monitor()

    def healthy_clients(self, role: WorkerRole) -> list[LlamaServerClient]:
        """Ready, live clients for *role*; a (re)starting server is excluded."""
        with self._lock:
            return [
                s.client
                for s in self._servers
                if s.role == role and s.ready and s.is_alive() and s.client is not None
            ]

    def _start_monitor(self) -> None:
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()

    def _monitor_loop(self) -> None:
        while not self._stop_monitor.wait(_MONITOR_POLL_S):
            self._restart_dead()

    def _restart_dead(self) -> None:
        """Restart any dead server that hasn't hit the restart cap, with backoff.

        The slow respawn (spawn + wait_ready) runs OUTSIDE the lock; the server is
        marked not-ready first so routing skips it, and ready again only once it is.
        Holding the lock across wait_ready would block all routing for the timeout.
        A server past the cap is left dead (calls to its role error) instead of
        being respawned forever.
        """
        with self._lock:
            dead = [s for s in self._servers if not s.is_alive() and not s.gave_up]
            for server in dead:
                server.ready = False
        for server in dead:
            self._backoff(server.restarts)
            if self._stop_monitor.is_set():
                return
            if server.is_alive():
                continue
            ready = server.restart()
            with self._lock:
                if self._stop_monitor.is_set():
                    server.stop()  # shutdown raced our respawn; don't strand it
                    continue
                server.ready = ready and server.is_alive()
            if not server.ready and server.gave_up:
                log.warning(
                    "llama-server for role %s failed %d consecutive restarts; leaving "
                    "that role down. stderr tail: %s",
                    server.role,
                    server.consecutive_failures,
                    server.failed_start_detail(),
                )

    @staticmethod
    def _backoff(restarts: int) -> None:
        idx = min(restarts, len(_RESTART_BACKOFF_S) - 1)
        time.sleep(_RESTART_BACKOFF_S[idx])

    def shutdown(self) -> None:
        """Stop the monitor, then stop every instance (group-kill) and forget them."""
        self._stop_monitor.set()
        if self._monitor is not None:
            self._monitor.join(timeout=_STOP_TIMEOUT_S)
            self._monitor = None
        with self._lock:
            for server in self._servers:
                server.stop()
            self._servers.clear()
