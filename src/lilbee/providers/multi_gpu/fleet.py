"""Supervisor for the llama-server sidecar fleet.

Each instance runs in its own process group so stop/crash kills the whole group
(PID-only kills strand GPU memory). Readiness is llama-server's ``/health``, which
returns 200 only once the model is loaded. ``Fleet`` spawns the planned instances,
waits for readiness, groups their clients by role, and tears the group down.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lilbee.providers.base import ProviderError
from lilbee.providers.multi_gpu.client import LlamaServerClient
from lilbee.providers.worker.transport import WorkerRole

_HOST = "127.0.0.1"
_READY_TIMEOUT_S = 180.0
_READY_POLL_S = 0.5
_STOP_TIMEOUT_S = 10.0
# Windows puts the child in a new process group via creationflags; POSIX uses
# start_new_session. The constant is Windows-only in stdlib, so resolve it
# dynamically (0 = no-op creationflags on POSIX) to keep one branchless line.
_CREATE_NEW_PROCESS_GROUP: int = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def pick_free_port() -> int:
    """Bind an ephemeral localhost port and return it (closed before reuse)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


def _child_env(devices: tuple[int, ...]) -> dict[str, str]:
    """Pin the child to *devices* via cross-vendor visible-device env vars."""
    env = dict(os.environ)
    visible = ",".join(str(d) for d in devices)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["GGML_VK_VISIBLE_DEVICES"] = visible
    return env


@dataclass
class InstanceLaunch:
    """Everything needed to spawn and address one server instance."""

    role: WorkerRole
    argv: list[str]
    devices: tuple[int, ...]
    port: int
    model: str
    port_file: Path


class FleetServer:
    """One supervised llama-server sidecar in its own process group."""

    def __init__(self, launch: InstanceLaunch) -> None:
        self._launch = launch
        self._proc: subprocess.Popen[bytes] | None = None
        self.client: LlamaServerClient | None = None

    @property
    def role(self) -> WorkerRole:
        return self._launch.role

    def spawn(self) -> LlamaServerClient:
        """Launch the process group, write the port file, return the client."""
        self._proc = subprocess.Popen(  # noqa: S603 - argv is built from a fixed template
            self._launch.argv,
            env=_child_env(self._launch.devices),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=sys.platform != "win32",
            creationflags=_CREATE_NEW_PROCESS_GROUP,
        )
        self._launch.port_file.write_text(str(self._launch.port))
        self.client = LlamaServerClient(f"http://{_HOST}:{self._launch.port}", self._launch.model)
        return self.client

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait_ready(self, timeout: float = _READY_TIMEOUT_S) -> bool:
        """Poll ``/health`` until ready, the process dies, or *timeout*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
                return False
            if self.client is not None and self.client.health():
                return True
            time.sleep(_READY_POLL_S)
        return False

    def stop(self) -> None:
        """Kill the process group (SIGTERM, escalate to SIGKILL) and clean up."""
        if self._proc is not None and self._proc.poll() is None:
            _terminate_group(self._proc)
        if self.client is not None:
            self.client.close()
        self._launch.port_file.unlink(missing_ok=True)


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
        os.killpg(pgid, signal.SIGKILL)


def _hard_stop(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the process, escalating to a hard kill on timeout (Windows path)."""
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()


@dataclass
class Fleet:
    """Owns the running server instances; maps roles to their clients."""

    ready_timeout: float = _READY_TIMEOUT_S
    _servers: list[FleetServer] = field(default_factory=list)

    def start(self, launches: list[InstanceLaunch]) -> dict[WorkerRole, list[LlamaServerClient]]:
        """Spawn every launch, wait for readiness, return clients grouped by role.

        On any instance failing to become ready, tears down the whole fleet and
        raises, so a partial fleet never serves.
        """
        by_role: dict[WorkerRole, list[LlamaServerClient]] = {}
        for launch in launches:
            server = FleetServer(launch)
            client = server.spawn()
            self._servers.append(server)
            if not server.wait_ready(timeout=self.ready_timeout):
                self.shutdown()
                raise ProviderError(f"llama-server for role {launch.role} failed to become ready")
            by_role.setdefault(launch.role, []).append(client)
        return by_role

    def shutdown(self) -> None:
        """Stop every instance (group-kill) and forget them."""
        for server in self._servers:
            server.stop()
        self._servers.clear()
