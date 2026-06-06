"""Supervise the single llama-swap process that fronts every fleet role.

llama-swap owns each role's llama-server lifecycle; this manages the one proxy
process and exposes its endpoint and readiness. See docs/architecture.md.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import httpx

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.binary import resolve_llama_swap
from lilbee.providers.fleet.launch import role_model_prefix
from lilbee.providers.fleet.swap_config import build_swap_config

if TYPE_CHECKING:
    from pathlib import Path

    from lilbee.providers.fleet.launch import InstanceLaunch
    from lilbee.providers.roles import WorkerRole

_HOST = "127.0.0.1"
_CONFIG_FILENAME = "llama-swap.json"
_CONFIG_FLAG = "-config"
_LISTEN_FLAG = "-listen"
_HEALTH_PATH = "/health"
_RUNNING_PATH = "/running"
# llama-swap's own proxy answers within a second; upstream model loads have their
# own (longer) budget inside llama-swap, so this only covers the proxy coming up.
_BOOT_TIMEOUT_S = 30.0
_BOOT_POLL_S = 0.25
_STOP_TIMEOUT_S = 10.0
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


class SwapManager:
    """Owns one llama-swap process fronting every configured role co-resident."""

    def __init__(self, data_dir: Path) -> None:
        self._config_path = data_dir / _CONFIG_FILENAME
        self._proc: subprocess.Popen[bytes] | None = None
        self._port: int | None = None

    def start(self, launches: list[InstanceLaunch]) -> None:
        """Write the config and spawn llama-swap, waiting for its proxy to answer."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(build_swap_config(launches))
        self._port = _pick_free_port()
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
        self._await_health()

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

    def shutdown(self) -> None:
        """Stop the llama-swap process group (a no-op when not running)."""
        proc = self._proc
        if proc is not None:
            _terminate_group(proc)
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


def _pick_free_port() -> int:
    """Bind an ephemeral localhost port and return it (closed before reuse)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_HOST, 0))
        return int(sock.getsockname()[1])


def _terminate_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the process group, escalating to SIGKILL on timeout.

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
