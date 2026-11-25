"""The launch argv of OpenVINO Model Server starts a process that the swap entry finds ready."""

from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from lilbee.providers.fleet.engines.base import LaunchRequest
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import RoleTarget, WorkerRole

_READY_WAIT_S = 15
_FAKE_OVMS = '''#!{python}
"""A stand-in ovms: refuses an argument the real server does not know, serves the ready route."""
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

KNOWN = {{"--model_path", "--model_name", "--rest_bind_address", "--rest_port",
          "--target_device", "--task", "--max_num_seqs"}}
args = sys.argv[1:]
flags = dict(zip(args[::2], args[1::2]))
if set(flags) - KNOWN or {{"--model_path", "--model_name", "--rest_port"}} - set(flags):
    sys.stderr.write("unknown or missing flag: " + " ".join(args) + "\\n")
    sys.exit(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        ok = self.path == "/v2/health/ready"
        self.send_response(200 if ok else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()


HTTPServer((flags.get("--rest_bind_address", "127.0.0.1"), int(flags["--rest_port"])),
           Handler).serve_forever()
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ready(url: str) -> bool:
    deadline = time.monotonic() + _READY_WAIT_S
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as reply:  # noqa: S310
                return bool(reply.status == 200)
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    return False


def test_the_launch_argv_starts_a_fake_ovms_that_serves_ready(tmp_path: Path) -> None:
    binary = tmp_path / "ovms"
    binary.write_text(_FAKE_OVMS.format(python=sys.executable), encoding="utf-8")
    binary.chmod(0o755)
    request = LaunchRequest(
        binary=binary,
        model_path=tmp_path,
        devices=(),
        n_gpu_layers=99,
        slots=1,
        ctx_per_slot=2048,
        target=RoleTarget.CPU,
    )
    argv = OVMS.launch_argv(OVMS.role_spec(WorkerRole.CHAT, None, None), request)
    launch = InstanceLaunch(role=WorkerRole.CHAT, argv=argv, env_overrides={}, model="m")
    launch = replace(launch, swap_extras=OVMS.swap_extras(launch))
    port = _free_port()
    config = json.loads(build_swap_config([launch], {launch.model_id: port}, engine_logs={}))
    entry = config["models"][launch.model_id]
    server = subprocess.Popen(shlex.split(entry["cmd"]))
    try:
        assert _ready(entry["proxy"] + entry["checkEndpoint"])
    finally:
        server.terminate()
        server.wait(timeout=10)
