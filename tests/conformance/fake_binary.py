"""A stand-in engine binary for the managed rows; FAKE_ENGINE_CFG holds its behaviour as JSON."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        time.sleep(30)


def _serve(cfg: dict[str, object], argv: list[str]) -> None:
    port = int(argv[argv.index("--port") + 1])
    if cfg["term"] == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    HTTPServer(("127.0.0.1", port), _Handler).serve_forever()


def main(argv: list[str]) -> int:
    cfg = json.loads(os.environ["FAKE_ENGINE_CFG"])
    if "--help" in argv:
        print("usage: engine " + " ".join(cfg["flags"]))
        return int(cfg["help_rc"])
    if "--list-devices" in argv:
        if cfg["devices"] == "llama":
            print("Available devices:\n  CUDA0: NVIDIA Test GPU (8192 MiB, 8000 MiB free)")
            return 0
        sys.stderr.write("error: unexpected argument '--list-devices' found\n")
        return 2
    if "--estimate" in argv:
        if cfg["estimate"]:
            print("123456789")
            return 0
        return 1
    if "--port" in argv:
        _serve(cfg, argv)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
