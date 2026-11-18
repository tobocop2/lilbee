"""Check llama-swap's group semantics against the simulator the acceptance script relies on.

    python3 probe_llama_swap.py <llama-swap binary> <scratch dir>

Starts the real llama-swap with stub model servers (a tiny HTTP server that answers every path),
loads models in every order, and compares the running set that llama-swap reports with the
simulator's. Prints one line per (config, sequence) and ``PROBE rc=<0|1>`` last.

The config shapes are the three the rule needs:
  flat      the base: one group, swap true (loading a member evicts the others)
  classes   chat and rerank in one non-swapping group, vision in another, both exclusive
  singles   one exclusive group per role (the flat fallback written as classes)
Positive control: ``flat`` must evict chat when rerank loads. Negative control: ``classes``
must keep chat resident when rerank loads. A probe where both controls do not hold fails.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

STUB = """
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def _ok(self):
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    do_GET = do_POST = _ok
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""

ROLES = ("chat", "rerank", "vision")
REAL: dict[tuple[str, tuple[str, ...]], set[str]] = {}  # what llama-swap reported, per run
PORT = 18090
BASE = 18100


def simulate(cfg: dict, order: list[str]) -> set[str]:
    """Same text as simulate() in accept_swap_groups.py."""
    groups = cfg["groups"]
    where = {m: g for g, spec in groups.items() for m in spec["members"]}
    resident: set[str] = set()
    for model in order:
        spec = groups[where[model]]
        if spec.get("swap", True):
            resident -= set(spec["members"]) - {model}
        if spec.get("exclusive", True):
            resident -= {
                x
                for x in resident
                if where[x] != where[model] and not groups[where[x]].get("persistent", False)
            }
        resident.add(model)
    return resident


def config(work: Path, groups: dict) -> dict:
    stub = work / "stub.py"
    stub.write_text(STUB)
    models = {}
    for i, role in enumerate(ROLES):
        port = BASE + i
        models[role] = {
            "cmd": f"{sys.executable} {stub} {port}",
            "proxy": f"http://127.0.0.1:{port}",
            "ttl": 0,
        }
    return {"healthCheckTimeout": 30, "logLevel": "warn", "models": models, "groups": groups}


def call(path: str, method: str = "GET", body: dict | None = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def running() -> set[str]:
    rows = json.loads(call("/running"))["running"]
    return {row["model"] for row in rows if row.get("state", "ready") != "stopped"}


def run_variant(binary: str, work: Path, name: str, groups: dict) -> list[str]:
    cfg = config(work, groups)
    path = work / f"{name}.json"
    path.write_text(json.dumps(cfg))
    proc = subprocess.Popen(
        [binary, "-config", str(path), "-listen", f"127.0.0.1:{PORT}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    bad: list[str] = []
    try:
        for _ in range(100):
            try:
                call("/running")
                break
            except OSError:
                time.sleep(0.1)
        for size in (1, 2, 3):
            for order in itertools.permutations(ROLES, size):
                call("/unload", "GET")
                for model in order:
                    call("/v1/chat/completions", "POST", {"model": model, "messages": []})
                got = running()
                REAL[(name, order)] = got
                want = simulate(cfg, list(order))
                verdict = "ok" if got == want else "MISMATCH"
                print(f"{name} {'>'.join(order)}: llama-swap {sorted(got)} simulator {sorted(want)} {verdict}")
                if got != want:
                    bad.append(f"{name} {order}")
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    return bad


def main() -> int:
    binary, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    variants = {
        "flat": {
            "lilbee": {
                "swap": True,
                "exclusive": False,
                "persistent": True,
                "members": list(ROLES),
            }
        },
        "classes": {
            "query": {
                "swap": False,
                "exclusive": True,
                "persistent": False,
                "members": ["chat", "rerank"],
            },
            "ingest": {
                "swap": False,
                "exclusive": True,
                "persistent": False,
                "members": ["vision"],
            },
        },
        "singles": {
            role: {"swap": False, "exclusive": True, "persistent": False, "members": [role]}
            for role in ROLES
        },
    }
    bad: list[str] = []
    for name, groups in variants.items():
        bad += run_variant(binary, work, name, groups)
    flat = REAL[("flat", ("chat", "rerank"))]
    classes = REAL[("classes", ("chat", "rerank"))]
    vision = REAL[("classes", ("chat", "rerank", "vision"))]
    controls = flat == {"rerank"} and classes == {"chat", "rerank"} and vision == {"vision"}
    print(f"real llama-swap: flat chat>rerank {sorted(flat)}, classes chat>rerank {sorted(classes)}, "
          f"classes chat>rerank>vision {sorted(vision)}")
    rc = 0 if (not bad and controls) else 1
    print(f"PROBE rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
