#!/usr/bin/env python3
"""CLI surface QA: every subcommand registers + key read paths run clean.

Catches import/registration breakage (common after a multi-PR merge) and
tracebacks in read-only command paths. Run from the repo root:
  LILBEE_QA_CLI=lilbee uv run --no-sync python scripts/qa/cli_surface_qa.py
"""

from __future__ import annotations

import os
import subprocess
import sys

CLI = os.environ.get("LILBEE_QA_CLI", "lilbee")
RESULTS: list[tuple[str, bool, str]] = []


def rec(cid: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((cid, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}" + (f"  -- {detail}" if detail else ""), flush=True)


def run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run([CLI, *args], capture_output=True, text=True, timeout=timeout)


def _traceback(cp: subprocess.CompletedProcess[str]) -> bool:
    return "Traceback (most recent call last)" in (cp.stderr + cp.stdout)


# Every top-level command (from the `lilbee --help` Commands panel). --help must
# register cleanly -- a failure here means a command's module fails to import.
COMMANDS = [
    "add",
    "agent-config",
    "ask",
    "chat",
    "chunks",
    "export",
    "import",
    "index",
    "init",
    "launch",
    "login",
    "mcp",
    "memory",
    "model",
    "placement",
    "rebuild",
    "remove",
    "reset",
    "search",
    "self-check",
    "self-check-extras",
    "serve",
    "setup",
    "status",
    "sync",
    "token",
    "topics",
    "use-embedder",
    "version",
    "wiki",
]

# Read-only invocations that must exit 0 (GPU listing is `placement show`, not a
# command). A traceback with rc==0 is a handled warning, reported as observation.
READ_INVOCATIONS = [
    ("status", ["status"]),
    ("model-list", ["model", "list"]),
    ("placement-show", ["placement", "show"]),
    ("placement-preview", ["placement", "preview"]),
    ("memory-list", ["memory", "list"]),
    ("search", ["search", "lilbee"]),
    ("version", ["version"]),
]


def _failure_detail(cp: subprocess.CompletedProcess[str]) -> str:
    """One-line failure summary from a completed process."""
    out = (cp.stderr or cp.stdout).strip()
    tail = out.splitlines()[-1][:100] if out else ""
    return f"rc={cp.returncode} {tail}"


def main() -> int:  # noqa: C901 -- linear QA checklist, one branch per surface probe
    print("=== CLI --help registration ===", flush=True)
    for cmd in COMMANDS:
        try:
            cp = run([cmd, "--help"], timeout=90)
            ok = cp.returncode == 0 and not _traceback(cp)
            rec(
                f"help:{cmd}",
                ok,
                "" if ok else _failure_detail(cp),
            )
        except subprocess.TimeoutExpired:
            rec(f"help:{cmd}", False, "timed out")
        except Exception as exc:
            rec(f"help:{cmd}", False, f"{type(exc).__name__}: {exc}")

    print("\n=== read-only invocations ===", flush=True)
    # A crash = nonzero exit. A traceback printed with rc==0 is an exc_info on a
    # handled WARNING (e.g. engine warm-up lost the health race), not a failure;
    # it is surfaced as an observation, not a FAIL.
    for cid, args in READ_INVOCATIONS:
        try:
            cp = run(args, timeout=150)
            ok = cp.returncode == 0
            note = ""
            if ok and _traceback(cp):
                note = "(rc=0 but a handled traceback was logged -- check it is intentional)"
            elif not ok:
                tail = (cp.stderr or cp.stdout).strip().splitlines()
                note = f"rc={cp.returncode}: {tail[-1][:120] if tail else ''}"
            rec(f"run:{cid}", ok, note)
        except subprocess.TimeoutExpired:
            rec(f"run:{cid}", False, "timed out")
        except Exception as exc:
            rec(f"run:{cid}", False, f"{type(exc).__name__}: {exc}")

    failed = [c for c, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed, {len(failed)} failed", flush=True)
    if failed:
        print("FAILURES:", flush=True)
        for cid, ok, detail in RESULTS:
            if not ok:
                print(f"  [FAIL] {cid}: {detail}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
