"""Run a command; if its output stalls, py-spy-dump every python process, then kill.

The in-process faulthandler watchdog (tests/_hang_watchdog.py) names the wedged
test but cannot always dump its stack: a continuous GIL hold in native code
blocks even faulthandler's own dump. py-spy reads the target process memory from
the OUTSIDE, so it captures a native stack from a GIL-frozen worker where
faulthandler cannot; on Windows it needs no administrator for a same-user
process. This wrapper watches the child's output and, when it goes silent past a
threshold, py-spy-dumps the whole python process tree and then kills it so the
job fails fast with the frame instead of hanging to timeout-minutes.

Opt-in: no-op passthrough unless ``LILBEE_HANG_WATCHDOG_STALL_S`` (seconds) is set.

Usage: python scripts/qa/hang_watchdog.py -- <command> [args...]
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_STALL_S = float(os.environ.get("LILBEE_HANG_WATCHDOG_STALL_S", "0") or 0)
_DUMP_DIR = Path(os.environ.get("LILBEE_HANG_WATCHDOG_DIR", "."))
_PYSPY = os.environ.get("LILBEE_HANG_WATCHDOG_PYSPY", "py-spy")


def _command_from_argv() -> list[str]:
    argv = sys.argv[1:]
    return argv[1:] if argv and argv[0] == "--" else argv


def _python_procs(root_pid: int) -> list:
    """Every live python process in root_pid's tree (the pytest xdist workers)."""
    import psutil

    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return []
    found = []
    for proc in (root, *root.children(recursive=True)):
        try:
            if "python" in proc.name().lower():
                found.append(proc)
        except psutil.Error:
            continue
    return found


def _dump(pid: int) -> str:
    """py-spy dump of one pid; try native, fall back to plain, never raise."""
    for extra in (["--native"], []):
        try:
            out = subprocess.run(
                [_PYSPY, "dump", "--nonblocking", *extra, "--pid", str(pid)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"py-spy dump pid={pid} failed to run: {exc}\n"
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
        last = out.stderr or out.stdout
    return f"py-spy dump pid={pid} returned nothing (rc={out.returncode}): {last}\n"


def _dump_all(procs: list) -> None:
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        f"===== hang watchdog: output stalled >{_STALL_S:.0f}s; "
        f"py-spy dump of {len(procs)} python process(es) ====="
    ]
    for proc in procs:
        report.append(f"\n----- pid {proc.pid} -----\n{_dump(proc.pid)}")
    text = "\n".join(report)
    (_DUMP_DIR / "pyspy-hang.txt").write_text(text)
    # Also to our own stderr so it lands in the step log even before the surface step.
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def main() -> int:
    cmd = _command_from_argv()
    if not cmd:
        sys.stderr.write("hang_watchdog: no command given\n")
        return 2
    if _STALL_S <= 0:
        return subprocess.call(cmd)  # disabled: transparent passthrough

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    last_output = time.monotonic()
    lock = threading.Lock()

    def _pump() -> None:
        nonlocal last_output
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            with lock:
                last_output = time.monotonic()

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    fired = False
    while proc.poll() is None:
        time.sleep(2.0)
        with lock:
            idle = time.monotonic() - last_output
        if idle > _STALL_S and not fired:
            fired = True
            procs = _python_procs(proc.pid)
            _dump_all(procs)
            # Kill the whole tree so the job fails fast instead of hanging.
            # psutil.kill() is TerminateProcess on Windows, SIGKILL elsewhere.
            for worker in procs:
                with contextlib.suppress(Exception):
                    worker.kill()
            proc.terminate()
            break

    pump.join(timeout=5.0)
    rc = proc.wait()
    if fired:
        sys.stderr.write(
            "hang_watchdog: killed the run after an output stall; see pyspy-hang.txt\n"
        )
        return rc or 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
