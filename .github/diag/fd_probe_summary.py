"""Join the flock-held close probe outputs: who closed a held lock descriptor, and who owned it.

Usage: fd_probe_summary.py OUT_DIR [--max-lines N] [--expect-canary N]

OUT_DIR holds bpftrace.txt (fd_closer.bt), fdp-owner-PID.txt and fdp-dump-PID.txt (tests/_fd_probe.py).
Every output line starts with ``FDC-`` so one grep reads the result from the job log.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

FIELDS = re.compile(r"(\w+)=(\S*)")
CANARY = re.compile(r"fd-canary-(?P<pid>\d+)\.lock")
DUMP_HEADER = re.compile(r"^(Current thread|Thread) 0x[0-9a-f]+")
NATIVE_LIMIT = 8


@dataclass
class Close:
    kind: str
    pid: int
    tid: int
    comm: str
    call: str
    fd: int
    owner_tid: int
    armed: bool
    ns: int
    native: list[str] = field(default_factory=list)
    python: list[str] = field(default_factory=list)


@dataclass
class Owner:
    pid: int
    tid: int
    fd: int
    ns: int
    path: str
    stack: list[str] = field(default_factory=list)


def _fields(line: str) -> dict[str, str]:
    return dict(FIELDS.findall(line))


def read_trace(path: Path) -> tuple[list[Close], list[dict[str, str]], set[int], Counter[str]]:
    """Held closes and ranges, EBADF victims, armed pids and a count of each record kind."""
    closes: list[Close] = []
    ebadf: list[dict[str, str]] = []
    armed: set[int] = set()
    kinds: Counter[str] = Counter()
    native: list[str] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if native is not None:
            if line.startswith("FDB-NATIVE-END"):
                native = None
            elif line and len(native) < NATIVE_LIMIT:
                native.append(line)
            continue
        kind = line.split(" ", 1)[0]
        kinds[kind] += 1
        values = _fields(line)
        if kind == "FDB-ARMED":
            armed.add(int(values["pid"]))
        elif kind == "FDB-EBADF":
            ebadf.append(values)
        elif kind in ("FDB-HELDCLOSE", "FDB-RANGE"):
            closes.append(
                Close(
                    kind=kind,
                    pid=int(values["pid"]),
                    tid=int(values["tid"]),
                    comm=values.get("comm", ""),
                    call=values.get("call", "close_range"),
                    fd=int(values.get("fd", values.get("first", "-1"))),
                    owner_tid=int(values.get("owner_tid", "0")),
                    armed=values.get("armed") == "1",
                    ns=int(values["ns"]),
                )
            )
        elif kind.startswith("FDB-NATIVE-BEGIN") and closes:
            native = closes[-1].native
    return closes, ebadf, armed, kinds


def read_owners(out: Path) -> dict[tuple[int, int], list[Owner]]:
    """Owner records keyed by (pid, fd), in the order each process wrote them."""
    owners: dict[tuple[int, int], list[Owner]] = defaultdict(list)
    for path in sorted(out.glob("fdp-owner-*.txt")):
        current: Owner | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("FDP-OWNER-STACK ") and current is not None:
                current.stack.append(line.removeprefix("FDP-OWNER-STACK ").strip())
            elif line.startswith("FDP-OWNER "):
                values = _fields(line)
                current = Owner(
                    pid=int(values["pid"]),
                    tid=int(values["tid"]),
                    fd=int(values["fd"]),
                    ns=int(values["ns"]),
                    path=values.get("path", "?"),
                )
                owners[(current.pid, current.fd)].append(current)
    return owners


def read_dumps(out: Path) -> dict[int, list[list[str]]]:
    """For each pid, the frames of the signalled thread in each faulthandler dump, in order."""
    dumps: dict[int, list[list[str]]] = {}
    for path in sorted(out.glob("fdp-dump-*.txt")):
        pid = int(path.stem.rsplit("-", 1)[1])
        blocks: list[list[str]] = []
        current: list[str] | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            header = DUMP_HEADER.match(line)
            if header is not None:
                current = [] if header.group(1) == "Current thread" else None
                if current is not None:
                    blocks.append(current)
            elif current is not None and line.strip().startswith("File "):
                current.append(line.strip())
        dumps[pid] = blocks
    return dumps


def _innermost_file(frame: str) -> str:
    match = re.search(r'File "([^"]+)"', frame)
    return match.group(1) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--max-lines", type=int, default=25)
    parser.add_argument("--expect-canary", type=int, default=0)
    options = parser.parse_args()
    out = Path(options.out)
    closes, ebadf, armed, kinds = read_trace(out / "bpftrace.txt")
    owners = read_owners(out)
    dumps = read_dumps(out)

    print(f"FDC-RESULT armed_processes={len(armed)} owner_files={len(list(out.glob('fdp-owner-*.txt')))}")
    print(
        f"FDC-RESULT locks={kinds['FDB-LOCK']} held_closes={kinds['FDB-HELDCLOSE']} "
        f"ranges={kinds['FDB-RANGE']} owner_records={sum(len(v) for v in owners.values())}"
    )

    signalled: dict[int, list[Close]] = defaultdict(list)
    for close in sorted(closes, key=lambda c: c.ns):
        if close.armed:
            signalled[close.pid].append(close)
    mismatched = 0
    for pid, events in signalled.items():
        blocks = dumps.get(pid, [])
        if len(blocks) != len(events):
            mismatched += 1
            print(f"FDC-DUMP-MISMATCH pid={pid} signals={len(events)} dumps={len(blocks)}")
        for event, frames in zip(events, blocks, strict=False):
            event.python = frames
    print(f"FDC-RESULT dump_mismatched_processes={mismatched}")

    foreign = own = unprobed = 0
    canary_pids: set[int] = set()
    for close in sorted(closes, key=lambda c: c.ns):
        if close.pid not in armed:
            unprobed += 1
            continue
        if close.kind == "FDB-RANGE":
            print(f"FDC-RANGE pid={close.pid} tid={close.tid} first={close.fd} armed={int(close.armed)}")
            for frame in close.python:
                print(f"FDC-RANGE-STACK {frame}")
            continue
        earlier = [o for o in owners.get((close.pid, close.fd), []) if o.ns <= close.ns]
        owner = earlier[-1] if earlier else None
        closer_file = _innermost_file(close.python[0]) if close.python else ""
        owner_file = _innermost_file(owner.stack[-1]) if owner and owner.stack else ""
        is_foreign = close.tid != close.owner_tid or closer_file != owner_file
        if not is_foreign:
            own += 1
            continue
        foreign += 1
        if foreign <= options.max_lines:
            print(
                f"FDC-FOREIGN pid={close.pid} tid={close.tid} comm={close.comm} call={close.call} fd={close.fd} "
                f"owner_tid={close.owner_tid} armed={int(close.armed)} ns={close.ns}"
            )
            for frame in close.python:
                print(f"FDC-FOREIGN-STACK {frame}")
            for frame in close.native:
                print(f"FDC-FOREIGN-NATIVE {frame}")
            if owner is None:
                print("FDC-FOREIGN-OWNER none recorded (not a Python flock in a probed process)")
            else:
                print(f"FDC-FOREIGN-OWNER pid={owner.pid} tid={owner.tid} fd={owner.fd} path={owner.path}")
                for frame in owner.stack:
                    print(f"FDC-FOREIGN-OWNER-STACK {frame}")
        if owner is not None and close.python and owner.stack:
            match = CANARY.search(owner.path)
            if match is not None and int(match["pid"]) == close.pid:
                canary_pids.add(close.pid)

    victims = [v for v in ebadf if int(v["pid"]) in armed]
    for victim in victims[: options.max_lines]:
        print(f"FDC-FAIL pid={victim['pid']} tid={victim['tid']} call={victim['call']} fd={victim['fd']}")
    print(f"FDC-RESULT held_closes_own={own} held_closes_foreign={foreign} held_closes_unprobed={unprobed}")
    print(f"FDC-RESULT ebadf_failures={len(victims)} ebadf_all_processes={len(ebadf)}")
    print(f"FDC-RESULT canary_processes={len(canary_pids)}")
    if options.expect_canary:
        if len(canary_pids) < options.expect_canary or mismatched:
            print(f"FDC-CANARY MISSING expected={options.expect_canary} seen={len(canary_pids)} mismatched={mismatched}")
            return 3
        print(f"FDC-CANARY OK processes={sorted(canary_pids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
