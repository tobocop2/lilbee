"""Summarize a strace log of close and flock calls: who closed a descriptor that a failing call used.

Usage: fd_closer_summary.py TRACE [--max-failures N]

The trace comes from ``strace -f -tt -y -k --decode-pids=comm -e trace=close,flock,clone,clone3,fork,vfork``.
Every output line starts with a ``FDC-`` marker so one grep reads the result from the job log.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

LINE = re.compile(r"^(?P<tid>\d+)(?:<(?P<comm>[^>]*)>)?\s+(?P<ts>\d\d:\d\d:\d\d\.\d+)\s+(?P<body>.*)$")
CALL = re.compile(r"^(?P<name>[a-z_0-9]+)\((?P<args>.*)$")
RESUMED = re.compile(r"^<\.\.\. (?P<name>[a-z_0-9]+) resumed>(?P<rest>.*)$")
FD = re.compile(r"^(?P<fd>\d+)(?:<(?P<path>[^>]*)>)?")
RET = re.compile(r"\)\s+=\s+(?P<ret>-?\d+|\?)(?P<err>\s+[A-Z]+)?")
FRAME_LIMIT = 14
LOCKS = ("lock",)


@dataclass
class Event:
    index: int
    tid: int
    comm: str
    ts: str
    name: str
    args: str
    ret: str
    err: str
    fd: int | None = None
    path: str = ""
    stack: list[str] = field(default_factory=list)


def parse(trace_path: str) -> list[Event]:
    """Read the trace into completed events, joining unfinished and resumed halves."""
    events: list[Event] = []
    pending: dict[int, tuple[str, str]] = {}
    current: Event | None = None
    with open(trace_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = raw.rstrip("\n")
            stripped = text.lstrip()
            if stripped.startswith(">"):
                if current is not None and len(current.stack) < FRAME_LIMIT:
                    current.stack.append(stripped[1:].strip())
                continue
            match = LINE.match(text)
            if match is None:
                continue
            tid = int(match["tid"])
            body = match["body"]
            if body.startswith("+++") or body.startswith("---"):
                continue
            resumed = RESUMED.match(body)
            if resumed is not None:
                name = resumed["name"]
                args = pending.pop(tid, (name, ""))[1]
                tail = resumed["rest"]
                joined = f"{args}{tail}"
            else:
                call = CALL.match(body)
                if call is None:
                    continue
                name = call["name"]
                if body.endswith("<unfinished ...>"):
                    pending[tid] = (name, call["args"].replace("<unfinished ...>", ""))
                    continue
                joined = call["args"]
            ret = RET.search(joined)
            if ret is None:
                continue
            args = joined[: ret.start()]
            event = Event(
                index=len(events),
                tid=tid,
                comm=match["comm"] or "",
                ts=match["ts"],
                name=name,
                args=args,
                ret=ret["ret"],
                err=(ret["err"] or "").strip(),
            )
            fd_match = FD.match(args)
            if name in ("close", "flock") and fd_match is not None:
                event.fd = int(fd_match["fd"])
                event.path = fd_match["path"] or ""
            events.append(event)
            current = event
    return events


def thread_groups(events: list[Event]) -> dict[int, int]:
    """Map each thread id to its process id from clone results that share the process."""
    parent_of: dict[int, int] = {}
    for event in events:
        if event.name in ("clone", "clone3") and event.ret.isdigit() and "CLONE_THREAD" in event.args:
            parent_of[int(event.ret)] = event.tid

    def root(tid: int) -> int:
        seen = set()
        while tid in parent_of and tid not in seen:
            seen.add(tid)
            tid = parent_of[tid]
        return tid

    groups = {event.tid: root(event.tid) for event in events}
    groups.update({child: root(child) for child in parent_of})
    return groups


def show(event: Event, group: dict[int, int]) -> str:
    return (
        f"tgid={group.get(event.tid, event.tid)} tid={event.tid} comm={event.comm or '-'} "
        f"{event.ts} {event.name}({event.args}) = {event.ret} {event.err}".rstrip()
    )


def emit_stack(prefix: str, event: Event) -> None:
    for frame in event.stack:
        print(f"{prefix} {frame}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--max-failures", type=int, default=25)
    options = parser.parse_args()
    events = parse(options.trace)
    group = thread_groups(events)

    counts = Counter((event.name, event.err or "ok") for event in events)
    print(f"FDC-RESULT events={len(events)} stacks={sum(1 for e in events if e.stack)}")
    for (name, err), number in sorted(counts.items()):
        print(f"FDC-RESULT count {name} {err} {number}")

    failures = [e for e in events if e.err == "EBADF" and e.fd is not None]
    print(f"FDC-RESULT ebadf_failures={len(failures)}")
    by_fd: dict[tuple[int, int], list[Event]] = defaultdict(list)
    lockers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for event in events:
        if event.fd is None:
            continue
        key = (group.get(event.tid, event.tid), event.fd)
        by_fd[key].append(event)
        if event.name == "flock" and event.err == "":
            lockers[key].add(event.tid)

    for failure in failures[: options.max_failures]:
        key = (group.get(failure.tid, failure.tid), failure.fd)
        print(f"FDC-FAIL {show(failure, group)}")
        emit_stack("FDC-FAIL-STACK", failure)
        earlier = [e for e in by_fd[key] if e.index < failure.index]
        for prior in earlier[-6:]:
            print(f"FDC-PRIOR {show(prior, group)}")
            emit_stack("FDC-PRIOR-STACK", prior)

    foreign = 0
    for event in events:
        if event.name != "close" or event.fd is None or not any(w in event.path for w in LOCKS):
            continue
        key = (group.get(event.tid, event.tid), event.fd)
        if lockers[key] and event.tid not in lockers[key]:
            foreign += 1
            if foreign <= options.max_failures:
                print(f"FDC-FOREIGN {show(event, group)}")
                emit_stack("FDC-FOREIGN-STACK", event)
    print(f"FDC-RESULT lock_closes_by_other_thread={foreign}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
