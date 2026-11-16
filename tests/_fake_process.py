"""A stand-in for psutil.Process that keeps the parts of its surface the orphan sweep reads.

The real surface: ``name()``, ``cmdline()``, ``parent()`` for one hop, ``parents()`` for the
whole chain at once (it raises when any hop raises), and ``info``, which exists only when a
process table call asked for attributes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import psutil

Hop = str | Exception


class FakeProcess:
    """A process with a name, a command line and a chain of ancestors, nearest first.

    An ancestor is a name, or an exception that reading that hop raises.
    """

    def __init__(self, name: str, cmdline: list[str], ancestors: list[Hop] | None = None) -> None:
        self._name = name
        self._cmdline = cmdline
        self._ancestors = list(ancestors or [])

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return self._cmdline

    def parent(self) -> FakeProcess | None:
        if not self._ancestors:
            return None
        head, *rest = self._ancestors
        if isinstance(head, Exception):
            raise head
        return FakeProcess(head, [], rest)

    def parents(self) -> list[FakeProcess]:
        chain: list[FakeProcess] = []
        parent = self.parent()
        while parent is not None:
            chain.append(parent)
            parent = parent.parent()
        return chain


class UnreadableCmdline(FakeProcess):
    def cmdline(self) -> list[str]:
        raise psutil.AccessDenied(1)


class UnreadableName(FakeProcess):
    def name(self) -> str:
        raise psutil.NoSuchProcess(1)


def process_table(procs: list[FakeProcess]) -> Callable[..., Iterator[FakeProcess]]:
    """A ``psutil.process_iter`` that sets ``info`` only for the attributes it is asked for.

    Like psutil, a name that cannot be read comes back as None.
    """

    def process_iter(attrs: list[str] | None = None) -> Iterator[FakeProcess]:
        for proc in procs:
            proc.__dict__.pop("info", None)
            if attrs is not None and "name" in attrs:
                try:
                    name: str | None = proc.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    name = None
                proc.__dict__["info"] = {"name": name}
        return iter(procs)

    return process_iter
