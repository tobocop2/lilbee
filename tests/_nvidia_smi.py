"""A stand-in for the NVIDIA adapter's tool runner, built from a fake that returns printed text."""

from __future__ import annotations

from collections.abc import Callable


def prints(fake: Callable[..., str]) -> Callable[..., tuple[str, int | None]]:
    """Exit 0 with the fake's text; an empty text is a tool that cannot start (no exit code)."""

    def run(*args: object, **kwargs: object) -> tuple[str, int | None]:
        out = fake(*args, **kwargs)
        return (out, 0) if out else ("", None)

    return run
