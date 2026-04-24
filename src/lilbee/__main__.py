"""Allow running as `python -m lilbee`."""

from __future__ import annotations

import sys


def _multiprocessing_child_code(argv: list[str]) -> str | None:
    """Return the ``-c`` payload if *argv* is a multiprocessing child reinvocation.

    Python's ``multiprocessing.resource_tracker`` respawns ``sys.executable``
    with interpreter flags followed by ``-c "<code>"`` (e.g.
    ``-B -s -E -c "from multiprocessing.resource_tracker import main;main(7)"``).
    Under a PyInstaller ``--onefile`` bundle ``sys.executable`` is this exe,
    so those interpreter flags leak into the typer parser and raise
    ``No such option: -B``. The stdlib ``freeze_support()`` hook only covers
    spawn children (``--multiprocessing-fork``), not resource_tracker.

    Require the payload to mention ``multiprocessing`` or ``spawn_main`` so a
    stray ``lilbee -c …`` typed by a human never reaches ``exec``.
    """
    try:
        idx = argv.index("-c")
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        return None
    code = argv[idx + 1]
    if "multiprocessing" not in code and "spawn_main" not in code:
        return None
    return code


def _dispatch_frozen_child() -> bool:
    """Exec a multiprocessing child payload and return True when handled."""
    if not getattr(sys, "frozen", False):
        return False
    code = _multiprocessing_child_code(sys.argv)
    if code is None:
        return False
    exec(  # noqa: S102 — payload is emitted by Python's own stdlib into sys.executable
        compile(code, "<pyi-mp-child>", "exec"),
        {"__name__": "__main__", "__builtins__": __builtins__},
    )
    return True


if __name__ == "__main__":
    # Make the frozen exe a valid subprocess target for multiprocessing's
    # sys.executable reinvocations, BEFORE any import that could pull typer.
    if _dispatch_frozen_child():
        sys.exit(0)

    import multiprocessing

    multiprocessing.freeze_support()

    from lilbee.launcher import main

    main()
