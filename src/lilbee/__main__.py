"""Allow running as `python -m lilbee`."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_FLATPAK_INFO = Path("/.flatpak-info")  # present in every Flatpak sandbox


def _isolate_vendored_openssl() -> None:
    """Default ``OPENSSL_CONF`` to the empty config in frozen Flatpak runs.

    Flatpak's freedesktop runtime ships an openssl.cnf whose engine section
    dlopens engine modules built against the runtime's own OpenSSL. Bundled
    wheels that vendor a static OpenSSL (pyarrow's libarrow) honor that
    config during import-time init, load the ABI-incompatible engine, and
    segfault before any lilbee code runs. An empty config keeps every
    vendored OpenSSL self-contained; certificate paths are unaffected.
    Scoped to Flatpak sandboxes so every other install keeps reading the
    host config, and an explicitly set ``OPENSSL_CONF`` always wins.
    """
    if not hasattr(sys, "frozen"):
        return
    if sys.platform != "linux" or not _FLATPAK_INFO.exists():
        return
    os.environ.setdefault("OPENSSL_CONF", os.devnull)


def _multiprocessing_child_code(argv: list[str]) -> str | None:
    """Return the ``-c`` payload if *argv* is a multiprocessing child reinvocation.

    Python's ``multiprocessing.resource_tracker`` respawns ``sys.executable``
    with interpreter flags followed by ``-c "<code>"`` (e.g.
    ``-B -s -E -c "from multiprocessing.resource_tracker import main;main(7)"``).
    In a frozen build (Nuitka onefile) ``sys.executable`` is this exe, so those
    interpreter flags leak into the typer parser and raise ``No such option: -B``
    (or, once typer has eaten the flags, ``No such command '<N>'`` for the
    leftover fd number). The stdlib ``freeze_support()`` hook only covers spawn
    children (``--multiprocessing-fork``), not resource_tracker.

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
    exec(  # noqa: S102  payload is emitted by Python's own stdlib into sys.executable
        compile(code, "<frozen-mp-child>", "exec"),
        {"__name__": "__main__", "__builtins__": __builtins__},
    )
    return True


_DASH_M_MIN_ARGV = 3  # [bin, "-m", "lilbee.<module>"]


def _dispatch_module_invocation() -> bool:
    """Run a `python -m lilbee.<module>` reinvocation and return True when handled.

    Internal subprocesses spawned via ``[sys.executable, "-m", "lilbee.X", ...]``
    (e.g. ``splash._splash_runner``) hit typer's `--model -m` short-form parser
    in a frozen build. Detect the pattern, route to runpy, never reach typer.
    Restricted to the ``lilbee.*`` namespace so a stray ``lilbee -m foo`` typed
    by a user can't reach exec.
    """
    if not getattr(sys, "frozen", False):
        return False
    if len(sys.argv) < _DASH_M_MIN_ARGV or sys.argv[1] != "-m":
        return False
    module_name = sys.argv[2]
    if not module_name.startswith("lilbee."):
        return False
    sys.argv = [module_name, *sys.argv[3:]]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    return True


if __name__ == "__main__":  # pragma: no cover - process entry glue; logic is unit-tested above
    # Must run before anything that can initialize OpenSSL (pyarrow import,
    # ssl), including the multiprocessing child payloads dispatched below.
    _isolate_vendored_openssl()

    # Make the frozen exe a valid subprocess target for multiprocessing's
    # sys.executable reinvocations, BEFORE any import that could pull typer.
    if _dispatch_module_invocation():
        sys.exit(0)
    if _dispatch_frozen_child():
        sys.exit(0)

    import multiprocessing

    multiprocessing.freeze_support()

    from lilbee.runtime.launcher import main

    main()
