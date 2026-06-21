"""Allow running as `python -m lilbee`."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

_FLATPAK_INFO = Path("/.flatpak-info")  # present in every Flatpak sandbox


def _isolate_vendored_openssl() -> None:
    """Default ``OPENSSL_CONF`` to the empty config inside Flatpak sandboxes.

    Flatpak's freedesktop runtime ships an openssl.cnf whose engine section
    dlopens engine modules built against the runtime's own OpenSSL. Bundled
    wheels that vendor a static OpenSSL (pyarrow's libarrow) honor that
    config during import-time init, load the ABI-incompatible engine, and
    segfault before any lilbee code runs. An empty config keeps every
    vendored OpenSSL self-contained; certificate paths are unaffected.
    Scoped to Flatpak sandboxes so every other install keeps reading the
    host config, and an explicitly set ``OPENSSL_CONF`` always wins.
    Deliberately not gated on a frozen-binary check: Nuitka does not set
    ``sys.frozen``, and a pip install run inside a sandbox crashes the same
    way.
    """
    if sys.platform != "linux" or not _FLATPAK_INFO.exists():
        return
    os.environ.setdefault("OPENSSL_CONF", os.devnull)


def _is_frozen() -> bool:
    """True when running from a frozen build.

    A deliberate local copy of :func:`lilbee._frozen.is_frozen`: the dispatch
    path below must import nothing from the ``lilbee`` package (the splash
    subprocess this intercepts is stdlib-only and must stay light), so it
    cannot import the shared helper. PyInstaller/cx_Freeze set ``sys.frozen``;
    Nuitka never does, but injects a ``__compiled__`` global into every module
    it compiles. Both must be checked so the dispatchers fire in the shipped
    Nuitka onefile binary, not just under PyInstaller.
    """
    return bool(sys.__dict__.get("frozen")) or "__compiled__" in globals()


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
    if not _is_frozen():
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
    (e.g. ``splash.start`` launching ``lilbee.runtime._splash_runner``) hit
    typer's `--model -m` short-form parser in a frozen build. Detect the
    pattern, run the target module's ``main()`` directly, never reach typer.

    Imports the module and calls ``main()`` rather than ``runpy.run_module``:
    Nuitka's module loader does not implement ``get_code``, so ``runpy`` raises
    ``AttributeError: ... has no attribute 'get_code'`` inside the onefile
    binary. Every ``lilbee.*`` module reinvoked this way exposes a ``main()``
    entry point. Restricted to the ``lilbee.*`` namespace so a stray
    ``lilbee -m foo`` typed by a user can't reach exec.
    """
    if not _is_frozen():
        return False
    if len(sys.argv) < _DASH_M_MIN_ARGV or sys.argv[1] != "-m":
        return False
    module_name = sys.argv[2]
    if not module_name.startswith("lilbee."):
        return False
    sys.argv = [module_name, *sys.argv[3:]]
    module = importlib.import_module(module_name)
    module.main()
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
