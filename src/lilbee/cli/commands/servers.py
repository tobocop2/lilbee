"""Serve (HTTP API) and mcp (stdio) server-boot commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer

from lilbee.app.services import (
    answer_provider_waiters,
    exit_after_signal,
    exit_code_of,
    exit_if_opens_remain,
    exit_status_after_stop,
    report_library_upgrade,
    reset_services_on_exit,
    set_server_exit_hook,
    stop_for_parent_death,
    stop_on_interrupt,
    wait_for_hard_exit_teardown,
)
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.commands.serve_logging import (
    arm_exit_deadline,
    install_health_access_filter,
    setup_server_logging,
)
from lilbee.cli.upgrade_line import UpgradeLine
from lilbee.core.config import cfg
from lilbee.runtime.asyncio_loop import run_main
from lilbee.runtime.console import PlainConsole
from lilbee.runtime.lock import (
    SERVER_LOCK_TIMEOUT,
    LockingUnsupportedError,
    acquire_scope_lock,
    acquire_server_lock,
    read_scope_owner,
)

if TYPE_CHECKING:
    import uvicorn
    from filelock import FileLock

    from lilbee.runtime.lock import ScopeHold


SCOPE_ENV = "LILBEE_EXCLUSIVE_SCOPE"
"""A directory that at most one server may serve at a time (a plugin's shared root)."""

LOCK_REFUSAL_EXIT_CODE = 3
"""Exit code for a start refused because another live server holds a lock.

Distinct from a generic failure so a supervisor can tell "another server owns
this" apart from a crash without parsing output.
"""

_STARTUP_FAILED_EXIT_STATUS = 1
"""Exit code for a server whose startup failed, such as a failed services build."""

_EXIT_DEADLINE_S = 60.0
"""Seconds from the start of shutdown until the process ends, whatever holds it."""

_GRACEFUL_SHUTDOWN_S = 10
"""Seconds open requests get before uvicorn cancels them and runs the lifespan teardown."""

_INTERRUPTED_EXIT_STATUS = 128 + signal.SIGINT
"""The status a shell reports for a process that Ctrl-C ended."""

_HARD_EXIT_GRACE_S = 1
"""The same wait on SIGTERM or SIGHUP: long enough for a woken write to send its answer."""


def port_file() -> Path:
    """Path to the running server's port file under ``cfg.data_dir``."""
    return cfg.data_dir / "server.port"


def _log_loop_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
    exc = context.get("exception")
    # isinstance: asyncio's context dict is untyped; "exception" may be absent
    if isinstance(exc, BaseException):
        logging.getLogger(__name__).error("asyncio task error", exc_info=exc)
    else:
        logging.getLogger(__name__).error("asyncio task error: %s", context.get("message"))


class _ExitRequest:
    """The exit a serving loop was asked for; a nonzero status cuts the wait for responses."""

    def __init__(self, server: uvicorn.Server) -> None:
        self.status = 0
        self._server = server

    def __call__(self, status: int) -> None:
        self.status = status
        if status:
            self._server.config.timeout_graceful_shutdown = _HARD_EXIT_GRACE_S
        self._server.should_exit = True


async def _run_server(server: uvicorn.Server, config: uvicorn.Config, host: str) -> int:
    """Start uvicorn, write port file, clean up on shutdown, and return the exit status."""
    import atexit

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_log_loop_exception)

    port_path = port_file()

    def _cleanup_port_file() -> None:
        port_path.unlink(missing_ok=True)

    exit_request = _ExitRequest(server)

    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)

    # `server.servers` is set inside `startup()`. The finally below must skip
    # `shutdown()` when startup never ran: uvicorn dereferences `self.servers`
    # there and the resulting AttributeError would mask the original failure.
    started = False
    try:
        set_server_exit_hook(exit_request)
        try:
            await server.startup()
        except SystemExit:
            # uvicorn ends a failed start (lifespan or bind) with the status a lock refusal uses.
            # A stop signal during the start has set its own nonzero status.
            return exit_request.status or _STARTUP_FAILED_EXIT_STATUS
        started = True

        if server.servers:
            sock = server.servers[0].sockets[0]
            actual_port = sock.getsockname()[1]
            port_path.parent.mkdir(parents=True, exist_ok=True)
            port_path.write_text(str(actual_port), encoding="utf-8")
            atexit.register(_cleanup_port_file)
            console.print(f"Listening on http://{host}:{actual_port}")
        await server.main_loop()
    finally:
        arm_exit_deadline(_EXIT_DEADLINE_S)
        answer_provider_waiters()
        set_server_exit_hook(None)
        port_path.unlink(missing_ok=True)
        if started:
            # Suppress AttributeError from a partial uvicorn bring-up so any
            # original exception from main_loop reaches the caller intact.
            with contextlib.suppress(AttributeError):
                await server.shutdown()
    return exit_request.status


def _report_upgrade_as_log_lines() -> None:
    """Show a library upgrade of this server as plain stderr lines, beside its log."""
    report_library_upgrade(UpgradeLine(PlainConsole(stderr=True), live=False))


def _refuse_to_start(message: str) -> NoReturn:
    """Report why the server will not start, to the log and the terminal, then exit."""
    logging.getLogger(__name__).error(message)
    console.print(message, soft_wrap=True)
    raise typer.Exit(LOCK_REFUSAL_EXIT_CODE)


def _take_the_scope() -> ScopeHold | None:
    """Hold the exclusive scope named by the environment, or refuse to start; None when unset."""
    scope_env = os.environ.get(SCOPE_ENV)
    if not scope_env:
        return None
    scope_dir = Path(scope_env)
    try:
        scope_hold = acquire_scope_lock(scope_dir, cfg.data_dir, timeout=SERVER_LOCK_TIMEOUT)
    except LockingUnsupportedError as exc:
        _refuse_to_start(str(exc))
    if scope_hold is None:
        owner = read_scope_owner(scope_dir)
        serving = f" It is serving {owner.data_dir}." if owner else ""
        message = (
            f"Another lilbee server is already running for this installation.{serving}"
            " Stop it or wait for it to exit, then retry."
        )
        _refuse_to_start(message)
    return scope_hold


def _watch_the_parent() -> None:
    """Stop the server once the process named by ``LILBEE_PARENT_PID`` is gone.

    It watches from the start, so a parent that dies while the server waits on a
    lock or builds its app stops it too.
    """
    from lilbee.parent_monitor import parse_parent_pid, watch_parent_thread

    parent_pid = parse_parent_pid()
    if parent_pid is not None:
        watch_parent_thread(parent_pid, stop_for_parent_death)


def serve(
    host: str = typer.Option(None, "--host", "-H", help="Bind address (default: 127.0.0.1)"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: 0/random)"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Start the HTTP API server."""
    _watch_the_parent()
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _report_upgrade_as_log_lines()
    if host is not None:
        cfg.server_host = host
    if port is not None:
        cfg.server_port = port

    setup_server_logging()

    # One managed server per scope: the plugin passes its shared root here so a
    # second vault's server cannot start while another vault's is serving it.
    scope_hold = _take_the_scope()

    # One server per data dir: a second instance would overwrite server.port
    # and spawn a second engine fleet against the same models and vector store.
    message = (
        "Another lilbee server is already running for this data directory. "
        "Stop it or wait for it to exit, then retry."
    )
    try:
        server_lock = acquire_server_lock(cfg.data_dir, timeout=SERVER_LOCK_TIMEOUT)
    except LockingUnsupportedError as exc:
        server_lock, message = None, str(exc)
    if server_lock is None:
        if scope_hold is not None:
            scope_hold.release()
        _refuse_to_start(message)

    import uvicorn

    from lilbee.server import create_app

    logging.getLogger("asyncio").setLevel(logging.ERROR)

    status = 1
    try:
        app = create_app()
        install_health_access_filter()
        config = uvicorn.Config(
            app,
            host=cfg.server_host,
            port=cfg.server_port,
            timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_S,
        )
        server = uvicorn.Server(config)
        # Keeps asyncio from installing its own handler, which only cancels the main task
        # through a loop callback. An inherited ignore of SIGINT stays ignored.
        stop_on_interrupt()
        status = run_main(_run_server(server, config, cfg.server_host))
    except Exception:
        # The exit in the finally can end the process before the traceback prints.
        logging.getLogger(__name__).error("The server run failed.", exc_info=True)
        raise
    except KeyboardInterrupt:
        # The exit in the finally can end the process before the interrupt reaches the
        # interpreter, so the status is the shell's own for Ctrl-C.
        status = _INTERRUPTED_EXIT_STATUS
        raise
    finally:
        _stop_server(status, server_lock, scope_hold)
    raise typer.Exit(exit_status_after_stop(status))


def _stop_server(status: int, server_lock: FileLock, scope_hold: ScopeHold | None) -> None:
    """Free the locks after the teardown, then end the process if a library open still runs."""
    # A signal-driven shutdown stops the fleet on its own thread; hold the
    # locks until it finishes so a successor cannot start while this
    # server's models still occupy memory.
    wait_for_hard_exit_teardown()
    server_lock.release()
    if scope_hold is not None:
        scope_hold.release()
    exit_if_opens_remain(status)


def mcp_cmd(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Start the MCP server (stdio transport) for agent integration."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    _report_upgrade_as_log_lines()
    from lilbee.mcp_server import main

    # Set before the loop starts, so Ctrl-C has this handler while no loop runs. The
    # serving loop takes the routed signals as callbacks and gives these handlers back
    # when it ends, so a late signal still reaches the stop.
    stop_on_interrupt()
    try:
        main()
    except SystemExit as exc:
        exit_after_signal(exit_code_of(exc))
        raise
    # The teardown runs here, not at atexit, so a signal that lands during it sets the status.
    reset_services_on_exit()
    status = exit_status_after_stop(0)
    if status:
        raise typer.Exit(status)
