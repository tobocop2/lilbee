"""Serve (HTTP API) and mcp (stdio) server-boot commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.commands.serve_logging import setup_server_logging
from lilbee.core.config import cfg

if TYPE_CHECKING:
    import uvicorn


def _port_file() -> Path:
    return cfg.data_dir / "server.port"


def _log_loop_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
    exc = context.get("exception")
    # isinstance: asyncio's context dict is untyped; "exception" may be absent
    if isinstance(exc, BaseException):
        logging.getLogger(__name__).error("asyncio task error", exc_info=exc)
    else:
        logging.getLogger(__name__).error("asyncio task error: %s", context.get("message"))


async def _run_server(server: uvicorn.Server, config: uvicorn.Config, host: str) -> None:
    """Start uvicorn, write port file, and clean up on shutdown."""
    import atexit

    from lilbee.parent_monitor import parse_parent_pid, watch_parent_async

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_log_loop_exception)

    port_path = _port_file()

    def _cleanup_port_file() -> None:
        port_path.unlink(missing_ok=True)

    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)

    # `server.servers` is set inside `startup()`. The finally below must skip
    # `shutdown()` when startup never ran: uvicorn dereferences `self.servers`
    # there and the resulting AttributeError would mask the original failure.
    started = False
    parent_watcher: asyncio.Task[None] | None = None
    try:
        await server.startup()
        started = True

        parent_pid = parse_parent_pid()
        if parent_pid is not None:

            def _on_parent_death() -> None:
                server.should_exit = True

            parent_watcher = asyncio.create_task(watch_parent_async(parent_pid, _on_parent_death))

        if server.servers:
            sock = server.servers[0].sockets[0]
            actual_port = sock.getsockname()[1]
            port_path.parent.mkdir(parents=True, exist_ok=True)
            port_path.write_text(str(actual_port))
            atexit.register(_cleanup_port_file)
            console.print(f"Listening on http://{host}:{actual_port}")
        await server.main_loop()
    finally:
        if parent_watcher is not None and not parent_watcher.done():
            parent_watcher.cancel()
        port_path.unlink(missing_ok=True)
        if started:
            # Suppress AttributeError from a partial uvicorn bring-up so any
            # original exception from main_loop reaches the caller intact.
            with contextlib.suppress(AttributeError):
                await server.shutdown()


def serve(
    host: str = typer.Option(None, "--host", "-H", help="Bind address (default: 127.0.0.1)"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: 0/random)"),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Start the HTTP API server."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if host is not None:
        cfg.server_host = host
    if port is not None:
        cfg.server_port = port

    setup_server_logging()

    import uvicorn

    from lilbee.server import create_app

    logging.getLogger("asyncio").setLevel(logging.ERROR)

    config = uvicorn.Config(create_app(), host=cfg.server_host, port=cfg.server_port)
    server = uvicorn.Server(config)
    asyncio.run(_run_server(server, config, cfg.server_host))


def mcp_cmd() -> None:
    """Start the MCP server (stdio transport) for agent integration."""
    from lilbee.mcp_server import main

    main()
