"""Serve (HTTP API) and mcp (stdio) server-boot commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.config import cfg

if TYPE_CHECKING:
    import uvicorn


def _port_file() -> Path:
    return cfg.data_dir / "server.port"


async def _run_server(server: uvicorn.Server, config: uvicorn.Config, host: str) -> None:
    """Start uvicorn, write port file, and clean up on shutdown."""
    import atexit

    port_path = _port_file()

    def _cleanup_port_file() -> None:
        port_path.unlink(missing_ok=True)

    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup()
    try:
        if server.servers:
            sock = server.servers[0].sockets[0]
            actual_port = sock.getsockname()[1]
            port_path.parent.mkdir(parents=True, exist_ok=True)
            port_path.write_text(str(actual_port))
            atexit.register(_cleanup_port_file)
            console.print(f"Listening on http://{host}:{actual_port}")
        await server.main_loop()
    finally:
        port_path.unlink(missing_ok=True)
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

    import logging

    import uvicorn

    from lilbee.server import create_app

    logging.getLogger("asyncio").setLevel(logging.ERROR)

    config = uvicorn.Config(create_app(), host=cfg.server_host, port=cfg.server_port)
    server = uvicorn.Server(config)
    asyncio.run(_run_server(server, config, cfg.server_host))


def mcp_cmd() -> None:
    """Start the MCP server (stdio transport) for agent integration."""
    from lilbee.mcp import main

    main()
