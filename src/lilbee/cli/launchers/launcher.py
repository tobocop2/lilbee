"""Launcher protocol and the orchestrator that runs any launcher."""

from __future__ import annotations

import subprocess
from typing import Protocol

import typer

from lilbee.cli.commands.agent_config import installed_chat_model_refs
from lilbee.cli.launchers.server import ensure_server_running, stop_spawned_server


class Launcher(Protocol):
    """A third-party AI client that lilbee knows how to launch."""

    name: str
    """CLI subcommand name; ``lilbee launch <name>`` runs this launcher."""

    install_hint: str
    """User-facing message shown when ``find_binary`` returns None."""

    def find_binary(self) -> str | None:
        """Return the absolute path to the client binary, or None if not installed."""
        ...

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        """Return ``(extra_args, env)`` for the client invocation.

        Side effects (skill installs, picker-state writes, config-file
        materialization) happen here. The orchestrator does not introspect
        them; whatever the launcher decides is the launcher's business.
        """
        ...


def run_launcher(launcher: Launcher) -> None:
    """Find the client, ensure a lilbee server is up, prepare, exec, clean up."""
    binary = launcher.find_binary()
    if binary is None:
        typer.secho(launcher.install_hint, err=True, fg=typer.colors.RED)
        raise typer.Exit(1)
    (token, port), spawned = ensure_server_running()
    model_refs = installed_chat_model_refs()
    extra_args, env = launcher.prepare(token=token, port=port, model_refs=model_refs)
    try:
        # binary resolved via the launcher's find_binary on PATH; no shell.
        result = subprocess.run([binary, *extra_args], env=env, check=False)  # noqa: S603
    finally:
        if spawned is not None:
            stop_spawned_server(spawned)
    raise typer.Exit(result.returncode)
