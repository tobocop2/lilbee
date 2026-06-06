"""Launcher protocol and the orchestrator that runs any launcher."""

from __future__ import annotations

import subprocess
from typing import Protocol

import typer

from lilbee.cli.launchers.server import (
    ensure_server_running,
    installed_chat_model_refs,
    stop_spawned_server,
    wait_for_chat_warm,
)


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
    if not model_refs:
        # The client provider is written with no models, so it cannot use lilbee.
        # Some clients (e.g. opencode) then silently fall back to their own default
        # provider, so make the cause loud instead of leaving an empty picker.
        typer.secho(
            "Warning: no chat models are installed, so the launched client will have "
            "no lilbee models to select. Pull one first, e.g. "
            "'lilbee model pull Qwen/Qwen3-8B-GGUF'.",
            err=True,
            fg=typer.colors.YELLOW,
        )
    # Wait out the cold model load before handing off, so the client opens onto a
    # warm engine instead of an apparently-dead stream. Only meaningful when a
    # chat model is configured to warm.
    if model_refs:
        wait_for_chat_warm(port)
    extra_args, env = launcher.prepare(token=token, port=port, model_refs=model_refs)
    try:
        # binary resolved via the launcher's find_binary on PATH; no shell.
        result = subprocess.run([binary, *extra_args], env=env, check=False)  # noqa: S603
    finally:
        if spawned is not None:
            stop_spawned_server(spawned)
    raise typer.Exit(result.returncode)
