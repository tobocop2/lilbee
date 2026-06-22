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
from lilbee.core.config import cfg
from lilbee.providers.model_ref import with_configured_remote_chat


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


def _warn_on_model_pin_gaps(model_refs: list[str]) -> None:
    """Warn when the launched client cannot open on a lilbee-served chat model."""
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
    elif str(cfg.chat_model) not in model_refs:
        # The startup pin would point at a model the provider does not serve,
        # so the client opens on its own default provider instead of lilbee.
        typer.secho(
            f"Warning: configured chat model '{cfg.chat_model}' is not installed; "
            "the launched client will not open on a lilbee model. Pull it first "
            "or set chat_model to an installed ref.",
            err=True,
            fg=typer.colors.YELLOW,
        )


def run_launcher(launcher: Launcher) -> None:
    """Find the client, ensure a lilbee server is up, prepare, exec, clean up."""
    binary = launcher.find_binary()
    if binary is None:
        typer.secho(launcher.install_hint, err=True, fg=typer.colors.RED)
        raise typer.Exit(1)
    # The launcher only reads the registry and talks to the spawned `lilbee serve`
    # over HTTP; it runs no inference itself. Skip the eager warm so get_services()
    # here doesn't start a second llama-swap that races the server's for the model
    # port (the loser gets connection-refused). The spawned serve warms its own.
    cfg.worker_pool_eager_start = False
    (token, port), spawned = ensure_server_running()
    # Everything after the spawn runs under the finally so a raise from prepare()
    # (e.g. the user declining opencode setup) or the warm wait can't leak the
    # spawned `lilbee serve` process.
    try:
        native_refs = installed_chat_model_refs()
        model_refs = with_configured_remote_chat(native_refs, cfg.chat_model)
        _warn_on_model_pin_gaps(model_refs)
        # Wait out the cold model load before handing off, so the client opens onto a
        # warm engine instead of an apparently-dead stream. Only meaningful when a
        # native chat model is installed to warm; a remote-configured model has no
        # local load to wait for.
        if native_refs:
            wait_for_chat_warm(port)
        extra_args, env = launcher.prepare(token=token, port=port, model_refs=model_refs)
        # The client paints its own UI only after its runtime boots, a few silent
        # seconds; announce the handoff so the warm bar isn't followed by a dead
        # screen with no explanation.
        typer.secho(f"Launching {launcher.name}...", fg=typer.colors.GREEN)
        # binary resolved via the launcher's find_binary on PATH; no shell.
        result = subprocess.run([binary, *extra_args], env=env, check=False)  # noqa: S603
    finally:
        if spawned is not None:
            stop_spawned_server(spawned)
    raise typer.Exit(result.returncode)
