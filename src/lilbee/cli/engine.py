"""Engine lifecycle commands: manage the shared engine without opening the TUI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from lilbee.app import engine_service
from lilbee.cli import theme
from lilbee.cli.app import console, json_out
from lilbee.core.config import cfg
from lilbee.providers.fleet.swap_manager import stop_engine
from lilbee.providers.roles import WorkerRole
from lilbee.runtime.console import PlainConsole
from lilbee.runtime.engine_lock import build_lock, machine_engine_dir, private_engine_dir

engine_app = typer.Typer(
    name="engine",
    help="Manage the local inference engine.",
    no_args_is_help=True,
)

_STOPPED = "Stopped the engine and freed its memory."
_NOTHING_RUNNING = "No engine is running."
_NO_BUILDS = "No engine builds are installed."


@engine_app.command("stop")
def stop() -> None:
    """Stop the shared engine now, whoever started it."""
    stopped: list[str] = []
    # Machine slot, this root's overflow dir, and the pre-upgrade legacy location
    # (base builds recorded state directly under data_dir): a stale legacy engine
    # must still be stoppable after an upgrade relocated the engine dirs.
    dirs = dict.fromkeys((machine_engine_dir(), private_engine_dir(cfg.data_root), cfg.data_dir))
    for engine_dir in dirs:
        # Serialize against a concurrent builder, whose spawn + state write + health
        # wait all run under this lock; an unlocked stop landing mid-build would kill
        # the just-spawned engine and unlink its fresh record. best_effort so a wedged
        # builder cannot make the off switch hang.
        with build_lock(engine_dir, best_effort=True):
            stopped.extend(stop_engine(engine_dir))
    if cfg.json_mode:
        json_out({"command": "engine stop", "stopped": sorted(set(stopped))})
        return
    console.print(_STOPPED if stopped else _NOTHING_RUNNING)


@engine_app.command("list")
def list_builds() -> None:
    """List the engine builds installed beside the models."""
    from lilbee.providers.fleet.engine_store import EngineStore, engines_root

    builds = EngineStore(engines_root(cfg.models_dir)).installed()
    if cfg.json_mode:
        json_out(
            {
                "command": "engine list",
                "builds": [
                    {"engine": b.engine, "version": b.version, "backend": b.backend} for b in builds
                ],
            }
        )
        return
    if not builds:
        console.print(_NO_BUILDS)
    for build in builds:
        console.print(f"{build.engine} {build.version} ({build.backend})")


@engine_app.command("install")
def install(
    name: str = typer.Argument(help="Engine name, as 'lilbee engine roles' shows it."),
    wheel: Annotated[
        Path | None,
        typer.Option("--wheel", help="Install from this wheel file instead of the packs folder."),
    ] = None,
) -> None:
    """Install an engine build from its wheel or from a pack in the packs folder."""
    try:
        built = engine_service.install(name, wheel)
    except engine_service.EngineRefusedError as exc:
        raise _refuse("engine install", exc) from exc
    label = f"{built.engine} {built.version} ({built.backend})"
    if cfg.json_mode:
        json_out({"command": "engine install", "installed": label})
        return
    console.print(f"Installed engine {label}.")


@engine_app.command("remove")
def remove(
    engine: str = typer.Argument(help="Engine name, as 'lilbee engine list' shows it."),
    version: str = typer.Argument(help="Build version."),
    backend: str = typer.Argument(help="Build backend."),
) -> None:
    """Remove one installed engine build."""
    from lilbee.providers.fleet.engine_store import (
        BundledEngineError,
        EngineInUseError,
        EngineNotInstalledError,
        EngineStore,
        StorePathError,
        engines_root,
        launch_uses_engine,
    )

    store = EngineStore(engines_root(cfg.models_dir))
    try:
        store.remove(engine, version, backend, in_use=launch_uses_engine)
    except (
        BundledEngineError,
        EngineInUseError,
        EngineNotInstalledError,
        StorePathError,
    ) as exc:
        if cfg.json_mode:
            json_out({"command": "engine remove", "error": str(exc)})
        else:
            PlainConsole(stderr=True).print(Text(str(exc), style=theme.ERROR), soft_wrap=True)
        raise typer.Exit(1) from exc
    if cfg.json_mode:
        json_out({"command": "engine remove", "removed": f"{engine} {version} ({backend})"})
        return
    console.print(f"Removed engine {engine} {version} ({backend}).")


def _refuse(command: str, exc: Exception) -> typer.Exit:
    """Show a refusal from the engine service on the surface and return the exit."""
    if cfg.json_mode:
        json_out({"command": command, "error": str(exc)})
    else:
        PlainConsole(stderr=True).print(Text(str(exc), style=theme.ERROR), soft_wrap=True)
    return typer.Exit(1)


@engine_app.command("roles")
def roles() -> None:
    """Show the engine of each role, with the notice when it fell back."""
    views = engine_service.roles()
    if cfg.json_mode:
        json_out(
            {
                "command": "engine roles",
                "roles": [
                    {
                        "role": str(v.role),
                        "model": v.model,
                        "format": v.format,
                        "engine": v.engine,
                        "notice": v.notice,
                    }
                    for v in views
                ],
            }
        )
        return
    for view in views:
        console.print(f"{view.role}: {view.model} ({view.format}) on {view.engine or 'no engine'}")
        if view.notice:
            console.print(view.notice)


@engine_app.command("use")
def use(
    role: Annotated[WorkerRole, typer.Argument(help="Role: chat, embed, rerank or vision.")],
    model: str = typer.Argument(help="The model of that role."),
    engine: str = typer.Argument(help="Engine name, or 'auto' to remove the choice."),
) -> None:
    """Choose the engine that runs the model of one role."""
    try:
        chosen = engine_service.use_for_role(role, model, engine)
    except (engine_service.EngineRefusedError, engine_service.EngineNotSavedError) as exc:
        raise _refuse("engine use", exc) from exc
    if cfg.json_mode:
        json_out({"command": "engine use", "role": str(role), "model": model, "engine": chosen})
        return
    console.print(f"{role} model {model} uses engine {chosen}. {engine_service.RESTART_NOTE}")


@engine_app.command("prefer")
def prefer(
    name: str = typer.Argument(help="Engine name, or 'auto' for the default for each model."),
) -> None:
    """Set the engine to use for every model that has no choice of its own."""
    try:
        chosen = engine_service.prefer(name)
    except (engine_service.EngineRefusedError, engine_service.EngineNotSavedError) as exc:
        raise _refuse("engine prefer", exc) from exc
    if cfg.json_mode:
        json_out({"command": "engine prefer", "engine": chosen})
        return
    console.print(f"The preferred engine is {chosen}. {engine_service.RESTART_NOTE}")
