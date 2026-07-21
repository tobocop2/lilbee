"""hermes launcher: registers lilbee in the user's real ~/.hermes, then runs hermes."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer
import yaml

from lilbee.cli.agent_configs import config_file
from lilbee.cli.agent_configs.hermes import hermes_config
from lilbee.cli.agent_configs.merge import deep_merge, prune_lilbee
from lilbee.cli.launchers.hermes_mcp import ensure_hermes_http_mcp
from lilbee.cli.launchers.launcher import LILBEE_TOKEN_ENV_VAR, run_launcher
from lilbee.cli.launchers.server import LOOPBACK, client_chat_ctx
from lilbee.cli.launchers.skill_install import install_bundled_skill
from lilbee.core.config import cfg

_HERMES_INSTALL_HINT = (
    "hermes binary not found on PATH. Install it from https://github.com/NousResearch/hermes-agent."
)
_TOKEN_REF = "${" + LILBEE_TOKEN_ENV_VAR + "}"
_MCP_CONTAINER_KEY = "mcp_servers"
_CONFIG_LABEL = "hermes config (config.yaml)"
# hermes refuses to start against a model whose window is under this, so a smaller
# one is worth naming here rather than leaving hermes to fail after the handoff.
_HERMES_MIN_CTX = 64_000
# hermes config keys gating its own auto-installs (security.allow_lazy_installs).
_SECURITY_KEY = "security"
_ALLOW_LAZY_INSTALLS_KEY = "allow_lazy_installs"


def _hermes_home() -> Path:
    """The user's real hermes home; never relocated, so memory and skills are shared."""
    return Path.home() / ".hermes"


def _hermes_config_path() -> Path:
    return _hermes_home() / "config.yaml"


def _hermes_env_path() -> Path:
    return _hermes_home() / ".env"


def _hermes_skill_dest() -> Path:
    return _hermes_home() / "skills" / "lilbee-mcp"


def _upsert_env_token(path: Path, token: str) -> None:
    """Set ``LILBEE_TOKEN=<token>`` in the hermes ``.env`` (0600), preserving other lines."""
    line = f"{LILBEE_TOKEN_ENV_VAR}={token}"
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in existing if not ln.startswith(f"{LILBEE_TOKEN_ENV_VAR}=")]
    # atomic_write_text creates the file 0600 and keeps that mode across the
    # replace, so the token is never briefly readable and needs no chmod after.
    config_file.atomic_write_text(path, "\n".join([*kept, line]) + "\n")


def warn_hermes_ungrounded() -> None:
    """Say plainly that hermes will run ungrounded when its MCP search did not connect.

    Without the ``mcp`` extra hermes never calls ``lilbee_search`` and answers from
    its own training, silently at exit 0, so the install hint alone is easy to miss.
    """
    typer.secho(
        "Warning: hermes could not connect lilbee's search (MCP), so it will run "
        "WITHOUT grounding -- it will not call lilbee_search and its answers come "
        "from its own training, not your indexed docs. Install hermes's mcp extra "
        "(shown above) and relaunch to ground it.",
        err=True,
        fg=typer.colors.YELLOW,
    )


def warn_if_below_hermes_minimum(chat_ctx: int | None) -> None:
    """Tell the user up front when hermes will reject the window lilbee serves."""
    if chat_ctx is None or chat_ctx >= _HERMES_MIN_CTX:
        return
    typer.secho(
        f"Warning: hermes requires at least a {_HERMES_MIN_CTX:,}-token context and "
        f"lilbee serves {chat_ctx:,}, so hermes will refuse to start. Chat with a "
        "longer-context model, a smaller quantization, or a higher gpu_memory_fraction.",
        err=True,
        fg=typer.colors.YELLOW,
    )


class HermesLauncher:
    """``Launcher`` implementation for hermes-agent."""

    name = "hermes"
    install_hint = _HERMES_INSTALL_HINT

    def __init__(self, *, include_mcp: bool = True) -> None:
        self._include_mcp = include_mcp
        self._binary: str | None = None

    def find_binary(self) -> str | None:
        self._binary = shutil.which("hermes")
        return self._binary

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        config = config_file.load_config_dict(
            _hermes_config_path(),
            parse=yaml.safe_load,
            parse_error=yaml.YAMLError,
            label=_CONFIG_LABEL,
        )
        chat_ctx = client_chat_ctx(port)
        warn_if_below_hermes_minimum(chat_ctx)
        fragment = hermes_config(
            base_url=f"http://{LOOPBACK}:{port}",
            api_key=_TOKEN_REF,
            model_refs=model_refs,
            default_ref=str(cfg.chat_model),
            chat_ctx=chat_ctx,
            include_mcp=self._include_mcp,
        )
        deep_merge(config, fragment)
        if not self._include_mcp:
            prune_lilbee(config, _MCP_CONTAINER_KEY)
        config_file.atomic_write_text(
            _hermes_config_path(), yaml.safe_dump(config, sort_keys=False)
        )
        _upsert_env_token(_hermes_env_path(), token)
        if self._include_mcp:
            install_bundled_skill(_hermes_skill_dest())
            # hermes ships HTTP MCP behind the optional `mcp` extra; without it
            # lilbee's MCP search shows "0 connected". Set it up before launch,
            # honoring hermes's own auto-install security gate.
            # _binary is non-None: run_launcher called find_binary() and gated on it.
            if self._binary is not None:
                allow_lazy = bool(
                    (config.get(_SECURITY_KEY) or {}).get(_ALLOW_LAZY_INSTALLS_KEY, True)
                )
                mcp_ready = ensure_hermes_http_mcp(
                    self._binary, allow_lazy_installs=allow_lazy, echo=typer.echo
                )
                # Requested but not connected: don't hand off a silently ungrounded hermes.
                if not mcp_ready:
                    warn_hermes_ungrounded()
        return ([], {**os.environ, LILBEE_TOKEN_ENV_VAR: token})


def hermes_cmd(
    mcp: bool | None = typer.Option(
        None,
        "--mcp/--no-mcp",
        help="Register lilbee's MCP search tool into hermes. Defaults to the "
        "agent_mcp_enabled config; --mcp/--no-mcp overrides it for this launch.",
    ),
) -> None:
    """Launch hermes with lilbee registered as its model provider."""
    include_mcp = cfg.agent_mcp_enabled if mcp is None else mcp
    run_launcher(HermesLauncher(include_mcp=include_mcp))
