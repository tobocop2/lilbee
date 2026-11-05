"""pi launcher: loads the lilbee pi extension for one session, then runs pi."""

from __future__ import annotations

import json
import os
import shutil

import typer

from lilbee.app.agent_configs.pi import pi_models
from lilbee.app.endpoints import MCP_PATH, OPENAI_PATH
from lilbee.catalog import agent_model_id
from lilbee.cli.launchers.launcher import LILBEE_TOKEN_ENV_VAR, run_launcher
from lilbee.cli.launchers.server import LOOPBACK, client_chat_ctx
from lilbee.cli.launchers.skill_install import bundled_pi_extension, bundled_skill_dir
from lilbee.core.config import cfg

_PI_INSTALL_HINT = (
    "pi binary not found on PATH. Install it with npm: @earendil-works/pi-coding-agent "
    "(needs Node 22.19 or newer)."
)
# Read by packages/lilbee/plugin/extensions/lilbee.js.
_OPENAI_URL_ENV_VAR = "LILBEE_OPENAI_URL"
_MCP_URL_ENV_VAR = "LILBEE_MCP_URL"
_MODELS_ENV_VAR = "LILBEE_PI_MODELS"


class PiLauncher:
    """``Launcher`` implementation for pi (@earendil-works/pi-coding-agent)."""

    name = "pi"
    install_hint = _PI_INSTALL_HINT

    def __init__(self, *, include_mcp: bool = True) -> None:
        self._include_mcp = include_mcp

    def find_binary(self) -> str | None:
        return shutil.which("pi")

    def prepare(
        self, *, token: str, port: int, model_refs: list[str]
    ) -> tuple[list[str], dict[str, str]]:
        base_url = f"http://{LOOPBACK}:{port}"
        served_ctx = client_chat_ctx(port)
        chat_ctx = cfg.chat_n_ctx_target if served_ctx is None else served_ctx
        env = {
            **os.environ,
            LILBEE_TOKEN_ENV_VAR: token,
            _OPENAI_URL_ENV_VAR: f"{base_url}{OPENAI_PATH}",
            _MODELS_ENV_VAR: json.dumps(pi_models(model_refs, chat_ctx)),
        }
        args = ["-e", str(bundled_pi_extension())]
        if self._include_mcp:
            env[_MCP_URL_ENV_VAR] = f"{base_url}{MCP_PATH}"
            args += ["--skill", str(bundled_skill_dir())]
        if str(cfg.chat_model) in model_refs:
            args += ["--model", f"lilbee/{agent_model_id(str(cfg.chat_model))}"]
        return (args, env)


def pi_cmd(
    mcp: bool | None = typer.Option(
        None,
        "--mcp/--no-mcp",
        help="Give pi lilbee's MCP search tool. Defaults to the "
        "agent_mcp_enabled config; --mcp/--no-mcp overrides it for this launch.",
    ),
) -> None:
    """Launch pi with lilbee as its model provider."""
    include_mcp = cfg.agent_mcp_enabled if mcp is None else mcp
    run_launcher(PiLauncher(include_mcp=include_mcp))
