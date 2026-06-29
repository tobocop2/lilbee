"""``lilbee launch <client>`` sub-app."""

from __future__ import annotations

import typer

from lilbee.cli.launchers.hermes import hermes_cmd
from lilbee.cli.launchers.opencode import opencode_cmd

launch_app = typer.Typer(help="Launch a third-party AI client wired to lilbee.")
launch_app.command("opencode")(opencode_cmd)
launch_app.command("hermes")(hermes_cmd)
