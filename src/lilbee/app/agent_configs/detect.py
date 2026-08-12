"""Which of the supported agent CLIs are installed on this machine."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.app.agent_configs.document import AgentClient
from lilbee.core.system import find_executable


@dataclass(frozen=True)
class ClientDetection:
    """Whether one client's CLI is installed, and where it was found."""

    client: AgentClient
    cli_detected: bool
    cli_path: str | None


def detect_agent_client(client: AgentClient) -> ClientDetection:
    """Probe for *client*'s executable. Its name on disk is the client name."""
    path = find_executable(client.value)
    return ClientDetection(client=client, cli_detected=path is not None, cli_path=path)


def detect_agent_clients() -> list[ClientDetection]:
    """Probe for every supported client, in declaration order."""
    return [detect_agent_client(client) for client in AgentClient]
