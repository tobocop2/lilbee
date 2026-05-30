"""Ollama local-server spec."""

from lilbee.providers.backend_names import BackendName
from lilbee.providers.local_servers.spec import LocalServerSpec

# Read-only like LM Studio: lilbee runs and lists Ollama models but never
# pulls them. supports_show stays on because /api/show surfaces generation
# defaults.
OLLAMA = LocalServerSpec(
    key="ollama",
    display_name=BackendName.OLLAMA,
    wire_prefix="ollama/",
    default_base_url="http://localhost:11434",
    url_patterns=("localhost:11434", "127.0.0.1:11434", "ollama"),
    appends_latest_tag=True,
    supports_pull=False,
    supports_show=True,
)
