"""LM Studio local-server spec."""

from lilbee.providers.backend_names import BackendName
from lilbee.providers.local_servers.spec import LocalServerSpec

# litellm's lm_studio provider posts to {api_base}/chat/completions, so the
# base URL must carry /v1 (what LM Studio's server panel shows). It injects a
# placeholder key, so no API key is required.
LM_STUDIO = LocalServerSpec(
    key="lm_studio",
    display_name=BackendName.LM_STUDIO,
    wire_prefix="lm_studio/",
    default_base_url="http://localhost:1234/v1",
    url_patterns=("localhost:1234", "127.0.0.1:1234"),
    appends_latest_tag=False,
    supports_pull=False,
    supports_show=False,
)
