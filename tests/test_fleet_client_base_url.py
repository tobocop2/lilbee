"""The base URL of a server client: a trailing slash never reaches the request URLs."""

from __future__ import annotations

import httpx
import pytest

from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole


def _client(base_url: str) -> LlamaServerClient:
    return LlamaServerClient(base_url, "m", profile=LLAMA.request_profile(WorkerRole.CHAT))


@pytest.mark.parametrize(
    ("base_url", "host"),
    [
        ("http://gpu0/", "gpu0"),
        ("http://gpu0//", "gpu0"),
        ("http://gpu0", "gpu0"),
        ("http://gpuX/", "gpuX"),
    ],
    ids=["one-slash", "two-slashes", "no-slash", "host-ends-in-x"],
)
def test_a_base_url_gives_single_slash_request_urls_and_keeps_its_host(
    base_url: str, host: str
) -> None:
    client = _client(base_url)

    assert client.sent(httpx.Request("GET", f"http://{host}/v1/models"))
    assert str(client._http.build_request("GET", "/v1/models").url) == f"http://{host}/v1/models"
