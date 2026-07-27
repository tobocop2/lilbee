"""Two surfaces, one answer about what is safe to send a client.

A fleet ProviderError's message carries the dead engine's stderr: loopback
ports, engine paths, allocator failures. The completions surface redacts that
for backend-describing kinds. The SSE surface returned it verbatim.
"""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.server.chat_completions_api.errors import (
    _BACKEND_FAILURE_MESSAGE,
    _INFRASTRUCTURE_KINDS,
    _PROVIDER_KIND_CLASSIFICATIONS,
)
from lilbee.server.handlers.rag import _classify_stream_error

_ENGINE_STDERR = (
    "llama-server returned HTTP 502\nupstream server output:\n"
    "/opt/lilbee/engine/llama-server --port 20001 ...\ncudaMalloc failed: out of memory"
)


@pytest.mark.parametrize(
    "kind",
    [
        ProviderErrorKind.CAPACITY,
        ProviderErrorKind.PORT_CONFLICT,
        ProviderErrorKind.CONNECTION,
        ProviderErrorKind.SERVER,
    ],
)
def test_a_backend_failure_never_reaches_an_sse_client_verbatim(kind) -> None:
    _code, message = _classify_stream_error(ProviderError(_ENGINE_STDERR, kind=kind))
    assert "cudaMalloc" not in message
    assert "20001" not in message
    assert message == _BACKEND_FAILURE_MESSAGE


def test_a_caller_facing_kind_still_says_what_happened() -> None:
    _code, message = _classify_stream_error(
        ProviderError("too many tokens", kind=ProviderErrorKind.CONTEXT_OVERFLOW)
    )
    assert message == "too many tokens"


@pytest.mark.parametrize("kind", sorted(_INFRASTRUCTURE_KINDS))
def test_every_infrastructure_kind_has_an_http_classification(kind) -> None:
    # Dropping a kind from either table is how the redaction silently stops.
    assert kind in _PROVIDER_KIND_CLASSIFICATIONS


@pytest.mark.parametrize("kind", [ProviderErrorKind.CAPACITY, ProviderErrorKind.PORT_CONFLICT])
def test_the_kinds_this_epic_added_are_pinned_as_infrastructure(kind) -> None:
    assert kind in _INFRASTRUCTURE_KINDS
