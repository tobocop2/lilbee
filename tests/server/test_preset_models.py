"""The preset response models: one dict shape for the CLI, HTTP and MCP surfaces."""

from lilbee.app.presets import ServerPreset
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import RerankWire
from lilbee.server.models import PresetListResponse, PresetRemovedResponse, PresetResponse


def test_a_view_dumps_to_plain_json_values_with_no_key_field() -> None:
    view = ServerPreset(
        "omlx",
        "http://127.0.0.1:8000/v1",
        (WorkerRole.CHAT, WorkerRole.RERANK),
        True,
        True,
        RerankWire.TEI,
    )
    assert PresetResponse.from_view(view).model_dump(mode="json") == {
        "name": "omlx",
        "base_url": "http://127.0.0.1:8000/v1",
        "roles": ["chat", "rerank"],
        "has_key": True,
        "valid": True,
        "rerank_wire": "tei",
    }


def test_an_invalid_entry_has_a_null_wire() -> None:
    view = ServerPreset("bad", "", (WorkerRole.CHAT,), False, False, None)
    assert PresetResponse.from_view(view).model_dump(mode="json")["rerank_wire"] is None


def test_the_list_and_removed_envelopes() -> None:
    assert PresetListResponse(presets=[]).model_dump(mode="json") == {"presets": []}
    assert PresetRemovedResponse(removed="omlx").model_dump(mode="json") == {"removed": "omlx"}
