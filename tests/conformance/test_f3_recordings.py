"""A3: the adapter answers to the recordings of real OpenVINO runs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.fleet.engines.openvino import OPENVINO, listed_devices
from lilbee.providers.roles import WorkerRole
from tests.conformance.preset_check import Exchange, replaying

FIXTURES = Path(__file__).parent / "fixtures" / "engines"
_LIST_MARK = "--- llama-server --list-devices"
_NEXT_MARK = "---"


def _recording(stem: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{stem}.json").read_text(encoding="utf-8"))


def _post(base: str, exchange: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(  # noqa: S310
        base + exchange["path"],
        data=json.dumps(exchange["request"]).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        return response.status, json.loads(response.read())


def _replay(stem: str) -> list[tuple[dict[str, Any], int, dict[str, Any]]]:
    doc = _recording(stem)
    with replaying([Exchange(**e) for e in doc["exchanges"]]) as base:
        return [(e, *_post(base, e)) for e in doc["exchanges"]]


def _device_section(stem: str) -> str:
    text = (FIXTURES / f"{stem}.log").read_text(encoding="utf-8")
    start = text.index(_LIST_MARK) + len(_LIST_MARK)
    end = text.find(f"\n{_NEXT_MARK}", start)
    return text[start : end if end != -1 else None]


@pytest.mark.parametrize("stem", ["openvino-cpu", "openvino-gpu"])
def test_a3_openvino_chat_replays_through_the_adapter_profile(stem: str) -> None:
    field = OPENVINO.request_profile(WorkerRole.CHAT).reasoning_field
    spec = OPENVINO.role_spec(WorkerRole.CHAT, None, None)
    [(exchange, status, body)] = _replay(stem)
    assert exchange["path"] == spec.endpoint_path
    assert status == 200
    message = body["choices"][0]["message"]
    assert message["content"].strip()
    assert field in message


def test_a3_the_cpu_recording_lists_the_cpu_and_the_gpu_recording_lists_both() -> None:
    assert listed_devices(_device_section("openvino-cpu")) == {"CPU"}
    assert listed_devices(_device_section("openvino-gpu")) == {"CPU", "GPU"}


def test_a3_the_recording_serves_chat_only_as_the_adapter_claims() -> None:
    for stem in ("openvino-cpu", "openvino-gpu"):
        assert _recording(stem)["roles"] == ["chat"]
    assert OPENVINO.served_roles == {WorkerRole.CHAT}
