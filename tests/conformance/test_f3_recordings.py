"""A3 and B3: the adapters answer to the recordings of real OpenVINO and Lemonade runs."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.fleet.engines.lemonade import CHILD_PORTS, LEMONADE
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


def test_b3_lemonade_routes_replay_through_the_adapter_specs() -> None:
    doc = _recording("lemonade-cpu-thinkpad")
    served = {
        LEMONADE.role_spec(role, None, None).endpoint_path: role
        for role in (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)
    }
    answers = {e["path"]: (status, body) for e, status, body in _replay("lemonade-cpu-thinkpad")}
    assert set(served) <= set(answers)
    assert {status for status, _ in answers.values()} == {200}
    chat = answers["/v1/chat/completions"][1]["choices"][0]["message"]
    assert chat["content"].strip()
    assert LEMONADE.request_profile(WorkerRole.CHAT).reasoning_field in chat
    vectors = answers["/v1/embeddings"][1]["data"]
    assert {len(v["embedding"]) for v in vectors} == {doc["embed_dim"]}
    rerank_path = LEMONADE.request_profile(WorkerRole.RERANK).rerank_path
    assert rerank_path == "/v1/reranking"
    results = answers[rerank_path][1]["results"]
    assert max(results, key=lambda r: r["relevance_score"])["index"] == 1


def test_b3_the_recorded_process_tree_uses_the_ports_the_adapter_checks() -> None:
    text = (FIXTURES / "lemonade-cpu-thinkpad.log").read_text(encoding="utf-8")
    for port in CHILD_PORTS:
        assert f"--port {port}" in text
