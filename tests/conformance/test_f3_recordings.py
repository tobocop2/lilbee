"""A3 and B3: the adapters answer to the recordings of real OpenVINO and Lemonade runs."""

from __future__ import annotations

import itertools
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.fleet.engines import lemonade
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.openvino import OPENVINO, parse_devices
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


def _names(stem: str) -> set[str]:
    return {d.name for d in parse_devices(_device_section(stem))}


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
    assert _names("openvino-cpu") == {"CPU"}
    assert _names("openvino-gpu") == {"CPU", "GPU"}


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


F5B = FIXTURES / "lemonade-f5b"
_ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)


def _p4_children() -> dict[str, dict[str, str]]:
    """Role -> the flags of the llama-server child in the recorded process tree of run p4."""
    found: dict[str, dict[str, str]] = {}
    for line in (F5B / "p4.txt").read_text(encoding="utf-8").splitlines():
        match = re.search(r"llama-server -m \S+/view/(\w+)/lilbee-(\w+)\.gguf (.*)", line)
        if match:
            flags = match.group(3).split()
            found[match.group(2)] = {flag: value for flag, value in itertools.pairwise(flags)}
            found[match.group(2)]["mode"] = match.group(1)
    return found


def test_the_f5b_fixtures_keep_their_source_lines() -> None:
    for name in ("p1.txt", "p2.txt", "p3.txt", "p4.txt", "p5.txt"):
        assert (F5B / name).read_text(encoding="utf-8").startswith("source: ssh thinkpad")


def test_r7_three_lemond_processes_serve_three_roles_with_recipe_flags_and_no_download() -> None:
    text = (F5B / "p4.txt").read_text(encoding="utf-8")
    assert len(re.findall(r"/lemond --port 1817\d ", text)) == 3
    children = _p4_children()
    assert set(children) == {role.value for role in _ROLES}
    assert children["chat"]["--ctx-size"] == "2048"
    assert children["chat"]["--parallel"] == "2"
    assert children["rerank"]["--ctx-size"] == "2048"
    for role in _ROLES:
        assert children[role.value]["mode"] == lemonade._MODES[role]
    bins = text.split("--- downloads?")[1]
    assert "llamacpp" not in bins
    assert bins.count("/cache/bin:") == len(_ROLES)


def test_r8_the_recorded_embed_server_ran_with_more_context_than_the_recipe_asked() -> None:
    script = (F5B / "p4.sh").read_text(encoding="utf-8")
    assert '"ctx_size":4096' in script
    assert _p4_children()["embed"]["--ctx-size"] == "8192"
