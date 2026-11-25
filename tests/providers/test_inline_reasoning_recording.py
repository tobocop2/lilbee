"""A recorded engine run whose reasoning sits inline in the answer text, with no reasoning field.

The recording `llama.cpp-cpu-inline` is a harness run of a llama.cpp server started with its
reasoning parser off, so the model's own think block stays in the message content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.base import THINK_CLOSE_TAG, THINK_OPEN_TAG
from lilbee.providers.sdk_llm_provider import SdkLLMProvider
from lilbee.retrieval.reasoning import split_reasoning
from tests import _fake_sdk as fakes
from tests.conformance.harness import reasoning_fault, reasoning_split_fault
from tests.conformance.preset_check import load_fixture
from tests.conformance.request_set import REASONING_ASK

RECORDING = (
    Path(__file__).resolve().parents[1]
    / "conformance"
    / "fixtures"
    / "engines"
    / "llama.cpp-cpu-inline.json"
)
_MESSAGES = [REASONING_ASK]
_MODEL = "ollama/qwen3:0.6b"
_SLICE = 37
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_choice() -> dict[str, Any]:
    """The first choice the engine answered to the reasoning request, as it came on the wire."""
    _setup, _results, exchanges, _doc = load_fixture(RECORDING)
    found = [
        json.loads(exchange.body)["choices"][0]
        for exchange in exchanges
        if exchange.path == "/v1/chat/completions"
        and exchange.status == 200
        and REASONING_ASK["content"] in json.dumps(exchange.request)
    ]
    assert found, "the recording holds no answer to the reasoning request"
    return found[0]


def _content() -> str:
    return str(_reasoning_choice()["message"]["content"])


def _provider() -> SdkLLMProvider:
    return SdkLLMProvider(LiterLlmBackend())


def test_the_recording_is_a_cpu_run_that_passed_every_step() -> None:
    setup, results, exchanges, doc = load_fixture(RECORDING)
    assert exchanges
    assert doc["strict"] is True
    assert set(results) >= {"reasoning", "named"}
    assert all(outcome.ok for outcome in results.values())
    assert RECORDING.with_suffix(".log").read_text(encoding="utf-8").splitlines()[3] == (
        "backend: cpu"
    )
    assert setup.preset == "llama-cpp-cpu-inline"


def test_the_log_names_the_server_flag_that_leaves_the_reasoning_inline() -> None:
    lines = RECORDING.with_suffix(".log").read_text(encoding="utf-8").splitlines()
    flags = [line for line in lines if line.startswith("server-flags: ")]
    assert len(flags) == 1
    assert "--reasoning-format none" in flags[0]


def test_the_engine_put_the_reasoning_inline_and_sent_no_field() -> None:
    choice = _reasoning_choice()
    message = choice["message"]
    assert [field for field in _REASONING_FIELDS if message.get(field)] == []
    content = message["content"]
    assert content.count(THINK_OPEN_TAG) == 1
    assert content.count(THINK_CLOSE_TAG) == 1
    block, _, answer = content.partition(THINK_CLOSE_TAG)
    assert block.removeprefix(THINK_OPEN_TAG).strip()
    assert answer.strip()
    assert reasoning_fault(message, choice["finish_reason"]) == ""
    assert reasoning_split_fault(message, choice["finish_reason"]) == ""


def test_the_inline_block_splits_into_reasoning_and_answer() -> None:
    split = split_reasoning(_content())
    assert split.reasoning.strip()
    assert split.answer.strip()
    assert THINK_OPEN_TAG not in split.answer
    assert THINK_CLOSE_TAG not in split.answer


def test_a_non_stream_chat_keeps_the_inline_block() -> None:
    content = _content()
    fake = fakes.FakeSdk()
    fake.chat_result = fakes.text_response(content)
    with fakes.installed(fake):
        text = _provider().chat(_MESSAGES, model=_MODEL).text
    assert text == content


def test_a_stream_chat_keeps_the_inline_block() -> None:
    content = _content()
    fake = fakes.FakeSdk()
    fake.stream_chunks = [
        *(fakes.chunk(content[i : i + _SLICE]) for i in range(0, len(content), _SLICE)),
        fakes.chunk(finish="stop"),
    ]
    with fakes.installed(fake):
        streamed = _provider().chat(_MESSAGES, stream=True, model=_MODEL)
        text = "".join(item for item in streamed if isinstance(item, str))
    assert text == content


@pytest.mark.parametrize("check", [reasoning_fault, reasoning_split_fault])
def test_the_checks_refuse_the_recorded_reply_without_its_block(check: Any) -> None:
    choice = _reasoning_choice()
    answer = split_reasoning(choice["message"]["content"]).answer
    assert check({"content": answer}, choice["finish_reason"]) != ""
    assert check(choice["message"], choice["finish_reason"]) == ""
