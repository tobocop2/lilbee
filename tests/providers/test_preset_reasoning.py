"""A preset chat keeps the server-extracted reasoning, driven by the recorded engine replies.

The liter-llm library is replaced by the scripted stand-in the backend tests use, so the
real library is not exercised here. The reasoning and answer text come from the recordings.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.base import THINK_CLOSE_TAG, THINK_OPEN_TAG
from lilbee.providers.roles import WorkerRole
from lilbee.providers.sdk_llm_provider import SdkLLMProvider
from tests import _fake_sdk as fakes

_ENGINES = Path(__file__).resolve().parents[1] / "conformance" / "fixtures" / "engines"
_RECORDINGS = ("llama.cpp-cpu", "mistralrs-cpu")
_PROMPT = "Think it through, then answer: what is 17 plus 25?"
_MESSAGES = [{"role": "user", "content": _PROMPT}]
_MODEL = "ollama/qwen3:0.6b"
_SLICE = 37


def _find_message(name: str) -> dict[str, Any]:
    data = json.loads((_ENGINES / f"{name}.json").read_text(encoding="utf-8"))
    found = [
        json.loads(ex["body"])["choices"][0]["message"]
        for ex in _walk(data)
        if ex.get("path") == "/v1/chat/completions"
        and _PROMPT in json.dumps(ex.get("request"))
        and ex.get("status") == 200
    ]
    assert found, name
    return found[0]


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "body" in node and "request" in node:
            yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _slices(text: str) -> list[str]:
    return [text[i : i + _SLICE] for i in range(0, len(text), _SLICE)]


def _provider() -> SdkLLMProvider:
    return SdkLLMProvider(LiterLlmBackend())


def _streamed() -> str:
    """The text pieces of a streamed preset chat, joined; the finish marker is not text."""
    return "".join(
        item
        for item in _provider().chat(_MESSAGES, stream=True, model=_MODEL)
        if isinstance(item, str)
    )


@pytest.mark.parametrize("name", _RECORDINGS)
class TestPresetKeepsReasoning:
    def test_recording_carries_reasoning_and_answer(self, name: str) -> None:
        message = _find_message(name)
        assert message["reasoning_content"].strip()
        assert message["content"].strip()

    def test_non_stream_chat_inlines_the_reasoning(self, name: str) -> None:
        message = _find_message(name)
        fake = fakes.FakeSdk()
        fake.chat_result = fakes.text_response(
            message["content"], reasoning=message["reasoning_content"]
        )
        with fakes.installed(fake):
            text = _provider().chat(_MESSAGES, model=_MODEL).text
        reasoning, _, answer = text.removeprefix(THINK_OPEN_TAG).partition(THINK_CLOSE_TAG)
        assert text.startswith(THINK_OPEN_TAG)
        assert reasoning == message["reasoning_content"]
        assert answer == message["content"]
        assert message["reasoning_content"] not in answer

    def test_stream_chat_inlines_the_reasoning(self, name: str) -> None:
        message = _find_message(name)
        fake = fakes.FakeSdk()
        fake.stream_chunks = [
            *(fakes.chunk(reasoning=piece) for piece in _slices(message["reasoning_content"])),
            *(fakes.chunk(piece) for piece in _slices(message["content"])),
            fakes.chunk(finish="stop"),
        ]
        with fakes.installed(fake):
            text = _streamed()
        reasoning, _, answer = text.removeprefix(THINK_OPEN_TAG).partition(THINK_CLOSE_TAG)
        assert text.startswith(THINK_OPEN_TAG)
        assert reasoning == message["reasoning_content"]
        assert answer == message["content"]

    def test_stream_that_ends_inside_reasoning_closes_the_block(self, name: str) -> None:
        message = _find_message(name)
        fake = fakes.FakeSdk()
        fake.stream_chunks = [
            fakes.chunk(reasoning=message["reasoning_content"]),
            fakes.chunk(finish="length"),
        ]
        with fakes.installed(fake):
            text = _streamed()
        assert text == f"{THINK_OPEN_TAG}{message['reasoning_content']}{THINK_CLOSE_TAG}"


def test_stream_cut_off_inside_reasoning_closes_the_block() -> None:
    fake = fakes.FakeSdk()
    fake.stream_chunks = [fakes.chunk(reasoning="partial")]
    with fakes.installed(fake):
        assert _streamed() == f"{THINK_OPEN_TAG}partial{THINK_CLOSE_TAG}"


def test_non_chat_role_drops_the_reasoning() -> None:
    fake = fakes.FakeSdk()
    fake.chat_result = fakes.text_response("page text", reasoning="thinking")
    fake.stream_chunks = [fakes.chunk(reasoning="thinking"), fakes.chunk("page", finish="stop")]
    with fakes.installed(fake):
        provider = _provider()
        assert provider.chat(_MESSAGES, model=_MODEL, role=WorkerRole.VISION).text == "page text"
        streamed = provider.chat(_MESSAGES, stream=True, model=_MODEL, role=WorkerRole.VISION)
        assert "".join(i for i in streamed if isinstance(i, str)) == "page"


def test_reply_without_reasoning_is_unchanged() -> None:
    fake = fakes.FakeSdk()
    fake.chat_result = fakes.text_response("Pong.")
    fake.stream_chunks = [fakes.chunk("Po"), fakes.chunk("ng.", finish="stop")]
    with fakes.installed(fake):
        assert _provider().chat(_MESSAGES, model=_MODEL).text == "Pong."
        assert _streamed() == "Pong."
