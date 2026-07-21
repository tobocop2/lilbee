"""Chat backends for question authoring and judging."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from evals.deps import install_hint

ChatFn = Callable[[str], str]

JUDGE_BASE_URL_ENV = "LILBEE_EVAL_JUDGE_BASE_URL"
JUDGE_MODEL_ENV = "LILBEE_EVAL_JUDGE_MODEL"
JUDGE_API_KEY_ENV = "LILBEE_EVAL_JUDGE_API_KEY"

RAGAS_INSTALL_HINT = install_hint("ragas", "for judging")

CHAT_MAX_TOKENS = 256
CHAT_TIMEOUT_SECONDS = 300.0
WARM_ATTEMPTS = 30
WARM_DELAY_SECONDS = 10.0


def openai_chat_fn(
    base_url: str, model: str, api_key: str | None = None, *, client: httpx.Client | None = None
) -> ChatFn:
    """Prompt-to-text over an OpenAI-compatible /chat/completions endpoint."""
    http = client or httpx.Client(timeout=CHAT_TIMEOUT_SECONDS)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"{base_url.rstrip('/')}/chat/completions"

    def chat(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": CHAT_MAX_TOKENS,
        }
        response = http.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    return chat


def lilbee_chat_fn() -> ChatFn:
    """Prompt-to-text on the configured lilbee chat model, reasoning stripped."""
    from lilbee.app.services import get_services  # heavy: builds the service container
    from lilbee.retrieval.reasoning import strip_reasoning

    provider = get_services().provider
    options = {"temperature": 0, "num_predict": CHAT_MAX_TOKENS, "think": False}

    def chat(prompt: str) -> str:
        result = provider.chat([{"role": "user", "content": prompt}], options=options)
        return strip_reasoning(result.text)

    return chat


@dataclass(frozen=True)
class JudgeBackend:
    """The judge's identity, its ragas LLM, and a plain chat fn for warming.

    ``llm`` is what ragas' rubric metrics grade through. ``chat`` exists only so
    ``warm_chat`` can block on model load over the same endpoint without
    spending a structured-output call to do it.
    """

    llm: Any
    chat: ChatFn
    model: str
    base_url: str


def ragas_judge_llm(
    base_url: str, model: str, api_key: str | None, temperature: float = 0.0
) -> Any:
    """A ragas LLM bound to this endpoint, for structured-output grading.

    The client is async because ragas' rubric metrics grade through
    ``agenerate``, which refuses a synchronous client outright.

    Without an explicit client here, ragas resolves a process-global default
    model from the environment, so the manifest's pinned judge and temperature
    would not be what actually graded the answers.
    """
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(RAGAS_INSTALL_HINT) from exc
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
    return llm_factory(model, provider="openai", client=client, temperature=temperature)


def judge_backend() -> JudgeBackend:
    """The judge backend, which must be an endpoint separate from the system under test.

    There is deliberately no fallback to the configured lilbee chat model. That
    model authors the questions and generates both arms' answers, so falling back
    to it means one model grades its own output against ground truth it
    paraphrased. Self-preference bias largely cancels on a mean where both arms
    share the generator, but not on the per-question variance the noise floor is
    supposed to bound, and a self-consistent grader shrinks that floor and widens
    what the report calls significant.
    """
    base_url = os.environ.get(JUDGE_BASE_URL_ENV, "").strip()
    model = os.environ.get(JUDGE_MODEL_ENV, "").strip()
    if not base_url or not model:
        raise RuntimeError(
            f"set {JUDGE_BASE_URL_ENV} and {JUDGE_MODEL_ENV} to an endpoint and model "
            "separate from the system under test. The judge must not be the model that "
            "wrote the questions and generated the answers, and an unnamed model would "
            "leave the run's grades unattributable."
        )
    api_key = os.environ.get(JUDGE_API_KEY_ENV)
    return JudgeBackend(
        llm=ragas_judge_llm(base_url, model, api_key),
        chat=openai_chat_fn(base_url, model, api_key),
        model=model,
        base_url=base_url,
    )


def warm_chat(
    chat: ChatFn, attempts: int = WARM_ATTEMPTS, delay: float = WARM_DELAY_SECONDS
) -> None:
    """Block until the chat backend answers; first calls race model load."""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            chat("ok")
        except Exception as exc:  # retried until the budget runs out
            last = exc
            time.sleep(delay)
        else:
            return
    raise RuntimeError(f"chat backend never came up: {last}")
