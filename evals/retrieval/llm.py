"""Chat backends for question authoring and judging."""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import httpx

ChatFn = Callable[[str], str]

JUDGE_BASE_URL_ENV = "LILBEE_EVAL_JUDGE_BASE_URL"
JUDGE_MODEL_ENV = "LILBEE_EVAL_JUDGE_MODEL"
JUDGE_API_KEY_ENV = "LILBEE_EVAL_JUDGE_API_KEY"

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


def judge_chat_fn() -> ChatFn:
    """The judge backend: env-configured OpenAI-compatible endpoint, else lilbee."""
    base_url = os.environ.get(JUDGE_BASE_URL_ENV)
    if base_url:
        model = os.environ.get(JUDGE_MODEL_ENV, "")
        return openai_chat_fn(base_url, model, os.environ.get(JUDGE_API_KEY_ENV))
    return lilbee_chat_fn()


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
