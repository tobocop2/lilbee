"""The requests that every engine and its oracle answer, and the version of that set."""

from __future__ import annotations

from typing import Any

REQUEST_SET_VERSION = 1
"""The version of the requests every record answers; a change to a request bumps it."""

CONVERSATION: list[dict[str, str]] = [
    {"role": "system", "content": "You answer in one short sentence."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Italy?"},
]
TEMPLATE_FACT = "rome"
"""The word that the answer to CONVERSATION holds."""
WEATHER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TOOL_ASK = "What is the weather in Paris? Use the tool."
TOOL_NAME = "get_weather"
REASONING_ASK: dict[str, str] = {
    "role": "user",
    "content": "Think it through, then answer: what is 17 plus 25?",
}
RERANK_QUERY = "What is the capital of France?"
RERANK_DOCS = ["Bananas are yellow.", "Paris is the capital of France.", "Dogs bark."]
RERANK_BEST = 1
"""The index in RERANK_DOCS of the document that answers the query."""
EMBED_TEXTS = ["alpha", "beta gamma"]
JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
    "required": ["city", "pop"],
    "additionalProperties": False,
}
JSON_ASK = "Reply with a JSON object for Paris: city is the name and pop is the population."
OVERSIZE = "w " * 60000
VISION_PROMPT = "What colour is this image? Answer in one word."
"""The vision request asks for one word, so the verdict can read the answer as the word."""
