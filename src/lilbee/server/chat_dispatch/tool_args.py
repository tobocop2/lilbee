"""Tool-arguments parsing shared by the route layer and the dispatcher."""

from __future__ import annotations

import json
from typing import Any


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Turn an OpenAI tool-call ``arguments`` JSON string into a dict.

    Falls back to ``{"_raw": raw}`` when the model produces malformed JSON or
    a non-object value, so the canonical layer holds invariants the route
    layer would otherwise have to defend.
    """
    if not raw.strip():
        # Whitespace-only counts as empty: json.loads would otherwise turn it
        # into {"_raw": "  "} while "" yields {}.
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        # JSONDecodeError only: raw is a str, so json.loads cannot raise
        # TypeError here, and catching it widened this to hide real bugs.
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}
