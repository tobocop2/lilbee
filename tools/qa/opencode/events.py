"""Reading the JSONL event stream the per-cell opencode plugin appends.

The plugin (qa_events_plugin.js, deployed into each cell workspace's
``.opencode/plugins/``) taps opencode's event bus, so the harness gets real
signals -- tool dispatched, session idle, session error -- instead of
inferring them from pane text. Pane scraping remains as the verdict
artifact (what the user saw) and as a fallback when the plugin fails to
load on an older opencode.
"""

from __future__ import annotations

import json
from pathlib import Path

_EVENTS_FILENAME = "qa-events.jsonl"
_TOOL_EVENT_TYPES = ("qa.tool.after", "tool.execute.after")
_SESSION_ERROR_TYPE = "session.error"
_SESSION_IDLE_TYPE = "session.idle"


def events_path(workspace: Path) -> Path:
    return workspace / ".lilbee" / _EVENTS_FILENAME


def read_events(workspace: Path) -> list[dict]:
    """Parse the event stream; unreadable lines are skipped, absence is empty."""
    path = events_path(workspace)
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def plugin_active(workspace: Path) -> bool:
    """True once the event tap has written anything (the plugin loaded)."""
    return bool(read_events(workspace))


def count_tool_dispatches(events: list[dict], tool_substr: str) -> int:
    """Completed tool executions whose tool name contains *tool_substr*."""
    return sum(
        1
        for e in events
        if e.get("type") in _TOOL_EVENT_TYPES and tool_substr in str(e.get("tool", ""))
    )


def count_session_idles(events: list[dict]) -> int:
    return sum(1 for e in events if e.get("type") == _SESSION_IDLE_TYPE)


def has_session_error(events: list[dict]) -> bool:
    return any(e.get("type") == _SESSION_ERROR_TYPE for e in events)
