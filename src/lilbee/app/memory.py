"""Use-case orchestration for long-term chat memory, shared by every surface.

Surfaces (TUI, CLI, MCP, REST, Python API) call these functions rather than
constructing ``MemoryRow`` objects or building owner predicates themselves, so
embedding, id/timestamp assignment, and scoping live in one place.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.data.store import (
    LOCAL_OWNER,
    MemoryKind,
    MemoryRow,
    MemorySource,
    agent_recall_predicate,
    escape_sql_string,
    human_recall_predicate,
)


def make_memory_row(
    text: str,
    embed: Callable[[str], list[float]],
    *,
    owner: str = LOCAL_OWNER,
    kind: MemoryKind = MemoryKind.FACT,
    source: MemorySource = MemorySource.MANUAL,
    shared: bool = False,
) -> MemoryRow:
    """Build a fully populated ``MemoryRow`` with a fresh id, timestamps, and
    an embedded vector. The single id/timestamp/embedding assignment point, so
    callers supply only their own ``embed`` and store the result.
    """
    now = datetime.now(UTC).isoformat()
    return MemoryRow(
        id=uuid.uuid4().hex,
        owner=owner,
        shared=shared,
        kind=kind,
        source=source,
        text=text,
        vector=embed(text),
        created_at=now,
        updated_at=now,
    )


MEMORY_DISABLED_HINT = (
    "Memory is off. Turn it on with /set memory_enabled true in the TUI, "
    "settings_set via MCP, or memory_enabled = true in config.toml."
)


def memory_enabled() -> bool:
    """True when the memory subsystem is switched on (off by default)."""
    return cfg.memory_enabled


def remember(
    text: str,
    *,
    owner: str = LOCAL_OWNER,
    kind: MemoryKind = MemoryKind.FACT,
    source: MemorySource = MemorySource.MANUAL,
    shared: bool = False,
) -> str:
    """Embed *text* and store it as a memory; returns the stored id."""
    services = get_services()
    record = make_memory_row(
        text,
        services.embedder.embed,
        owner=owner,
        kind=kind,
        source=source,
        shared=shared,
    )
    return services.store.add_memory(record)


def recall(query: str, owner: str = LOCAL_OWNER, *, top_k: int | None = None) -> list[MemoryRow]:
    """Recall *owner*'s facts (plus human-shared facts for agents)."""
    services = get_services()
    predicate = human_recall_predicate() if owner == LOCAL_OWNER else agent_recall_predicate(owner)
    return services.store.search_memories(
        services.embedder.embed_query(query),
        owner_predicate=predicate,
        top_k=cfg.memory_top_k if top_k is None else top_k,
        max_distance=cfg.memory_max_distance,
    )


def list_memories(owner: str = LOCAL_OWNER) -> list[MemoryRow]:
    """List all of *owner*'s memories (any kind), newest first."""
    predicate = (
        human_recall_predicate() if owner == LOCAL_OWNER else f"owner = '{escape_sql_string(owner)}'"
    )
    return get_services().store.get_memories(owner_predicate=predicate)


def forget(memory_id: str, *, owner: str = LOCAL_OWNER) -> bool:
    """Delete *owner*'s memory by id; returns True when it existed and was owned.

    Defaults to the local human's namespace (TUI/CLI/REST/Python API); MCP passes
    the calling agent's owner so an agent can only delete its own memories.
    """
    return get_services().store.delete_memory(memory_id, owner=owner)


def set_memory_shared(memory_id: str, *, shared: bool, owner: str = LOCAL_OWNER) -> bool:
    """Set *owner*'s memory shared-with-agents flag; returns True when found and owned."""
    return get_services().store.update_memory(memory_id, shared=shared, owner=owner)


def auto_extract_enabled() -> bool:
    """True when auto-extraction is on (requires the master gate too)."""
    return cfg.memory_enabled and cfg.memory_auto_extract


@dataclass(frozen=True, slots=True)
class SavedMemory:
    """A memory created by auto-extraction: its stored id, kind, and text."""

    id: str
    kind: MemoryKind
    text: str


def auto_extract(question: str, answer: str) -> list[SavedMemory]:
    """Extract durable memories from a chat turn and store them.

    Returns one :class:`SavedMemory` per stored memory. Stored memories are
    ``source=EXTRACTED`` and are recalled like any other; the user manages them
    in ``/memories``. A no-op (returns ``[]``) unless both the master gate and
    ``memory_auto_extract`` are on.
    """
    from lilbee.retrieval.query.memory_extract import extract_memories

    if not auto_extract_enabled():
        return []
    services = get_services()

    def _chat_text(messages: list[dict[str, str]], **_kwargs: object) -> str:
        return services.provider.chat(messages, stream=False).text

    extracted = extract_memories(question, answer, _chat_text)
    stored: list[SavedMemory] = []
    for memory in extracted:
        memory_id = remember(memory.text, kind=memory.kind, source=MemorySource.EXTRACTED)
        stored.append(SavedMemory(id=memory_id, kind=memory.kind, text=memory.text))
    return stored
