"""Use-case orchestration for long-term chat memory, shared by every surface.

Surfaces (TUI, CLI, MCP, REST, Python API) call these functions rather than
constructing ``MemoryRow`` objects or building owner predicates themselves, so
embedding, id/timestamp assignment, and scoping live in one place.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
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
    local_owner_predicate,
)


def make_memory_row(
    text: str,
    embed: Callable[[str], list[float]],
    *,
    owner: str = LOCAL_OWNER,
    kind: MemoryKind = MemoryKind.FACT,
    source: MemorySource = MemorySource.MANUAL,
    shared: bool = False,
    confirmed: bool = True,
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
        confirmed=confirmed,
        text=text,
        vector=embed(text),
        created_at=now,
        updated_at=now,
    )


MEMORY_DISABLED_HINT = (
    "Memory is off. Enable it in Settings or set memory_enabled=true "
    "(/set memory_enabled true, lilbee set memory_enabled true, or settings_set)."
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
    confirmed: bool = True,
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
        confirmed=confirmed,
    )
    return services.store.add_memory(record)


def recall(query: str, owner: str = LOCAL_OWNER, *, top_k: int | None = None) -> list[MemoryRow]:
    """Recall *owner*'s confirmed facts (plus human-shared facts for agents)."""
    services = get_services()
    predicate = local_owner_predicate() if owner == LOCAL_OWNER else agent_recall_predicate(owner)
    return services.store.search_memories(
        services.embedder.embed(query),
        owner_predicate=predicate,
        top_k=cfg.memory_top_k if top_k is None else top_k,
        max_distance=cfg.memory_max_distance,
    )


def list_memories(owner: str = LOCAL_OWNER) -> list[MemoryRow]:
    """List all of *owner*'s memories (any kind, confirmed or not), newest first."""
    predicate = (
        local_owner_predicate() if owner == LOCAL_OWNER else f"owner = '{escape_sql_string(owner)}'"
    )
    return get_services().store.get_memories(owner_predicate=predicate)


def forget(memory_id: str) -> None:
    """Delete a memory by id."""
    get_services().store.delete_memory(memory_id)


def set_memory_flags(
    memory_id: str, *, shared: bool | None = None, confirmed: bool | None = None
) -> bool:
    """Toggle a memory's shared/confirmed flags; returns True when the id exists."""
    return get_services().store.update_memory(memory_id, shared=shared, confirmed=confirmed)


def auto_extract_enabled() -> bool:
    """True when auto-extraction is on (requires the master gate too)."""
    return cfg.memory_enabled and cfg.memory_auto_extract


def auto_extract(question: str, answer: str) -> list[str]:
    """Extract durable memories from a chat turn and store them unconfirmed.

    Returns the texts stored. Stored memories are ``source=EXTRACTED`` and
    ``confirmed=False`` so they are never recalled until the user confirms them
    in ``/memories``. A no-op (returns ``[]``) unless both the master gate and
    ``memory_auto_extract`` are on.
    """
    from lilbee.retrieval.query.memory_extract import extract_memories

    if not auto_extract_enabled():
        return []
    services = get_services()
    extracted = extract_memories(question, answer, services.provider.chat)
    stored: list[str] = []
    for memory in extracted:
        remember(
            memory.text,
            kind=memory.kind,
            source=MemorySource.EXTRACTED,
            confirmed=False,
        )
        stored.append(memory.text)
    return stored
