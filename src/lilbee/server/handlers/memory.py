"""Memory CRUD handlers shared by the HTTP memory routes.

Each handler delegates to :mod:`lilbee.app.memory` so the embedding, owner
scoping, and store access live in one place. All operate on the human's
``owner=local`` memories; agent-scoped memory is reached through MCP only.
"""

from __future__ import annotations

from litestar.exceptions import NotFoundException

from lilbee.app.memory import (
    MEMORY_DISABLED_HINT,
    forget,
    list_memories,
    memory_enabled,
    remember,
    set_memory_shared,
)
from lilbee.data.store import MemoryKind
from lilbee.server.models import (
    MemoryFlagsResponse,
    MemoryItem,
    MemoryListResponse,
    MemoryRemoveResponse,
    RememberResponse,
)


def _require_memory() -> None:
    """Raise 404 if the memory subsystem is disabled (off by default)."""
    if not memory_enabled():
        raise NotFoundException(detail=MEMORY_DISABLED_HINT)


async def remember_memory(text: str, kind: MemoryKind, shared: bool) -> RememberResponse:
    """Store a memory and return its id and kind."""
    _require_memory()
    memory_id = remember(text, kind=kind, shared=shared)
    return RememberResponse(id=memory_id, kind=kind)


async def list_local_memories() -> MemoryListResponse:
    """Return the human's stored memories, newest first."""
    _require_memory()
    return MemoryListResponse(
        memories=[
            MemoryItem(
                id=m.id,
                kind=m.kind,
                shared=m.shared,
                text=m.text,
            )
            for m in list_memories()
        ]
    )


async def update_memory_shared(memory_id: str, shared: bool) -> MemoryFlagsResponse:
    """Set a memory's shared-with-agents flag."""
    _require_memory()
    updated = set_memory_shared(memory_id, shared=shared)
    return MemoryFlagsResponse(id=memory_id, updated=updated)


async def remove_memory(memory_id: str) -> MemoryRemoveResponse:
    """Delete a memory by id; ``deleted`` is False when the id was unknown."""
    _require_memory()
    deleted = forget(memory_id)
    return MemoryRemoveResponse(id=memory_id, deleted=deleted)
