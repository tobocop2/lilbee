"""Memory route handlers: list, remember, update flags, forget.

``GET`` is ``@read_only`` so a read-only session token can list memories;
the mutating routes are unmarked so read-only tokens cannot write memory.
"""

from __future__ import annotations

from litestar import delete, get, patch, post

from lilbee.server.auth import read_only
from lilbee.server.handlers.memory import (
    list_local_memories,
    remember_memory,
    remove_memory,
    update_memory_shared,
)
from lilbee.server.models import (
    MemoryFlagsResponse,
    MemoryListResponse,
    MemoryRemoveResponse,
    MemorySharedRequest,
    RememberRequest,
    RememberResponse,
)


@get("/api/memories")
@read_only
async def memories_list_route() -> MemoryListResponse:
    """List the human's stored memories."""
    return await list_local_memories()


@post("/api/memories")
async def memories_remember_route(data: RememberRequest) -> RememberResponse:
    """Store a fact or preference in the human's memory."""
    return await remember_memory(data.text, data.kind, data.shared)


@patch("/api/memories/{memory_id:str}")
async def memories_update_route(memory_id: str, data: MemorySharedRequest) -> MemoryFlagsResponse:
    """Set a memory's shared-with-agents flag."""
    return await update_memory_shared(memory_id, data.shared)


@delete("/api/memories/{memory_id:str}", status_code=200)
async def memories_remove_route(memory_id: str) -> MemoryRemoveResponse:
    """Delete a memory by id."""
    return await remove_memory(memory_id)
