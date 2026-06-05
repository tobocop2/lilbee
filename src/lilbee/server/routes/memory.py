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
    update_memory_flags,
)
from lilbee.server.models import (
    MemoryFlagsRequest,
    MemoryFlagsResponse,
    MemoryListResponse,
    MemoryRemoveResponse,
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
async def memories_update_route(memory_id: str, data: MemoryFlagsRequest) -> MemoryFlagsResponse:
    """Toggle a memory's shared/confirmed flags."""
    return await update_memory_flags(memory_id, data.shared, data.confirmed)


@delete("/api/memories/{memory_id:str}", status_code=200)
async def memories_remove_route(memory_id: str) -> MemoryRemoveResponse:
    """Delete a memory by id."""
    return await remove_memory(memory_id)
