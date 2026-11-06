"""Profile route handlers: each runs one app.profiles operation off the event loop.

A profile name from a URL only ever reaches the profile lookup, never a file path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import TypeVar

from litestar.exceptions import (
    ClientException,
    NotFoundException,
    ServiceUnavailableException,
    ValidationException,
)
from litestar.status_codes import HTTP_409_CONFLICT, HTTP_413_REQUEST_ENTITY_TOO_LARGE

from lilbee.app import profiles
from lilbee.app.profiles import ExportedProfile, ProfileNotFoundError, file_failure_message
from lilbee.app.services import get_services
from lilbee.core.profile_files import ProfileNameClashError, ProfileStore
from lilbee.server.handlers.ingest import sync_stream
from lilbee.server.handlers.sse import sse_done
from lilbee.server.models import (
    ActiveProfileResponse,
    ProfileApplyResponse,
    ProfileDiffResponse,
    ProfileDiscardResponse,
    ProfileDuplicateRequest,
    ProfileEntryResponse,
    ProfileImportRequest,
    ProfileListResponse,
    ProfileLocationResponse,
    ProfileNewRequest,
    ProfileSaveRequest,
    ProfileSaveResponse,
    ProfileValidateRequest,
    ProfileValidationResponse,
)

T = TypeVar("T")


class ProfileConflictException(ClientException):
    """A profile file of that name already exists in the target folder."""

    status_code = HTTP_409_CONFLICT


class ProfileTooLargeException(ClientException):
    """An upload over the route's body cap. Litestar answers it before the handler runs."""

    status_code = HTTP_413_REQUEST_ENTITY_TOO_LARGE


# What each kind of profile route can answer besides success. The routes list
# these as ``raises``, so the schema documents every error ``_run`` maps to.
FILE_ERRORS = [ServiceUnavailableException]
REFUSAL_ERRORS = [ValidationException, ServiceUnavailableException]
UPLOAD_ERRORS = [ProfileTooLargeException]
NAMED_ERRORS = [NotFoundException, ValidationException, ServiceUnavailableException]
WRITE_ERRORS = [ProfileConflictException, ValidationException, ServiceUnavailableException]
NAMED_WRITE_ERRORS = [NotFoundException, *WRITE_ERRORS]


def _store() -> ProfileStore:
    return get_services().profile_store


async def _run(operation: Callable[[], T]) -> T:
    """Run *operation* on a thread: missing is a 404, a name clash 409, other refusals 400."""
    try:
        return await asyncio.to_thread(operation)
    except ProfileNotFoundError as exc:
        raise NotFoundException(detail=str(exc)) from exc
    except ProfileNameClashError as exc:
        raise ProfileConflictException(detail=str(exc)) from exc
    except ValueError as exc:
        raise ValidationException(detail=str(exc)) from exc
    except OSError as exc:
        raise ServiceUnavailableException(detail=file_failure_message(exc)) from exc


async def list_profiles() -> ProfileListResponse:
    """Every profile file, Default first, then the other built-ins, then the rest."""
    catalog = await _run(lambda: profiles.list_profiles(_store()))
    return ProfileListResponse(
        profiles=[ProfileEntryResponse.from_entry(e) for e in catalog.entries]
    )


async def active_profile() -> ActiveProfileResponse:
    """The project's applied profile and the state of its file."""
    return ActiveProfileResponse.from_active(await _run(lambda: profiles.active(_store())))


async def show_profile(name: str) -> ProfileEntryResponse:
    """The profile *name* picks."""
    return ProfileEntryResponse.from_entry(await _run(lambda: profiles.show(_store(), name)))


async def diff_profile(name: str) -> ProfileDiffResponse:
    """What applying *name* changes."""
    return ProfileDiffResponse.from_diff(await _run(lambda: profiles.diff(_store(), name)))


async def apply_profile(name: str) -> ProfileApplyResponse:
    """Make *name* the project's profile."""
    return ProfileApplyResponse.from_result(await _run(lambda: profiles.apply(_store(), name)))


async def rebuild_after_apply_stream(applied: ProfileApplyResponse) -> AsyncGenerator[str, None]:
    """The rebuild an apply needs as sync SSE events; done carries the apply and ``rebuild``."""
    needed = await asyncio.to_thread(
        profiles.apply_needs_rebuild, _store(), applied.name, applied.reindex_required
    )
    async for frame in _rebuild_stream(applied.model_dump(mode="json"), needed):
        yield frame


async def rebuild_after_discard_stream(
    discarded: ProfileDiscardResponse,
) -> AsyncGenerator[str, None]:
    """The rebuild a discard needs as sync SSE events; done carries the discard and ``rebuild``."""
    reason = await asyncio.to_thread(
        profiles.rebuild_reason, discarded.dropped, discarded.reindex_required
    )
    async for frame in _rebuild_stream(discarded.model_dump(mode="json"), reason is not None):
        yield frame


async def _rebuild_stream(payload: dict[str, object], needed: bool) -> AsyncGenerator[str, None]:
    """A forced sync's SSE events when *needed*; done carries *payload* and ``rebuild``."""
    if not needed:
        yield sse_done({**payload, "rebuild": None})
        return
    async for frame in sync_stream(
        force_rebuild=True,
        done_payload=lambda result: {**payload, "rebuild": result.model_dump()},
    ):
        yield frame


async def save_profile(data: ProfileSaveRequest) -> ProfileSaveResponse:
    """Save the project's settings as a new profile and switch to it."""
    return ProfileSaveResponse.from_save(
        await _run(lambda: profiles.save_as(data.name, data.target))
    )


async def update_profile(name: str) -> ProfileSaveResponse:
    """Write the project's settings into its profile, which *name* must pick."""
    return ProfileSaveResponse.from_save(await _run(lambda: profiles.update(_store(), name)))


async def discard_changes() -> ProfileDiscardResponse:
    """Remove your settings of profile keys so the profile's values show through."""
    return ProfileDiscardResponse.from_result(await _run(profiles.discard))


async def new_profile(data: ProfileNewRequest) -> ProfileLocationResponse:
    """Write a template profile, optionally from another profile's values."""
    location = await _run(
        lambda: profiles.new(_store(), data.name, data.target, from_name=data.from_profile)
    )
    return ProfileLocationResponse.from_location(location)


async def duplicate_profile(name: str, data: ProfileDuplicateRequest) -> ProfileLocationResponse:
    """Copy *name* under a new name."""
    location = await _run(lambda: profiles.duplicate(_store(), name, data.new_name, data.target))
    return ProfileLocationResponse.from_location(location)


async def rename_profile(name: str, new_name: str) -> ProfileLocationResponse:
    """Rename a project or global profile."""
    location = await _run(lambda: profiles.rename(_store(), name, new_name))
    return ProfileLocationResponse.from_location(location)


async def delete_profile(name: str) -> ProfileLocationResponse:
    """Remove a project or global profile file."""
    return ProfileLocationResponse.from_location(
        await _run(lambda: profiles.delete(_store(), name))
    )


async def export_profile(name: str) -> ExportedProfile:
    """The profile *name* picks as the text of a clean file."""
    return await _run(lambda: profiles.export_text(_store(), name))


async def import_profile(data: ProfileImportRequest) -> ProfileLocationResponse:
    """Validate an uploaded profile file and copy it into a profile folder."""
    location = await _run(
        lambda: profiles.import_text(
            _store(), data.content, data.filename, data.target, overwrite=data.overwrite
        )
    )
    return ProfileLocationResponse.from_location(location)


async def validate_profile(data: ProfileValidateRequest) -> ProfileValidationResponse:
    """Every problem with an uploaded profile file."""
    result = await _run(lambda: profiles.validate_content(data.content, data.filename, data.folder))
    return ProfileValidationResponse.from_validation(result)
