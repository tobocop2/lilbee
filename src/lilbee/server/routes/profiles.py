"""Profile routes: list, show, diff, apply, and the profile file operations."""

from __future__ import annotations

from litestar import Response, Router, delete, get, patch, post, put
from litestar.params import FromPath
from litestar.status_codes import HTTP_200_OK

from lilbee.core.profile_files import MAX_PROFILE_BYTES
from lilbee.server.content_disposition import CONTENT_DISPOSITION, attachment_disposition
from lilbee.server.handlers import profiles as handlers
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
    ProfileRenameRequest,
    ProfileSaveRequest,
    ProfileSaveResponse,
    ProfileValidateRequest,
    ProfileValidationResponse,
)

PROFILE_MEDIA_TYPE = "application/toml"
# JSON escapes one byte of a file to at most six characters; the rest covers the other fields
PROFILE_BODY_MAX_BYTES = 6 * MAX_PROFILE_BYTES + 64 * 1024


@get("/api/profiles")
async def profiles_list_route() -> ProfileListResponse:
    """Every profile file, highest precedence first; broken files carry their reason."""
    return await handlers.list_profiles()


@get("/api/profiles/active")
async def profiles_active_route() -> ActiveProfileResponse:
    """The project's applied profile and whether its file is current, changed, gone or broken."""
    return await handlers.active_profile()


@post("/api/profiles", status_code=HTTP_200_OK)
async def profiles_save_route(data: ProfileSaveRequest) -> ProfileSaveResponse:
    """Save the project's settings as a new profile and switch the project to it."""
    return await handlers.save_profile(data)


@post("/api/profiles/discard", status_code=HTTP_200_OK)
async def profiles_discard_route() -> ProfileDiscardResponse:
    """Remove your settings of profile keys so the profile's values show through."""
    return await handlers.discard_changes()


@post("/api/profiles/new", status_code=HTTP_200_OK)
async def profiles_new_route(data: ProfileNewRequest) -> ProfileLocationResponse:
    """Write a template listing every profile setting, optionally from a profile's values."""
    return await handlers.new_profile(data)


@post("/api/profiles/import", status_code=HTTP_200_OK, request_max_body_size=PROFILE_BODY_MAX_BYTES)
async def profiles_import_route(data: ProfileImportRequest) -> ProfileLocationResponse:
    """Validate an uploaded profile file and copy it into a profile folder."""
    return await handlers.import_profile(data)


@post(
    "/api/profiles/validate", status_code=HTTP_200_OK, request_max_body_size=PROFILE_BODY_MAX_BYTES
)
async def profiles_validate_route(data: ProfileValidateRequest) -> ProfileValidationResponse:
    """Every problem with an uploaded profile file."""
    return await handlers.validate_profile(data)


@get("/api/profiles/{name:str}")
async def profiles_show_route(name: FromPath[str]) -> ProfileEntryResponse:
    """The profile *name* picks."""
    return await handlers.show_profile(name)


@get("/api/profiles/{name:str}/diff")
async def profiles_diff_route(name: FromPath[str]) -> ProfileDiffResponse:
    """What applying the profile changes and which values of yours it keeps."""
    return await handlers.diff_profile(name)


@post("/api/profiles/{name:str}/apply", status_code=HTTP_200_OK)
async def profiles_apply_route(name: FromPath[str]) -> ProfileApplyResponse:
    """Make the profile the project's; ``reindex_required`` says whether to rebuild."""
    return await handlers.apply_profile(name)


@put("/api/profiles/{name:str}")
async def profiles_update_route(name: FromPath[str]) -> ProfileSaveResponse:
    """Write the project's settings into its profile, which the name must pick."""
    return await handlers.update_profile(name)


@post("/api/profiles/{name:str}/duplicate", status_code=HTTP_200_OK)
async def profiles_duplicate_route(
    name: FromPath[str], data: ProfileDuplicateRequest
) -> ProfileLocationResponse:
    """Copy the profile under a new name."""
    return await handlers.duplicate_profile(name, data)


@patch("/api/profiles/{name:str}")
async def profiles_rename_route(
    name: FromPath[str], data: ProfileRenameRequest
) -> ProfileLocationResponse:
    """Rename a project or global profile."""
    return await handlers.rename_profile(name, data.new_name)


@delete("/api/profiles/{name:str}", status_code=HTTP_200_OK)
async def profiles_delete_route(name: FromPath[str]) -> ProfileLocationResponse:
    """Remove a project or global profile file; projects keep their recorded copy."""
    return await handlers.delete_profile(name)


@get("/api/profiles/{name:str}/export", media_type=PROFILE_MEDIA_TYPE)
async def profiles_export_route(name: FromPath[str]) -> Response[str]:
    """Download the profile as a clean file."""
    exported = await handlers.export_profile(name)
    return Response(
        content=exported.text,
        media_type=PROFILE_MEDIA_TYPE,
        headers={CONTENT_DISPOSITION: attachment_disposition(exported.filename)},
    )


profiles_router = Router(
    path="/",
    route_handlers=[
        profiles_list_route,
        profiles_active_route,
        profiles_save_route,
        profiles_discard_route,
        profiles_new_route,
        profiles_import_route,
        profiles_validate_route,
        profiles_show_route,
        profiles_diff_route,
        profiles_apply_route,
        profiles_update_route,
        profiles_duplicate_route,
        profiles_rename_route,
        profiles_delete_route,
        profiles_export_route,
    ],
)
