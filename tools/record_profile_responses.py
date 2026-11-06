#!/usr/bin/env python3
"""Record the server's real /api/profiles responses, error bodies included.

The release attaches the result as ``profile-responses.json`` beside
``openapi.json``. The schema says which fields a response carries; a client
also reads what those fields hold (the diff rows, the kept keys, the folder of
each profile) and the ``detail`` of a refusal, so its tests replay these bodies.

It builds the app in-process against a throwaway home and data root, so it
never touches a real profile folder, and drives one fixed scenario through it.
The home and data root are set before lilbee is imported, because the config
reads them at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

_DEFAULT_OUT = Path("profile-responses.json")

_CREDITED = (
    '[profile]\nname = "Court filings"\ndescription = "Scanned US court PDFs with dense tables."\n'
    'authors = [{ name = "Jane Doe", github = "janedoe" }]\n'
    'tested_on = "4,000 scanned county court filings, 1990-2010"\n'
    "[values]\ntable_extraction = true\nchunk_size = 768\n"
)
_INVALID = "[values]\nchat_model = 'x'\n"
_TOO_LARGE = "#" * (256 * 1024 + 1)


def _isolate(scratch: Path) -> Path:
    """Point every home and data variable lilbee reads at ``scratch``; return the project root."""
    home = scratch / "home"
    project = scratch / "project"
    os.environ.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_DATA_HOME": str(home / "xdg"),
            "LOCALAPPDATA": str(home / "local"),
            "LILBEE_DATA": str(project),
        }
    )
    return project


# The body of the ``export`` step, sent back by the validate and import steps.
_EXPORTED = object()


def _url(name: str, suffix: str = "") -> str:
    return f"/api/profiles/{quote(name, safe='')}{suffix}"


def _scenario() -> list[tuple[str | None, str, str, Any]]:
    """The fixed steps as (label, method, path, JSON body); an unlabelled step is not recorded."""
    exported = {"content": _EXPORTED, "filename": "old-scans.toml"}
    duplicate = {"new_name": "Scans", "target": "global"}
    return [
        (None, "PATCH", "/api/config", {"chunk_size": 500}),
        ("list", "GET", "/api/profiles", None),
        ("active_default", "GET", "/api/profiles/active", None),
        ("show", "GET", _url("Court filings"), None),
        ("diff", "GET", _url("Notes and markdown", "/diff"), None),
        ("apply", "POST", _url("Notes and markdown", "/apply"), None),
        ("active_applied", "GET", "/api/profiles/active", None),
        ("diff_active", "GET", _url("Notes and markdown", "/diff"), None),
        ("save_as", "POST", "/api/profiles", {"name": "My notes", "target": "project"}),
        (None, "PATCH", "/api/config", {"chunk_overlap": 80}),
        ("active_saved", "GET", "/api/profiles/active", None),
        ("diff_saved", "GET", _url("My notes", "/diff"), None),
        ("update", "PUT", _url("My notes"), None),
        (None, "PATCH", "/api/config", {"chunk_overlap": 90}),
        ("discard", "POST", "/api/profiles/discard", None),
        ("duplicate", "POST", _url("Scanned archive", "/duplicate"), duplicate),
        ("rename", "PATCH", _url("Scans"), {"new_name": "Old scans"}),
        ("export", "GET", _url("Old scans", "/export"), None),
        ("delete", "DELETE", _url("Old scans"), None),
        ("validate_ok", "POST", "/api/profiles/validate", exported),
        (
            "validate_bad",
            "POST",
            "/api/profiles/validate",
            {"content": _INVALID, "filename": "bad.toml"},
        ),
        ("import", "POST", "/api/profiles/import", exported),
        ("error_missing", "GET", _url("No such profile", "/diff"), None),
        ("error_clash", "POST", _url("Scanned archive", "/duplicate"), {"new_name": "Old scans"}),
        ("error_builtin", "PATCH", _url("Default"), {"new_name": "Mine"}),
        ("error_reserved", "POST", "/api/profiles", {"name": "active"}),
        (
            "error_too_large",
            "POST",
            "/api/profiles/import",
            {"content": _TOO_LARGE, "filename": "big.toml"},
        ),
        ("error_no_route", "GET", "/api/no-such-route", None),
    ]


def _prepare(project: Path) -> None:
    """Point the config at ``project``, stub the engine, and write the scenario's own profiles."""
    from unittest.mock import MagicMock

    import lilbee.app.services as services
    import lilbee.server.auth as auth
    from lilbee.core.config import cfg
    from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileStore
    from lilbee.core.system import default_data_dir

    cfg.data_root = project
    cfg.data_dir = project / "data"
    cfg.documents_dir = project / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    # No engine: a stand-in container, which a settings write must not tear down.
    services.set_services(MagicMock(profile_store=ProfileStore()))
    services.reset_services = lambda: None  # type: ignore[assignment]
    auth.session_manager.disable()

    for folder in (default_data_dir() / PROFILES_DIRNAME, project / PROFILES_DIRNAME):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "court-filings.toml").write_text(_CREDITED, encoding="utf-8")
    (default_data_dir() / PROFILES_DIRNAME / "broken.toml").write_text(_INVALID, encoding="utf-8")


def record(project: Path) -> dict[str, Any]:
    """Drive the profile scenario through the app and return each response by label."""
    from litestar.testing import TestClient

    from lilbee.core.profile_files import PACKAGE_PROFILES_DIR
    from lilbee.core.system import default_data_dir
    from lilbee.server.app import create_app

    _prepare(project)
    client = TestClient(create_app())
    recorded: dict[str, Any] = {}
    for label, method, path, body in _scenario():
        if body is not None and body.get("content") is _EXPORTED:
            body = {**body, "content": recorded["export"]["body"]}
        resp = client.request(method, path, json=body)
        if label is None:
            continue
        is_json = resp.headers.get("content-type", "").startswith("application/json")
        entry = {"status": resp.status_code, "body": resp.json() if is_json else resp.text}
        disposition = resp.headers.get("content-disposition")
        if disposition is not None:
            entry["content_disposition"] = disposition
        recorded[label] = entry

    roots = {
        str(PACKAGE_PROFILES_DIR): "<package>",
        str(project): "<project>",
        str(default_data_dir()): "<global>",
    }
    scrubbed: dict[str, Any] = _scrub(recorded, roots)
    return scrubbed


def _scrub(value: Any, roots: dict[str, str]) -> Any:
    """Swap each throwaway root inside every string for its placeholder.

    It works on the string values, not on the serialized JSON, where a Windows
    path has its backslashes escaped and no longer matches its root. On Windows
    the server writes some paths with ``/`` and some with ``\\``, so each root
    matches in both forms. A path that starts with a root gets forward slashes,
    so the recording reads the same on every platform.
    """
    if isinstance(value, dict):
        return {key: _scrub(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, roots) for item in value]
    if not isinstance(value, str):
        return value
    forms = {form: name for root, name in roots.items() for form in (root, root.replace("\\", "/"))}
    for root, name in sorted(forms.items(), key=lambda pair: -len(pair[0])):
        if value.startswith(root):
            return name + value[len(root) :].replace("\\", "/")
        value = value.replace(root, name)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="lilbee-profile-responses-") as scratch:
        # Resolved, because the server answers with resolved paths, and on macOS the
        # temp dir is a symlink: an unresolved root would leave /private in the output.
        recorded = record(_isolate(Path(scratch).resolve()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
