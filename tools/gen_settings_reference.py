#!/usr/bin/env python3
"""Generate ``docs/settings.md``, the cross-surface settings reference.

Every row comes from ``Config.model_fields``, so a new setting appears in the
reference as soon as it exists. The surface columns are derived the same way
the surfaces themselves derive them: ``config_meta`` for the writable and
public sets, ``SETTINGS_MAP`` for the TUI rows, ``PROVIDER_SWITCHING_KEYS`` for
the HTTP refusal, ``mcp_server.TOOL_GATE_SETTINGS`` for the flags that gate
tool registration. Nothing here restates a rule that lives in the source.

Help text has one home per field: the ``SettingDef`` for settings the TUI
renders, and the pydantic field description for the rest. A field carrying
neither fails the run, so a setting cannot land undocumented.

Variables read straight from ``os.environ`` are not Config fields and so are
invisible to that generation. They live in ``ENV_ONLY`` here, and
``_check_env_only_registry`` fails the run when ``src/`` reads one that is
listed in neither place.

Run ``make docs-settings`` to regenerate; CI checks the committed file matches.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lilbee.app.settings import _annotation_name, _setting_default  # noqa: E402
from lilbee.app.settings_map import SETTINGS_MAP, SettingGroup  # noqa: E402
from lilbee.config_meta import (  # noqa: E402
    PUBLIC_CONFIG_FIELDS,
    REINDEX_FIELDS,
    WRITABLE_CONFIG_FIELDS,
)
from lilbee.core.config import Config  # noqa: E402
from lilbee.core.config.keys import PROVIDER_SWITCHING_KEYS  # noqa: E402
from lilbee.mcp_server import TOOL_GATE_SETTINGS  # noqa: E402
from lilbee.providers.roles import MODEL_ROLE_FIELDS  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "settings.md"

# Order the groups from most-often-touched to rarely-touched, the same shape
# the TUI settings screen presents. Groups absent from SETTINGS_MAP are skipped.
GROUP_ORDER: tuple[SettingGroup, ...] = (
    SettingGroup.MODELS,
    SettingGroup.RETRIEVAL,
    SettingGroup.GENERATION,
    SettingGroup.INGEST,
    SettingGroup.WIKI,
    SettingGroup.MEMORY,
    SettingGroup.CRAWLING,
    SettingGroup.LOCAL_SERVERS,
    SettingGroup.API_KEYS,
    SettingGroup.DISPLAY,
    SettingGroup.SYSTEM,
    SettingGroup.GENERAL,
)

# The only CLI commands that write a setting to config.toml. Everything else
# reaches the CLI through the environment or the config file, so the column
# stays honest rather than implying a `lilbee set` that does not exist.
CLI_COMMANDS: dict[str, str] = {
    "embedding_model": "`lilbee use-embedder REF`",
    "placement": "`lilbee placement set`",
}

MAX_DEFAULT_CHARS = 48

# Environment variables that are NOT Config fields. The code reads them straight
# from os.environ, so nothing in Config.model_fields can find them and the tables
# above cannot cover them. Each entry is (description, internal). Internal ones
# are registered so the scan below stays green, but stay out of the document:
# they are test hooks and process plumbing, not settings anyone should set.
ENV_ONLY: dict[str, tuple[str, bool]] = {
    "LILBEE_DATA": (
        "Data directory for this library, the same value as `data_root`. The "
        "older and more common spelling; `--data-dir` overrides it",
        False,
    ),
    "LILBEE_LOG_LEVEL": (
        "Logging level: DEBUG, INFO, WARNING, or ERROR. `--log-level` overrides it",
        False,
    ),
    "LILBEE_ENGINE_DIR": (
        "Directory holding the llama-server engine binaries. Set it to run against "
        "an engine build other than the bundled one",
        False,
    ),
    "LILBEE_TOKEN": (
        "Auth token for the HTTP server. The launchers set it to the live session "
        "token so no literal token is written to a config file on disk",
        False,
    ),
    "LILBEE_AGENT_ID": (
        "Owner namespace for an MCP agent's memories and sessions. An explicit "
        "`agent_id` tool argument wins over it",
        False,
    ),
    "LILBEE_CPU_QUOTA": (
        "CPU concurrency cap. Defaults to half the available cores; a "
        "non-positive or unparseable value falls back to that default",
        False,
    ),
    "LILBEE_NO_SPLASH": ("Set to any value to suppress the startup splash animation", False),
    "LILBEE_INGEST_CONCURRENCY": (
        "Extraction-admission mode: `static` (the default), "
        "`adaptive-conservative`, or `adaptive-aggressive`",
        False,
    ),
    "LILBEE_INGEST_MAX_WORKERS": ("Files per embed replica kept in flight during ingest", False),
    "LILBEE_INGEST_TRACE": ("Set to any value to trace each document through ingest", False),
    "LILBEE_INGEST_TRACE_FILE": ("File the ingest trace is written to", False),
    "LILBEE_EXCLUSIVE_SCOPE": (
        "A directory that at most one server may serve at a time, for a plugin's shared root",
        False,
    ),
    "LILBEE_OCR_FORCE": (
        "Force vision OCR on pages that already carry a text layer. It has no "
        "effect on those pages today; set `vlm_fallback` instead",
        False,
    ),
    # Internal: not settings, and documenting them would invite misuse.
    "LILBEE_PARENT_PID": ("Process plumbing: the parent to watch and exit with", True),
    "LILBEE_LAUNCHER_SERVE_QUIET": ("Internal launcher flag", True),
    "LILBEE_SKIP_TOML_CONFIG": ("Test hook: ignore config.toml for hermetic runs", True),
    "LILBEE_SKIP_MODEL_TASK_VALIDATION": ("Test hook: skip catalog task validation", True),
}

ENV_LITERAL = re.compile(r"""["'](LILBEE_[A-Z0-9_]+)["']""")

# Settings whose default is computed from the host rather than fixed. Rendering
# the number would bake one machine's answer into a committed file, so the
# document names the rule instead and the help text carries the detail.
# tests/test_settings_reference.py renders under two simulated hosts and fails
# if any other row moves, which is what keeps this list complete.
HOST_SCALED: dict[str, str] = {
    "chat_n_ctx_target": "*(scales with host RAM)*",
}


def _env_var(key: str) -> str:
    return f"LILBEE_{key.upper()}"


def _help_text(key: str) -> str:
    """Return the one documented description for *key*.

    ``SettingDef.help_text`` wins because it is what the TUI already shows;
    fields with no ``SettingDef`` fall back to the pydantic description.
    """
    definition = SETTINGS_MAP.get(key)
    if definition is not None and definition.help_text:
        return definition.help_text
    description = Config.model_fields[key].description
    return description or ""


def _render_text_default(value: str) -> str:
    if not value:
        return "*(empty)*"
    return f"`{value}`" if len(value) <= MAX_DEFAULT_CHARS else "*(built-in text)*"


def _render_collection_default(
    value: frozenset[Any] | set[Any] | list[Any] | tuple[Any, ...],
) -> str:
    if not value:
        return "*(empty)*"
    rendered = ", ".join(sorted(str(item) for item in value))
    return f"`{rendered}`" if len(rendered) <= MAX_DEFAULT_CHARS else "*(built-in list)*"


def _render_default(key: str) -> str:
    """Render a field's default as a short, table-safe cell."""
    if key in HOST_SCALED:
        return HOST_SCALED[key]
    value: Any = _setting_default(key)
    if value is None:
        rendered = "*(none)*"
    elif isinstance(value, bool):
        rendered = f"`{str(value).lower()}`"
    elif isinstance(value, str):
        rendered = _render_text_default(value)
    elif isinstance(value, Path):
        # The path fields default to an unresolved sentinel; the real value is
        # computed at process start from the platform data directory.
        rendered = "*(computed)*" if str(value) in {".", ""} else f"`{value}`"
    elif isinstance(value, frozenset | set | list | tuple):
        rendered = _render_collection_default(value)
    elif isinstance(value, dict):
        rendered = "*(empty)*" if not value else "*(built-in mapping)*"
    else:
        rendered = f"`{value}`"
    return rendered


def _tui_cell(key: str) -> str:
    definition = SETTINGS_MAP.get(key)
    if definition is None or definition.hidden:
        return "no"
    if key in MODEL_ROLE_FIELDS:
        return "picker"
    if not definition.writable:
        return "read-only"
    return "yes"


def _mcp_cell(key: str) -> str:
    if key in MODEL_ROLE_FIELDS:
        return "yes"
    if key not in WRITABLE_CONFIG_FIELDS:
        return "no"
    return "yes" if key in PUBLIC_CONFIG_FIELDS else "write-only"


def _http_cell(key: str) -> str:
    if key in MODEL_ROLE_FIELDS:
        return "role API"
    if key not in WRITABLE_CONFIG_FIELDS:
        return "no"
    if key in PROVIDER_SWITCHING_KEYS:
        return "refused"
    return "yes" if key in PUBLIC_CONFIG_FIELDS else "write-only"


def _cli_cell(key: str) -> str:
    return CLI_COMMANDS.get(key, "no")


def _notes(key: str) -> list[str]:
    """Per-setting warnings that change what a caller must do after writing."""
    notes: list[str] = []
    if key in TOOL_GATE_SETTINGS:
        notes.append("**Reconnect MCP** for the tool list to change.")
    if key in REINDEX_FIELDS:
        notes.append("**Reindex** with `lilbee rebuild` after changing.")
    if key in PROVIDER_SWITCHING_KEYS:
        notes.append("Refused over HTTP and over MCP mounted on the HTTP server.")
    return notes


def _description_cell(key: str) -> str:
    parts = [_help_text(key).strip().rstrip(".") + "."]
    definition = SETTINGS_MAP.get(key)
    if definition is not None and definition.choices:
        parts.append("One of " + ", ".join(f"`{c}`" for c in definition.choices) + ".")
    parts.extend(_notes(key))
    return " ".join(parts).replace("|", "\\|").replace("\n", " ")


def _row(key: str) -> str:
    cells = (
        f"`{key}`",
        f"`{_env_var(key)}`",
        f"`{_annotation_name(Config.model_fields[key].annotation)}`",
        _render_default(key),
        _tui_cell(key),
        _mcp_cell(key),
        _http_cell(key),
        _cli_cell(key),
        _description_cell(key),
    )
    return "| " + " | ".join(cells) + " |"


HEADER = (
    "| Setting | Environment | Type | Default | TUI | MCP | HTTP | CLI | Description |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def _table(keys: list[str]) -> str:
    return "\n".join([HEADER, *(_row(key) for key in sorted(keys))])


def _check_help_coverage(keys: list[str]) -> None:
    missing = sorted(key for key in keys if not _help_text(key))
    if missing:
        raise SystemExit(
            "These settings have no help text. Add a `help_text` to their SettingDef in "
            "src/lilbee/app/settings_map.py, or a `description=` to the field in "
            "src/lilbee/core/config/model.py:\n  " + "\n  ".join(missing)
        )


def _check_env_only_registry() -> None:
    """Fail when src/ reads a LILBEE_* variable this file does not know about.

    The generator can only see settings that are Config fields. Anything read
    straight from os.environ is invisible to it, so without this scan a new
    environment-only variable would silently never be documented, which is the
    gap that let LILBEE_DATA and LILBEE_LOG_LEVEL go missing.
    """
    known = {f"LILBEE_{key.upper()}" for key in Config.model_fields} | set(ENV_ONLY)
    found: dict[str, str] = {}
    for path in sorted((REPO_ROOT / "src" / "lilbee").rglob("*.py")):
        for match in ENV_LITERAL.finditer(path.read_text(encoding="utf-8")):
            found.setdefault(match.group(1), path.relative_to(REPO_ROOT).as_posix())
    unknown = sorted(name for name in found if name not in known)
    if unknown:
        raise SystemExit(
            "These environment variables are read in src/ but are neither a Config "
            "field nor listed in ENV_ONLY in tools/gen_settings_reference.py:\n"
            + "\n".join(f"  {name}  ({found[name]})" for name in unknown)
        )


def _env_only_table() -> str:
    rows = [
        f"| `{name}` | {description} |"
        for name, (description, internal) in sorted(ENV_ONLY.items())
        if not internal
    ]
    header = "| Variable | Description |\n|---|---|"
    return "\n".join([header, *rows])


TEMPLATE = REPO_ROOT / "tools" / "templates" / "settings_reference.md"
GROUP_MARKER = "<!-- GROUP TABLES -->"
ENV_ONLY_MARKER = "<!-- ENV ONLY TABLE -->"
NOT_A_SETTING_MARKER = "<!-- ENV ONLY VARIABLES -->"


def render() -> str:
    """Build the reference: hand-written prose from the template, tables from Config."""
    all_keys = list(Config.model_fields)
    _check_help_coverage(all_keys)
    _check_env_only_registry()

    grouped: dict[SettingGroup, list[str]] = {group: [] for group in GROUP_ORDER}
    env_only: list[str] = []
    for key in all_keys:
        definition = SETTINGS_MAP.get(key)
        if definition is None:
            env_only.append(key)
        else:
            grouped[definition.group].append(key)

    sections = [
        f"## {group.value}\n\n{_table(keys)}" for group in GROUP_ORDER if (keys := grouped[group])
    ]
    document = TEMPLATE.read_text(encoding="utf-8")
    document = document.replace(GROUP_MARKER, "\n\n".join(sections))
    document = document.replace(ENV_ONLY_MARKER, _table(env_only))
    return document.replace(NOT_A_SETTING_MARKER, _env_only_table())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed file is stale instead of rewriting it.",
    )
    args = parser.parse_args()

    content = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != content:
            raise SystemExit(
                f"{OUTPUT.relative_to(REPO_ROOT)} is out of date. Run `make docs-settings`."
            )
        print(f"{OUTPUT.relative_to(REPO_ROOT)} is up to date.")
        return

    OUTPUT.write_text(content, encoding="utf-8")
    total = len(Config.model_fields)
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({total} settings).")


if __name__ == "__main__":
    main()
