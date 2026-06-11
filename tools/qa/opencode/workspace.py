"""Per-cell workspace: corpus seeding, config.toml, AGENTS.md, fixture indexing."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

from harness_config import (
    _CHAT_CTX_TARGET,
    _EMBED_REF,
    _GODOT_CORPUS,
    _INDEX_TIMEOUT_S,
    _NUM_CTX_OVERRIDE,
    QA_DIR,
    SHARED_WORKSPACE,
    WORKSPACE_DIR,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def seed_shared_workspace() -> Path:
    """Drop three markdown fixtures so lilbee_search returns real results."""
    SHARED_WORKSPACE.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "chat_worker.md": (
            "# Chat engine\n\n"
            "Chat inference in lilbee runs on a managed llama-server fleet. "
            "llama-swap supervises the server processes behind an "
            "OpenAI-compatible proxy, gguf-parser estimates each model's "
            "memory footprint for placement, and tokens stream back via SSE."
        ),
        "dispatch.md": (
            "# Dispatch layer\n\n"
            "chat_dispatch.dispatch_chat is the canonical entry point that "
            "the OpenAI-compatible route forwards to. It resolves the model "
            "through KnownModelCache, enforces tool capability, and routes "
            "to either the local llama-server fleet or the SDK backend."
        ),
        "tool_extraction.md": (
            "# Tool extraction\n\n"
            "Lilbee launches llama-server with --jinja, so the server renders "
            "each model's own chat template and parses its native tool-call "
            "syntax into structured message.tool_calls. A recovery pass in "
            "providers/fleet/client.py catches bare-JSON tool calls that "
            "models emit as plain content."
        ),
    }
    for filename, content in fixtures.items():
        target = SHARED_WORKSPACE / filename
        if not target.exists() or target.read_text() != content:
            target.write_text(content)
    return SHARED_WORKSPACE


_AGENTS_MD = """\
## Instructions

You are working with Godot 4.4. Your training data is outdated -- many classes and
methods were renamed or replaced in Godot 4.x, so you cannot rely on memory.

**You MUST use the `lilbee_search` MCP tool to look up Godot APIs.** The Godot 4.4
class reference is indexed locally in lilbee. Do NOT guess or use web search -- call
`lilbee_search` with the exact class or method name.

### Workflow

1. **Plan** -- list every Godot class and method your answer or code needs.
2. **Search** -- call `lilbee_search` for each one individually. Do not skip a class
   even if you are confident; verify every method signature.
3. **Answer / write** -- use only class names, methods, and properties exactly as
   confirmed by the search results. If something was not found, do not use it.
4. **Verify** -- before finishing, re-check every class and method you used against a
   `lilbee_search` result and fix anything that does not match.
"""


def write_per_cell_workspace(family: str, model_ref: str) -> Path:
    """Per-cell workspace seeded by copying the pre-built Godot corpus lancedb.

    Wipes any prior ``.lilbee/`` from previous cells, then copies the shared
    corpus's ``data/`` + ``documents/`` (a fast file copy, no re-embed) so
    lilbee_search has the Godot class reference. The cell's config.toml pins the
    target chat model. Requires the one-time corpus build (see README); a missing
    corpus is a hard error so cells never silently run against an empty index.
    """
    if not (_GODOT_CORPUS / "data").is_dir():
        raise RuntimeError(
            f"Godot corpus not found at {_GODOT_CORPUS}/data. Build it once: "
            f"LILBEE_DATA={_GODOT_CORPUS} uv run lilbee add <godot>/doc/classes"
        )
    workspace = WORKSPACE_DIR / family
    workspace.mkdir(parents=True, exist_ok=True)
    config_dir = workspace / ".lilbee"
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True)
    for sub in ("data", "documents"):
        src = _GODOT_CORPUS / sub
        if src.is_dir():
            shutil.copytree(src, config_dir / sub)
    config_lines = [
        f'chat_model = "{model_ref}"',
        f'embedding_model = "{_EMBED_REF}"',
        f"chat_n_ctx_target = {_CHAT_CTX_TARGET}",
    ]
    if _NUM_CTX_OVERRIDE:
        config_lines.append(f"num_ctx = {_NUM_CTX_OVERRIDE}")
    (config_dir / "config.toml").write_text("\n".join(config_lines) + "\n")
    # opencode resolves its project root by traversing up to the nearest git
    # directory; without this the workspace (which lives inside the lilbee
    # repo) would resolve to the repo root, so the per-cell opencode.json and
    # the event-tap plugin below would never load.
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    # Event tap: opencode loads project plugins from .opencode/plugins/; the tap
    # appends real bus events (tool dispatch, session idle/error) to
    # .lilbee/qa-events.jsonl, which scenarios.py prefers over pane scraping.
    plugins_dir = workspace / ".opencode" / "plugins"
    if plugins_dir.exists():
        shutil.rmtree(plugins_dir)
    plugins_dir.mkdir(parents=True)
    shutil.copyfile(QA_DIR / "qa_events_plugin.js", plugins_dir / "qa-events.js")
    # opencode reads AGENTS.md from the project dir. This is the published
    # godot-with-lilbee workflow: it tells the model its Godot knowledge is stale
    # and it MUST look every class/method up via lilbee_search before using it.
    # Without it a capable coder model just writes GDScript from memory and never
    # calls the tool, so the integration looks broken when it is only untriggered.
    (workspace / "AGENTS.md").write_text(_AGENTS_MD)
    return workspace


def index_workspace(workspace: Path, log_path: Path) -> None:
    """Run lilbee add on each fixture so lilbee_search has content.

    Sets ``LILBEE_DATA`` explicitly so the `lilbee add` subprocess writes
    into the workspace lancedb. matrix.py inherits a polluted ``LILBEE_DATA``
    pointing at the global data root via ``lilbee.core.config``'s module-import
    side effect, so without this override the fixtures index into the wrong
    lancedb and `lilbee_search` calls return empty in opencode.
    """
    import os

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["LILBEE_DATA"] = str(workspace / ".lilbee")
    with log_path.open("a") as log:
        log.write("=== lilbee add ===\n")
        log.flush()
        for fixture in ("chat_worker.md", "dispatch.md", "tool_extraction.md"):
            subprocess.run(
                ["uv", "run", "lilbee", "add", str(workspace / fixture)],
                cwd=workspace,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=_INDEX_TIMEOUT_S,
            )
