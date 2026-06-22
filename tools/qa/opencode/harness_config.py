"""Module-level constants, path globals, and the scenario-status enum for the
opencode QA harness."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

QA_DIR = Path(__file__).resolve().parent
REPO_ROOT = QA_DIR.parent.parent.parent
RESULTS_DIR = QA_DIR / "results"
WORKSPACE_DIR = QA_DIR / "workspace"
LOG_DIR = QA_DIR / "logs"
SHARED_WORKSPACE = QA_DIR / "_shared_workspace"

_TMUX_SESSION_PREFIX = "lilbee-qa"
_TMUX_HISTORY_LINES = 3000
_TMUX_WINDOW_COLS = 200
_TMUX_WINDOW_ROWS = 50
_POST_SEND_SLEEP_S = 0.1

_SERVE_BOOT_TIMEOUT_S = 30.0
_SERVE_TERMINATE_TIMEOUT_S = 10.0
_OPENCODE_BOOT_SETTLE_S = 30.0  # boot + first-prompt prefill warmup
# Launcher warm gate (600s) plus a giant's cold load inside the fleet's
# weights-scaled health budget can hold the warm spinner for many minutes
# before opencode execs; only then may the scenario clock start.
_OPENCODE_UI_TIMEOUT_S = 1200.0
_UI_WAIT_HEARTBEAT_S = 60.0
_TMUX_COMMAND_TIMEOUT_S = 30.0
_INDEX_TIMEOUT_S = 120.0
_MODEL_PULL_TIMEOUT_S = 3600.0  # residential bandwidth, multi-quant repos can run 30+ min
_POLL_INTERVAL_S = 2.0
_SCENARIO_TIMEOUT_S = (
    1200.0  # giant MoE (30B) on Apple Silicon Metal: multi-turn agentic loop is slow
)
_MULTI_TOOL_TIMEOUT_S = 600.0
_INTER_SCENARIO_SETTLE_S = 15.0  # let opencode finish the prior turn before queuing the next
# Fail-fast: declare a scenario dead when the pane stops changing (any output) for
# this long. Keeps a quiet model from eating the full timeout. Set high enough that
# a giant MoE on Metal (static "thinking" spinner reads as an idle pane) isn't
# killed mid-reason before it emits its first tool call.
_PANE_IDLE_TIMEOUT_S = 480.0

_PANE_EXCERPT_TAIL = 2000
_OPENCODE_PICKER_STATE = Path.home() / ".local" / "state" / "opencode" / "model.json"
_OPENCODE_SHARE_DIR = Path.home() / ".local" / "share" / "opencode"
# Pre-built Godot 4 class-reference corpus (one-time `lilbee add /root/godot/doc/classes`
# into LILBEE_DATA=<this>); each cell copies its data/ + documents/ so lilbee_search
# has the reference without re-embedding. Override with LILBEE_QA_CORPUS.
_GODOT_CORPUS = Path(os.environ.get("LILBEE_QA_CORPUS", str(Path.home() / "godot_corpus")))
# opencode's built-in tools that compete with lilbee_search; disabled per cell so
# the model uses lilbee's MCP search instead of webfetch/read/grep (search mode).
# ``question`` is off too: a model (especially a giant on the codegen prompt) that
# calls it renders an interactive multiple-choice prompt and blocks forever waiting
# for a keypress the headless harness never sends, which reads as a pane-idle
# timeout. With the tool gone the model picks its own defaults and proceeds.
_TOOLS_OFF = (
    "webfetch",
    "read",
    "write",
    "edit",
    "patch",
    "bash",
    "glob",
    "grep",
    "list",
    "todowrite",
    "todoread",
    "task",
    "question",
)

# Substrings whose appearance in the pane means the cell can't recover --
# record the scenario as FAIL immediately instead of polling to timeout.
_FAIL_FAST_MARKERS = (
    "does not support tool calls",
    "context_length_exceeded",
    "exceeds the usable budget",
    "Internal Server Error",
)
_RAW_MARKER_FORBIDDEN = (
    "<tool_call>",
    "[TOOL_CALLS]",
    "functools[",
    "Error:",
    "Traceback",
    # Model emitted the tool call as raw JSON text instead of using opencode's
    # tool-call channel. lilbee's per-family extractor did not pick the
    # payload up, so opencode renders the JSON as a chat reply. The cell
    # cannot be marked "supported" if this happens.
    '{"name": "lilbee_',
    '{"name":"lilbee_',
)
_SUSPENDED_SUFFIX = ".qa-suspended"
_CHAT_CTX_TARGET = 131072  # ~24K goes to opencode's system + tools schema; the
# AGENTS.md plan/search/verify workflow needs the rest. resolve_chat_ctx clamps this
# to min(model trained ctx, host VRAM), so small-context models are unaffected and
# giants (qwen3-coder 256K, etc.) get the window the multi-search codegen turn needs.
# Memory-constrained hosts (e.g. a 32 GB Apple Silicon laptop) can't hold the KV
# cache for a giant's full per-slot context across the 4 chat slots. Setting
# LILBEE_QA_NUM_CTX pins cfg.num_ctx for the cell, which the fleet honors directly
# (planning._role_ctx returns cfg.num_ctx before resolve_chat_ctx), capping the
# launched --ctx-size to num_ctx * slots. Unset on pods, where the giant gets the
# full window.
_NUM_CTX_OVERRIDE = os.environ.get("LILBEE_QA_NUM_CTX", "").strip()
_EMBED_REF = "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"


class ScenarioStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"


class ExpectedSupport(StrEnum):
    """Whether docs/opencode-models.md claims a family works through opencode.

    Lets a full-matrix run separate a real regression (a SUPPORTED family that
    failed) from a documented non-supported family (expected to fail). A run is
    green when every supported family passes and every unsupported one fails for
    its documented reason.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
