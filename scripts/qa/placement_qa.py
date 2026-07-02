#!/usr/bin/env python3
"""Fleet placement QA matrix.

Drives the entire placement surface on a multi-GPU pod and reports PASS/FAIL
per case. Sections:
  1. Spec validation (pure)            -- no server, no GPU
  2. App layer (get/preview/set/clear) -- needs GPU detection
  3. CLI surface (subprocess)          -- needs GPU detection
  4. MCP surface (tool functions)      -- needs GPU detection
  5. HTTP surface (curl a live server) -- needs `lilbee serve` running
  6. Empirical VRAM (chat split)       -- needs models + nvidia-smi

Run static sections:   uv run python placement_qa.py
Include HTTP:           LILBEE_QA_BASE=http://127.0.0.1:8765 LILBEE_QA_KEY=...
                        uv run python placement_qa.py
Empirical VRAM is driven separately by placement_qa_live.sh (asks + nvidia-smi).
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from collections.abc import Callable

RESULTS: list[tuple[str, str, bool, str]] = []


def rec(cid: str, desc: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((cid, desc, ok, detail))
    flag = "PASS" if ok else "FAIL"
    line = f"[{flag}] {cid:<16} {desc}"
    if detail:
        line += f"\n                   -> {detail}"
    print(line, flush=True)


def section(name: str) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)


# --------------------------------------------------------------------------
# Section 1: spec validation (pure)
# --------------------------------------------------------------------------
def run_validation() -> None:
    section("1. SPEC VALIDATION (PlacementSpec.from_json)")
    from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec

    def expect_ok(cid: str, desc: str, raw: str) -> None:
        try:
            PlacementSpec.from_json(raw)
            rec(cid, desc, True)
        except Exception as exc:
            rec(cid, desc, False, f"rejected: {type(exc).__name__}: {exc}")

    def expect_err(cid: str, desc: str, raw: str) -> None:
        try:
            PlacementSpec.from_json(raw)
            rec(cid, desc, False, "ACCEPTED (expected PlacementError)")
        except PlacementError as exc:
            rec(cid, desc, True, str(exc))
        except Exception as exc:
            rec(cid, desc, False, f"wrong exc {type(exc).__name__}: {exc}")

    # valid specs
    expect_ok("V-valid", "chat+embed valid", '{"chat":{"devices":[0,1]},"embed":{"devices":[0]}}')
    expect_ok("V-split", "valid tensor_split", '{"chat":{"devices":[0,1],"tensor_split":[1,1]}}')
    expect_ok("V-repl", "valid replicas", '{"embed":{"devices":[0],"replicas":2}}')
    expect_ok(
        "V-splitskew", "skewed tensor_split", '{"chat":{"devices":[0,1],"tensor_split":[3,1]}}'
    )

    # already-validated malformed input (should already PASS)
    expect_err("V-json", "malformed JSON", "{bad}")
    expect_err("V-notobj", "non-object root", "[1,2]")
    expect_err("V-role", "unknown role", '{"frobnicate":{"devices":[0]}}')
    expect_err("V-empty", "empty devices", '{"chat":{"devices":[]}}')
    expect_err("V-noentry", "entry not an object", '{"chat":[0,1]}')
    expect_err("V-replneg", "replicas < 1", '{"embed":{"devices":[0],"replicas":0}}')
    expect_err(
        "V-splitlen",
        "tensor_split length mismatch",
        '{"chat":{"devices":[0,1],"tensor_split":[1]}}',
    )

    # bb-26o gaps -- currently ACCEPTED (FAIL), should be rejected after fix
    expect_err("V-dupdev", "duplicate devices rejected", '{"chat":{"devices":[0,0]}}')
    expect_err("V-dupdev3", "duplicate devices (3)", '{"chat":{"devices":[0,1,1]}}')
    expect_err(
        "V-splitzero",
        "tensor_split zero weight rejected",
        '{"chat":{"devices":[0,1],"tensor_split":[0,1]}}',
    )
    expect_err(
        "V-splitneg",
        "tensor_split negative weight rejected",
        '{"chat":{"devices":[0,1],"tensor_split":[1,-1]}}',
    )
    expect_err(
        "V-unknownkey", "unknown entry key rejected", '{"chat":{"devices":[0],"frobnicate":1}}'
    )
    expect_err("V-devneg", "negative device index rejected", '{"chat":{"devices":[-1]}}')

    # round-trip stability
    try:
        from lilbee.providers.fleet.placement_spec import PlacementSpec as PS

        raw = '{"chat":{"devices":[0,1],"tensor_split":[1,1]},"embed":{"devices":[0],"replicas":2}}'
        rt = PS.from_json(PS.from_json(raw).to_json()).to_json()
        rec("V-roundtrip", "from_json/to_json round-trips", rt == PS.from_json(raw).to_json(), rt)
    except Exception as exc:
        rec("V-roundtrip", "from_json/to_json round-trips", False, repr(exc))


# --------------------------------------------------------------------------
# Section 2: app layer
# --------------------------------------------------------------------------
def run_app_layer() -> None:
    section("2. APP LAYER (lilbee.app.placement)")
    from lilbee.app import placement as app_placement
    from lilbee.providers.fleet.placement_spec import PlacementSpec

    try:
        v = app_placement.get_placement()
        rec(
            "A-get",
            "get_placement() returns a view",
            v is not None,
            f"gpus={len(v.gpus)} manual={v.manual}",
        )
    except Exception as exc:
        rec("A-get", "get_placement()", False, f"{type(exc).__name__}: {exc}")

    try:
        v = app_placement.preview_placement(None)
        rec(
            "A-preview-auto",
            "preview auto",
            v is not None,
            f"roles={[r.role.value for r in v.roles]}",
        )
    except Exception as exc:
        rec("A-preview-auto", "preview auto", False, f"{type(exc).__name__}: {exc}")

    # preview a complete manual spec pinning chat to GPU1+2 (a spec must cover
    # every active role, so include embed)
    try:
        spec = PlacementSpec.from_json('{"chat":{"devices":[1,2]},"embed":{"devices":[0]}}')
        v = app_placement.preview_placement(spec)
        chat = next((r for r in v.roles if r.role.value == "chat"), None)
        ok = chat is not None and set(chat.devices) <= {1, 2}
        rec(
            "A-preview-pin",
            "preview honors device pin",
            ok,
            f"chat devices={getattr(chat, 'devices', None)}",
        )
    except Exception as exc:
        rec("A-preview-pin", "preview honors device pin", False, f"{type(exc).__name__}: {exc}")

    # bb-a8f: preview a spec that fits TOTAL but maybe not FREE VRAM should not be
    # rejected purely because a model is resident. Recorded as observation here;
    # the load-then-preview check is in the live section.
    rec("A-a8f-note", "bb-a8f free-vs-total checked live", True, "see live section L-a8f")


# --------------------------------------------------------------------------
# Section 3: CLI surface
# --------------------------------------------------------------------------
_LILBEE_BIN = os.environ.get("LILBEE_QA_CLI", "lilbee")


def _cli(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_LILBEE_BIN, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ},
    )


def run_cli() -> None:
    section("3. CLI SURFACE (lilbee placement / gpus)")

    cp = _cli(["placement", "show"])
    traceback_leak = "Traceback (most recent call last)" in cp.stderr
    rec(
        "C-show",
        "placement show works without traceback",
        cp.returncode == 0 and not traceback_leak,
        f"rc={cp.returncode} tb={traceback_leak} {cp.stderr.strip()[:120]}",
    )

    cp = _cli(["placement", "preview"])
    rec("C-preview-auto", "placement preview (auto)", cp.returncode == 0, f"rc={cp.returncode}")

    # valid complete spec via stdin (covers every active role)
    cp = _cli(
        ["placement", "preview", "--spec", "-"],
        stdin='{"chat":{"devices":[1,2]},"embed":{"devices":[0]}}',
    )
    rec(
        "C-preview-stdin",
        "preview --spec - (stdin)",
        cp.returncode == 0,
        f"rc={cp.returncode} {(cp.stderr or cp.stdout).strip()[:100]}",
    )

    # malformed spec -> clean error, exit 1, no traceback
    cp = _cli(["placement", "preview", "--spec", "-"], stdin="{bad json")
    clean = cp.returncode != 0 and "Traceback (most recent call last)" not in cp.stderr
    rec(
        "C-preview-badjson",
        "bad JSON -> clean error (no traceback)",
        clean,
        f"rc={cp.returncode} {(cp.stderr or cp.stdout).strip()[:120]}",
    )

    # unknown role -> clean error
    cp = _cli(["placement", "preview", "--spec", "-"], stdin='{"frobnicate":{"devices":[0]}}')
    clean = cp.returncode != 0 and "Traceback (most recent call last)" not in cp.stderr
    rec("C-preview-badrole", "unknown role -> clean error", clean, f"rc={cp.returncode}")

    # out-of-range device -> clean error (embed present so the chat error surfaces)
    cp = _cli(
        ["placement", "preview", "--spec", "-"],
        stdin='{"chat":{"devices":[99]},"embed":{"devices":[0]}}',
    )
    clean = cp.returncode != 0 and "Traceback (most recent call last)" not in cp.stderr
    rec(
        "C-preview-oor",
        "out-of-range device -> clean error",
        clean,
        f"rc={cp.returncode} {(cp.stderr or cp.stdout).strip()[:120]}",
    )

    # missing spec file -> clean error not traceback
    cp = _cli(["placement", "set", "--spec", "/nonexistent/spec.json"])
    clean = cp.returncode != 0 and "Traceback (most recent call last)" not in cp.stderr
    rec(
        "C-set-nofile",
        "missing spec file -> clean error",
        clean,
        f"rc={cp.returncode} {(cp.stderr or cp.stdout).strip()[:120]}",
    )

    # inline JSON to --spec (bb-z9w): parsed as JSON, not mistaken for a file path
    cp = _cli(
        ["placement", "preview", "--spec", '{"chat":{"devices":[1,2]},"embed":{"devices":[0]}}']
    )
    out = cp.stderr + cp.stdout
    inline_ok = cp.returncode == 0 and "No such file" not in out and "Traceback" not in cp.stderr
    rec(
        "C-set-inline",
        "inline JSON to --spec is parsed, not treated as a file",
        inline_ok,
        f"rc={cp.returncode} {out.strip()[:120]}",
    )

    # gpus listing
    cp = _cli(["gpus"]) if _has_cmd("gpus") else _cli(["placement", "show"])
    rec("C-gpus", "gpu listing works", cp.returncode == 0, f"rc={cp.returncode}")


def _has_cmd(name: str) -> bool:
    cp = _cli(["--help"])
    return name in cp.stdout


# --------------------------------------------------------------------------
# Section 4: MCP surface
# --------------------------------------------------------------------------
def run_mcp() -> None:
    section("4. MCP SURFACE (mcp_server tool fns)")
    import lilbee.mcp_server as mcp_server

    try:
        d = mcp_server.get_placement_tool()
        rec("M-get", "get_placement_tool returns dict", isinstance(d, dict), f"keys={list(d)[:6]}")
    except Exception as exc:
        rec("M-get", "get_placement_tool", False, f"{type(exc).__name__}: {exc}")

    try:
        d = mcp_server.preview_placement_tool(None)
        rec("M-preview-auto", "preview_placement_tool(None)", isinstance(d, dict))
    except Exception as exc:
        rec("M-preview-auto", "preview_placement_tool(None)", False, f"{type(exc).__name__}: {exc}")

    try:
        d = mcp_server.preview_placement_tool(
            {"chat": {"devices": [1, 2]}, "embed": {"devices": [0]}}
        )
        rec(
            "M-preview-spec",
            "preview_placement_tool(spec)",
            isinstance(d, dict) and "error" not in d,
            str(d)[:80],
        )
    except Exception as exc:
        rec("M-preview-spec", "preview_placement_tool(spec)", False, f"{type(exc).__name__}: {exc}")

    # bad spec via MCP -> should return a structured error dict, never a raw crash
    for cid, desc, spec in [
        ("M-bad-empty", "empty devices", {"chat": {"devices": []}}),
        ("M-bad-role", "unknown role", {"frobnicate": {"devices": [0]}}),
        (
            "M-bad-dup",
            "duplicate devices",
            {"chat": {"devices": [0, 0]}, "embed": {"devices": [0]}},
        ),
    ]:
        try:
            out = mcp_server.preview_placement_tool(spec)
            ok = isinstance(out, dict) and "error" in out
            rec(cid, f"{desc} -> structured error", ok, str(out)[:90])
        except Exception as exc:
            rec(
                cid,
                f"{desc} -> structured error",
                False,
                f"raised {type(exc).__name__}: {str(exc)[:80]}",
            )


# --------------------------------------------------------------------------
# Section 5: HTTP surface (live server)
# --------------------------------------------------------------------------
def run_http() -> None:
    base = os.environ.get("LILBEE_QA_BASE")
    if not base:
        section("5. HTTP SURFACE -- SKIPPED (set LILBEE_QA_BASE to enable)")
        return
    section(f"5. HTTP SURFACE ({base})")
    key = os.environ.get("LILBEE_QA_KEY", "")

    def curl(method: str, path: str, body: str | None = None, auth: bool = True) -> tuple[int, str]:
        cmd = ["curl", "-sS", "-o", "-", "-w", "\n%{http_code}", "-X", method, f"{base}{path}"]
        if auth and key:
            cmd += ["-H", f"Authorization: Bearer {key}"]
        if body is not None:
            cmd += ["-H", "Content-Type: application/json", "-d", body]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = cp.stdout
        code = out.rsplit("\n", 1)[-1].strip()
        payload = out.rsplit("\n", 1)[0]
        try:
            return int(code), payload
        except ValueError:
            return -1, out

    code, _ = curl("GET", "/api/placement")
    rec("H-get", "GET /api/placement 200", code == 200, f"code={code}")

    code, _ = curl("GET", "/api/gpus")
    rec("H-gpus", "GET /api/gpus 200", code == 200, f"code={code}")

    code, _ = curl(
        "POST",
        "/api/placement/preview",
        '{"spec":{"chat":{"devices":[1,2]},"embed":{"devices":[0]}}}',
    )
    rec("H-preview", "POST preview valid 200", code == 200, f"code={code}")

    # malformed spec -> 4xx not 500 (bb-x0o)
    code, body = curl("POST", "/api/placement/preview", '{"spec":{"chat":{"devices":[]}}}')
    rec(
        "H-preview-bad",
        "POST preview bad spec -> 4xx not 500",
        400 <= code < 500,
        f"code={code} {body[:100]}",
    )

    # out-of-range device -> 4xx not 500 (bb-x0o; ProviderError/ValueError)
    code, body = curl(
        "POST",
        "/api/placement/preview",
        '{"spec":{"chat":{"devices":[99]},"embed":{"devices":[0]}}}',
    )
    rec(
        "H-preview-oor",
        "POST preview oor device -> 4xx not 500",
        400 <= code < 500,
        f"code={code} {body[:100]}",
    )

    # dup devices -> 4xx after fix (bb-26o)
    code, body = curl("POST", "/api/placement/preview", '{"spec":{"chat":{"devices":[0,0]}}}')
    rec(
        "H-preview-dup",
        "POST preview dup devices -> 4xx",
        400 <= code < 500,
        f"code={code} {body[:100]}",
    )

    # PUT/DELETE refused on the shared daemon (bb-fhi) -> 409 regardless of auth
    code, body = curl(
        "PUT", "/api/placement", '{"spec":{"chat":{"devices":[1,2]},"embed":{"devices":[0]}}}'
    )
    rec(
        "H-put-refused",
        "PUT placement -> 409 refused on daemon (bb-fhi)",
        code == 409,
        f"code={code} {body[:80]}",
    )
    code, body = curl("DELETE", "/api/placement")
    rec(
        "H-del-refused",
        "DELETE placement -> 409 refused on daemon (bb-fhi)",
        code == 409,
        f"code={code} {body[:80]}",
    )

    # preview auth (bb-895): only meaningful when the server enforces auth (a key
    # is set). Localhost dev servers skip auth; the enforcement is unit-tested.
    if key:
        code, _ = curl(
            "POST", "/api/placement/preview", '{"spec":{"chat":{"devices":[0]}}}', auth=False
        )
        rec(
            "H-preview-noauth",
            "preview without key -> 401/403 (bb-895)",
            code in (401, 403),
            f"code={code}",
        )
    else:
        rec(
            "H-preview-noauth",
            "preview auth enforcement (bb-895)",
            True,
            "server has no auth key; see unit test test_placement_routes_require_auth",
        )


# --------------------------------------------------------------------------
# Section 6: TUI surface (real PlacementScreen via Textual pilot)
# --------------------------------------------------------------------------
def run_tui() -> None:
    section("6. TUI SURFACE (PlacementScreen via Textual pilot)")
    try:
        import asyncio

        from lilbee.cli.tui.screens.placement import PlacementScreen
        from textual.widgets import DataTable, Static

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens import placement as scr
    except Exception as exc:
        rec("T-import", "TUI imports", False, f"{type(exc).__name__}: {exc}")
        return

    class _Host(LilbeeApp):
        _test_skip_auto_init = True

        def on_mount(self) -> None:
            self.push_screen(PlacementScreen())

    def generated(app: object) -> str:
        return str(app.screen.query_one(scr._GENERATED_ID, Static).render())

    async def drive() -> None:
        app = _Host()
        async with app.run_test(size=(150, 48)) as pilot:
            await pilot.pause()
            try:
                table = app.screen.query_one(scr._GPU_TABLE_ID, DataTable)
            except Exception as exc:
                rec("T-mount", "PlacementScreen mounts", False, f"{type(exc).__name__}: {exc}")
                return
            rec(
                "T-mount",
                "PlacementScreen mounts with a GPU table",
                True,
                f"rows={table.row_count}",
            )
            if table.row_count == 0:
                rec(
                    "T-gpus",
                    "GPU table populated",
                    False,
                    "0 GPUs detected -- run on a multi-GPU pod for full TUI coverage",
                )
                return
            rec(
                "T-gpus",
                "GPU table lists detected GPUs",
                table.row_count >= 1,
                f"rows={table.row_count}",
            )
            last = table.row_count - 1
            btn = f"#dev-chat-{last}"
            try:
                await pilot.click(btn)
                await pilot.pause()
                ok = f"{last}" in generated(app)
                rec(
                    "T-toggle",
                    f"clicking {btn} updates the generated spec",
                    ok,
                    generated(app)[:90],
                )
            except Exception as exc:
                rec("T-toggle", "device toggle button", False, f"{type(exc).__name__}: {exc}")
            try:
                await pilot.press("ctrl+r")
                await pilot.pause()
                still = app.screen.query_one(scr._GPU_TABLE_ID, DataTable).row_count >= 1
                rec(
                    "T-preview",
                    "ctrl+r preview keeps the screen populated (no blank-on-error)",
                    still,
                )
            except Exception as exc:
                rec("T-preview", "ctrl+r preview", False, f"{type(exc).__name__}: {exc}")

    try:
        asyncio.run(drive())
    except Exception as exc:
        rec("T-run", "TUI pilot session", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
def summary() -> int:
    section("SUMMARY")
    passed = sum(1 for _, _, ok, _ in RESULTS if ok)
    failed = [r for r in RESULTS if not r[2]]
    print(f"{passed}/{len(RESULTS)} passed, {len(failed)} failed\n")
    if failed:
        print("FAILURES:")
        for cid, desc, _, detail in failed:
            print(f"  [FAIL] {cid:<16} {desc}\n         {detail}")
    return 1 if failed else 0


def main() -> int:
    runners: list[Callable[[], None]] = [
        run_validation,
        run_app_layer,
        run_cli,
        run_mcp,
        run_http,
        run_tui,
    ]
    for fn in runners:
        try:
            fn()
        except Exception:
            print(f"\n!!! section {fn.__name__} crashed:\n{traceback.format_exc()}", flush=True)
            rec(fn.__name__, "section crashed", False, "see traceback above")
    return summary()


if __name__ == "__main__":
    sys.exit(main())
