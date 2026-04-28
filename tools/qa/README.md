# lilbee QA matrix

Pre-publish QA harness. Runs in GitHub Actions against the built-but-not-yet-public
wheel and PyInstaller binary. Publication of a release is gated on success.

Plan: see `docs/plans/qa-matrix-ci.md` (gitignored).

## Layout

```
tools/qa/
  conftest.py           # fixtures: lane, server_url, mcp_client, tui_session
  drivers/
    tui.py              # cross-platform PTY harness (pywinpty/ptyprocess + pyte)
    mcp.py              # MCP stdio JSON-RPC client
    http.py             # SSE event-stream parsing helpers
  fixtures/             # tiny committed corpus
  test_smoke.py         # T0
  test_cli_*.py         # T1
  test_http_*.py        # T2
  test_mcp_*.py         # T3
  test_tui_*.py         # T4
  ...
  _phase0_harness_smoke.py   # GO/NO-GO validation; deleted once green on all OSes
```

## Local run

```bash
uv pip install -r tools/qa/requirements.txt
LILBEE_QA_LANE=l1-source LILBEE_QA_BIN="$(which lilbee)" \
  pytest tools/qa/ -m smoke -n auto --dist=loadgroup
```

`LILBEE_QA_LANE` selects the lane:
- `l1-source` — already-installed `lilbee` from the dev venv (local dev only)
- `l1-wheel` — installed from a wheel artifact (CI L1 lane)
- `l2-binary` — direct PyInstaller binary execution (CI L2 lane)

`LILBEE_QA_BIN` overrides the resolved binary path.

## CI

`.github/workflows/qa-matrix.yml` runs the full matrix on `release: types: [created]`
(draft creation), `pull_request` (reduced subset), `workflow_dispatch`, and weekly
`schedule`. See plan for gating granularity.
