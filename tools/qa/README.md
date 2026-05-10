# lilbee QA matrix

Pre-publish QA harness. Runs in GitHub Actions against the built-but-not-yet-public
wheel and release binary. Publication of a release is gated on success.

## Layout

```
tools/qa/
  conftest.py           # fixtures: lane, server_url, mcp_client, tui, models_pulled
  drivers/
    tui.py              # cross-platform PTY harness (pywinpty + pyte on Windows; pty.openpty + pyte on POSIX)
    mcp.py              # MCP stdio JSON-RPC client
  fixtures/notes/       # tiny committed corpus (coffee + EV)
  scripts/              # tools copied into the runner (sitecustomize_loopback.py)
  test_smoke.py         # T0
  test_cli_*.py         # T1
  test_http_*.py        # T2
  test_mcp_*.py         # T3
  test_tui_*.py         # T4
  test_e2e_*.py         # writer tests with real models (sync, search, ask, stream, crawl, wiki)
  test_model_pull_lifecycle.py
  test_reranker_smoke.py
```

## Local run

```bash
uv pip install -r tools/qa/requirements.txt
LILBEE_QA_LANE=l1-source LILBEE_QA_BIN="$(which lilbee)" \
  pytest tools/qa/ -m smoke -n auto --dist=loadgroup
```

`LILBEE_QA_LANE` selects the lane:
- `l1-source`: locally-installed `lilbee` from the dev venv (local dev only)
- `l1-pypi`: installed from a wheel (CI L1 lane; pulls from PyPI or a sibling-run artifact)
- `l2-binary`: direct release-binary execution (CI L2 lane)

`LILBEE_QA_BIN` overrides the resolved binary path.

## CI

`.github/workflows/qa-matrix.yml` runs the full matrix:
- `push` to the `qa-matrix` branch (paths: `tools/qa/**`, `.github/workflows/qa-matrix.yml`)
- `workflow_dispatch` with optional `lilbee_version`
- `workflow_call` from `release-candidate.yml` and `emergency-publish.yml` against the
  freshly-built wheel and binary artifacts of the same run

`release-candidate.yml` gates PyPI publish on this matrix passing.
