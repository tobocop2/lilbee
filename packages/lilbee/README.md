# lilbee (npm)

Run [lilbee](https://lilbee.sh) from npm. The package is a small launcher: on
first use it downloads the standalone lilbee binary for your platform
(sha256-verified, cached), then passes every command straight through.

```bash
npm install -g lilbee
lilbee chat          # the TUI, on your own machine
lilbee serve         # the HTTP API server
lilbee model list    # anything the lilbee CLI does
```

One-shot without installing:

```bash
npx -y lilbee chat
```

The first local run downloads a large binary (about 370MB on macOS, more for
CUDA builds). Run `lilbee prepare` once to do that ahead of time. Every later
run execs the cached binary directly.

## MCP

Two equivalent entry points start the MCP server (stdio) for agent hosts:

```bash
npx -y lilbee mcp
npx -y lilbee-mcp        # same thing, one token for host configs
```

- **Local mode** (default): runs `lilbee mcp` on the resolved binary.
- **Remote mode** (`LILBEE_URL` set): skips the binary entirely and bridges
  stdio to a lilbee server's streamable-http `/mcp` endpoint, sending your
  session token as a bearer header. Use this when lilbee runs on another
  machine, such as a GPU box.

Example host config:

```json
{
  "mcpServers": {
    "lilbee": {
      "command": "npx",
      "args": ["-y", "lilbee", "mcp"],
      "env": { "LILBEE_DATA_DIR": "/path/to/your/knowledge-base" }
    }
  }
}
```

Remote:

```json
{
  "mcpServers": {
    "lilbee": {
      "command": "npx",
      "args": ["-y", "lilbee", "mcp"],
      "env": {
        "LILBEE_URL": "http://localhost:8383/mcp",
        "LILBEE_TOKEN": "…"
      }
    }
  }
}
```

## Configuration

| Variable | Effect |
| --- | --- |
| `LILBEE_URL` | Remote lilbee `/mcp` URL. Enables the MCP bridge mode. |
| `LILBEE_TOKEN` | Session token for the remote server (from its `server.json`). |
| `LILBEE_BIN` | Explicit path to a lilbee binary. |
| `LILBEE_DATA_DIR` | Library location for `mcp`; same as `--data-dir`. |
| `LILBEE_VARIANT` | Download variant: `cu121`, `cu124`, `cu125`, `rocm`, `compat`. |
| `LILBEE_RELEASE` | Override the pinned lilbee release tag. |
| `LILBEE_MCP_CACHE` | Override the download cache directory. |

Binary resolution order: `LILBEE_BIN` → `lilbee` on PATH → the shared-root
binary other lilbee installers manage (`<data root>/bin/lilbee`, read-only) →
cached download → download the pinned release asset. The pinned release lives in `package.json`
under `lilbee.release`, so each npm version maps to one lilbee release.

## Development

```bash
npm install
npm test
```

The package lives in the lilbee repo at `packages/lilbee/`.
