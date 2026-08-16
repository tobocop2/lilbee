# lilbee-mcp

Run [lilbee](https://lilbee.sh)'s MCP server from `npx`. One command gives any MCP
client — Cabinet, Claude Code, Claude Desktop, Cursor — lilbee's tools: cited search
over your indexed files, ingest, crawling, and model management (browse the catalog,
pull, list, remove, GPU placement).

```bash
npx -y lilbee-mcp
```

## What it does

- **Local mode** (default). Finds a lilbee install and runs `lilbee mcp` (stdio).
  Resolution order: `LILBEE_BIN` → `lilbee` on PATH → cached download → download the
  standalone binary for your platform from the lilbee release (sha256-verified,
  cached for next time).
- **Remote mode** (`LILBEE_URL` set). Skips the binary entirely and bridges stdio to
  a lilbee server's streamable-http `/mcp` endpoint, sending your session token as a
  bearer header. Use this when lilbee runs on another machine, such as a GPU box.

The first local run downloads a large binary (about 370MB on macOS). Run
`npx -y lilbee-mcp prepare` once to do the download ahead of time, so your MCP
client's startup timeout never sees it.

## Configuration

| Variable | Effect |
| --- | --- |
| `LILBEE_URL` | Remote lilbee `/mcp` URL. Enables remote mode. |
| `LILBEE_TOKEN` | Session token for the remote server (from its `server.json`). |
| `LILBEE_BIN` | Explicit path to a lilbee binary. |
| `LILBEE_DATA_DIR` | Library location; same as `--data-dir`. |
| `LILBEE_VARIANT` | Download variant: `cu121`, `cu124`, `cu125`, `rocm`, `compat`. |
| `LILBEE_RELEASE` | Override the pinned lilbee release tag. |
| `LILBEE_MCP_CACHE` | Override the download cache directory. |

## Example: client config

```json
{
  "mcpServers": {
    "lilbee": {
      "command": "npx",
      "args": ["-y", "lilbee-mcp"],
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
      "args": ["-y", "lilbee-mcp"],
      "env": {
        "LILBEE_URL": "http://localhost:8383/mcp",
        "LILBEE_TOKEN": "…"
      }
    }
  }
}
```

## Development

```bash
npm install
npm test
```

The package lives in the lilbee repo at `packages/lilbee-mcp/` and pins the lilbee
release it bootstraps in `package.json` (`lilbee.release`).
