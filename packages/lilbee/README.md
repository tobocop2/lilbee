<p align="center">
  <a href="https://lilbee.sh/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tobocop2/lilbee/main/docs/lilbee-logo-dark.svg">
      <img alt="lilbee" src="https://raw.githubusercontent.com/tobocop2/lilbee/main/docs/lilbee-logo-light.svg" width="340">
    </picture>
  </a>
</p>

<p align="center"><strong>The whole local AI stack in one executable: it runs and manages the models, and searches everything you own with them.</strong></p>

<p align="center"><a href="https://lilbee.sh/">Project site</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/tutorial">Tutorial reels</a> &nbsp;·&nbsp; <a href="https://github.com/tobocop2/lilbee">GitHub</a> &nbsp;·&nbsp; <a href="https://lilbee.sh/api/">REST API</a> &nbsp;·&nbsp; <a href="https://obsidian.lilbee.sh/">Obsidian plugin</a> &nbsp;·&nbsp; <a href="https://web.libera.chat/#lilbee">Chat (#lilbee)</a></p>

<p align="center">
  <a href="https://www.npmjs.com/package/lilbee"><img src="https://img.shields.io/npm/v/lilbee?label=npm&logo=npm&logoColor=ebbcba&style=flat-square&labelColor=191724&color=ebbcba" alt="lilbee on npm"></a>
  <a href="https://github.com/tobocop2/lilbee/releases/latest"><img src="https://img.shields.io/github/v/release/tobocop2/lilbee?label=release&logo=github&logoColor=c4a7e7&style=flat-square&labelColor=191724&color=c4a7e7" alt="Latest release"></a>
  <img src="https://img.shields.io/badge/node-18%2B-9ccfd8?logo=nodedotjs&logoColor=9ccfd8&style=flat-square&labelColor=191724" alt="Node 18+">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-908caa?style=flat-square&labelColor=191724" alt="Platforms">
  <a href="https://github.com/tobocop2/lilbee/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-6e6a86?style=flat-square&labelColor=191724" alt="License: MIT"></a>
</p>

lilbee runs and manages your models: chat, embedding, vision, and rerank, placed across every GPU you have. It puts them to work as a search engine you can talk to, over your files, notes, code, and the web, where every answer cites the exact file and line. It crawls websites into your library, launches your coding agents on local models, and hands any MCP-aware agent cited answers from everything you've indexed. Ask in plain English. No containers, no networking, nothing else to install or set up.

And it is private. Your files, the index, the embeddings, your questions, and the answers stay on your machine. lilbee sends no telemetry, needs no account, and makes no cloud call unless you configure a cloud model yourself.

![ask lilbee "what is lilbee in one sentence?" and get a cited answer drawn from its own README](https://raw.githubusercontent.com/tobocop2/lilbee/gh-pages/demos/what_is_lilbee.gif)

It's all one program: no separate model server, vector database, or container to stand up. lilbee runs the models and keeps the index itself, built on llama.cpp, with its own model manager and multi-GPU fleet. Reach it as a terminal app, CLI, Model Context Protocol server, HTTP API, or Python library. The [project README](https://github.com/tobocop2/lilbee#readme) covers the full feature set; this page covers the npm route.

> **Beta software.** lilbee is in active beta development. Interfaces, command names, and on-disk formats may shift between betas. Feedback and bug reports are very welcome; that's the whole point of the beta.

## Quick start

```bash
npm install -g lilbee
lilbee               # the terminal app; pick your models from the catalog on first run
lilbee serve         # the HTTP API server
lilbee model list    # anything the lilbee CLI does
```

One-shot without installing:

```bash
npx -y lilbee chat
```

## How the package works

The package is a small launcher with zero runtime dependencies. On first use it detects your hardware and downloads the matching standalone binary of the **latest lilbee release** — CUDA on NVIDIA, ROCm on AMD, Metal on Apple silicon, and an AVX-baseline build on older x86-64 CPUs — sha256-verified against the release manifest and cached. Every later run execs the cached binary directly, with no network.

The first download is large (about 370MB on macOS, more for CUDA builds). Run `lilbee prepare` once to do it ahead of time; run it again any time to upgrade to the newest release (the old binary is removed). If the latest-release lookup is unreachable, the launcher falls back to the release pinned in `package.json` — the one this launcher version was tested against.


## MCP

Two equivalent entry points start the MCP server (stdio) for agent hosts:

```bash
npx -y lilbee mcp
npx -y lilbee-mcp        # same thing, one token for host configs
```

- **Local mode** (default): runs `lilbee mcp` on the resolved binary.
- **Remote mode** (`LILBEE_URL` set): skips the binary entirely and bridges stdio to a lilbee server's streamable-http `/mcp` endpoint, sending your session token as a bearer header. Use this when lilbee runs on another machine, such as a GPU box.

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
| `LILBEE_VARIANT` | Override the detected download variant: `default` (the plain build), `cu121`, `cu124`, `cu125`, `rocm`, `compat`. |
| `LILBEE_RELEASE` | Run an exact lilbee release tag instead of the latest. |
| `LILBEE_DEBUG` | `=1` prints binary resolution detail on every run. |
| `LILBEE_MCP_CACHE` | Override the download cache directory. |

The launcher always runs its own sha256-verified download; it ignores lilbee installs from other package managers, so `npm install` means a fresh, known binary. `LILBEE_BIN` is the one escape hatch: set it to run a specific binary instead. The fallback release lives in `package.json` under `lilbee.release`; it is used only when the latest-release lookup fails.

## Uninstall

`npm uninstall -g lilbee` removes only the launcher — the npm package is a small shim, and the lilbee binary it downloaded stays in the cache (up to ~1.2GB). npm runs no uninstall scripts, so no package can clean its own cache on uninstall. Delete the binaries first, then the package:

```bash
lilbee unprepare        # deletes every downloaded binary
npm uninstall -g lilbee
```

Already uninstalled the package? `npx -y lilbee unprepare` still works. The cache lives at `~/Library/Caches/lilbee-npm` (macOS), `~/.cache/lilbee-npm` (Linux), or `%LOCALAPPDATA%\lilbee-npm` (Windows) if you prefer to delete it by hand. Models and your library are separate: they belong to lilbee itself, not this launcher, and `lilbee unprepare` does not touch them.

## Requirements

Node 18+ for the launcher. The binary runs on macOS (Apple silicon and Intel), Linux x86_64 (glibc 2.34+), and Windows x86_64. No GPU required; with one, lilbee uses it. See the [hardware requirements](https://github.com/tobocop2/lilbee#hardware-requirements) for the full breakdown.

## Other install channels

Prefer a system package manager? lilbee also ships via [Homebrew, AUR, Nix, Docker, Flatpak, Snap, Scoop, PyPI, and standalone binaries](https://github.com/tobocop2/lilbee#install). Same `lilbee` command everywhere.

## Development

```bash
npm install
npm test
```

The package lives in the lilbee repo at [`packages/lilbee/`](https://github.com/tobocop2/lilbee/tree/main/packages/lilbee).
