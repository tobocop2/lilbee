# Troubleshooting

When something goes wrong, the logs almost always say why. This page covers where they are, how to make them more verbose, and what to send when you [open an issue](https://github.com/tobocop2/lilbee/issues).

## Where the logs live

Everything lands in `logs/` under your data root:

| File | What's in it |
| --- | --- |
| `server.log` | Everything `lilbee serve` does at INFO and above: startup, requests, indexing, errors. Rotates at 2 MiB; the previous files are kept as `server.log.1` through `server.log.3`. |
| `server-fault.log` | Native crash tracebacks from `faulthandler`. Usually empty. If the server died without a Python traceback (a segfault in llama.cpp, for instance), this is where the evidence is. |
| `worker-<role>.log` | One file per model worker subprocess: `worker-chat.log`, `worker-embed.log`, `worker-rerank.log`, `worker-vision.log`. Model load output and per-request activity for that role. |
| `tui.log` | Logs from the full-screen terminal app, routed here so they don't draw over the screen. |

**The data root is resolved in this order**, highest precedence first:

1. `--global` forces the platform default below, ignoring any local `.lilbee/`.
2. `--data-dir <path>` uses exactly that path.
3. `LILBEE_DATA` environment variable, if set.
4. A `.lilbee/` directory found by walking up from the current directory (per-project libraries).
5. The platform default: `~/Library/Application Support/lilbee` on macOS, `$XDG_DATA_HOME/lilbee` (default `~/.local/share/lilbee`) on Linux, `%LOCALAPPDATA%\lilbee` on Windows.

Not sure which one you're on? `lilbee status` prints the active data directory.

## Turning up verbosity

The log level is controlled by the `LILBEE_LOG_LEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, `ERROR`; default `WARNING`) or the `--log-level` flag, which overrides the variable:

```bash
LILBEE_LOG_LEVEL=DEBUG lilbee serve
# or
lilbee --log-level DEBUG serve
```

`WARNING` keeps the console to problems only. `INFO` adds the normal lifecycle: startup, requests, indexing progress. `DEBUG` adds everything underneath, which is what you want when reproducing a bug.

**The file log is more verbose than the console.** `server.log` captures INFO and above even at the default level, so the console staying quiet doesn't mean the file is empty. Check `server.log` first; raise the level only when you need DEBUG detail.

## Worker crashes

lilbee runs each model in its own subprocess. When one dies mid-request you get a `WorkerCrashError` naming the role, for example:

```
WorkerCrashError: Worker 'embed' subprocess exited unexpectedly. See <data root>/logs/worker-embed.log for details.
```

The error already includes the last few log lines, but the full story is in the worker's own file: `worker-chat.log`, `worker-embed.log`, `worker-rerank.log`, or `worker-vision.log`. The usual causes are a model that doesn't fit in memory, an unsupported GGUF architecture, or a GPU driver issue, and the log tail names which.

When reporting a worker crash, attach the last ~100 lines of the worker log plus `server-fault.log` if it's non-empty:

```bash
tail -n 100 "<data root>/logs/worker-embed.log"
```

## Fleet refuses to start after a hardware change

If you have a manual placement spec set and a GPU is removed, replaced, or
renumbered, the fleet will refuse to start with an error naming the card that
no longer fits. The error is intentional: lilbee won't silently start in a
broken state.

To return to automatic placement:

```bash
lilbee placement clear
```

After that, the auto planner takes over again and places models across whatever
GPUs are now available.

## Using with the Obsidian plugin

The [Obsidian plugin](https://github.com/tobocop2/obsidian-lilbee) in managed mode runs this server for you, with each vault's data root under the plugin's shared install at `vaults/<id>/`. The same `logs/` layout applies inside that directory.

You don't have to dig those paths out by hand: the plugin's **Export diagnostics** command bundles these logs (plus plugin-side state) into a single file you can attach to an issue. See the plugin's own [TROUBLESHOOTING.md](https://github.com/tobocop2/obsidian-lilbee/blob/main/TROUBLESHOOTING.md) for that side of the setup.
