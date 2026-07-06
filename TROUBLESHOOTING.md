# Troubleshooting

When something goes wrong, the logs almost always say why. This page covers where they are, how to make them more verbose, and what to send when you [open an issue](https://github.com/tobocop2/lilbee/issues).

## Where the logs live

Everything lands in `logs/` under your data root:

| File | What's in it |
| --- | --- |
| `server.log` | Everything `lilbee serve` does at INFO and above: startup, requests, indexing, errors. Rotates at 2 MiB; the previous files are kept as `server.log.1` through `server.log.3`. |
| `server-fault.log` | Native crash tracebacks from `faulthandler`. Usually empty. If the server died without a Python traceback (a segfault in llama.cpp, for instance), this is where the evidence is. |
| `llama-swap.log` | Output from the engine supervisor (llama-swap) that fronts every model server: its HTTP access log plus the lines around a model server starting, stopping, or exiting. Check this when a model won't load or a request fails with "exited prematurely". |
| `launcher-serve.log` | Captures the `lilbee serve` a launcher (e.g. `lilbee launch opencode`) spawns for you, so a crash during a launched session leaves a trace. Capped at 5 MB. |
| `tui.log` | Logs from the full-screen terminal app, routed here so they don't draw over the screen. |

Each model runs in its own `llama-server` subprocess behind llama-swap; that
server's own startup/inference output is not a file on disk, it is streamed by
llama-swap and embedded into the error you get back (see "Model server crashes"
below).

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

## Model server crashes

lilbee runs each model (chat, embed, rerank, vision) in its own `llama-server`
subprocess, supervised by llama-swap. When one dies mid-request, llama-swap
reports it as **"exited prematurely"** and lilbee surfaces the server's own recent
output inline with the error, so the cause is usually right there in the message.
The usual causes are a model that doesn't fit in memory, an unsupported GGUF
architecture, or a GPU driver issue.

For the fuller picture, check `llama-swap.log` (the supervisor's view of every
server starting and stopping) and `server.log` (what `lilbee serve` was doing
around the failure). When reporting one, attach the last ~100 lines of each plus
`server-fault.log` if it's non-empty:

```bash
tail -n 100 "<data root>/logs/llama-swap.log"
tail -n 100 "<data root>/logs/server.log"
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
## Ingest replicas and VRAM reclaim

When an ingest job finishes, any extra embed or vision replicas that were started
for it are unloaded automatically. This frees their VRAM so chat and search
capacity is restored without a restart.

If you set `embed_replicas` or `vision_replicas` explicitly in your config, that
count is used during ingest and reclaimed the same way afterward. The default
(`0`) picks one replica per GPU, capped by whatever VRAM remains after the
persistent query servers (chat, embed-0, rerank, vision-0) are placed.

If the fleet ever reports "no model server" transiently, lilbee now re-probes the
engine and rebuilds a restarted llama-swap automatically before surfacing a
failure. A single retry is attempted; if the rebuilt fleet still cannot serve the
role, the normal `ProviderError` is returned.

## Using with the Obsidian plugin

The [Obsidian plugin](https://github.com/tobocop2/obsidian-lilbee) in managed mode runs this server for you, with each vault's data root under the plugin's shared install at `vaults/<id>/`. The same `logs/` layout applies inside that directory.

You don't have to dig those paths out by hand: the plugin's **Export diagnostics** command bundles these logs (plus plugin-side state) into a single file you can attach to an issue. See the plugin's own [TROUBLESHOOTING.md](https://github.com/tobocop2/obsidian-lilbee/blob/main/TROUBLESHOOTING.md) for that side of the setup.
