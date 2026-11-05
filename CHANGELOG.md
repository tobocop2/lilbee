# Changelog

## Unreleased

The `[engine]` extra is gone. `pip install lilbee[engine]` no longer installs an engine. Run `pip install --pre lilbee`, then run `lilbee engine install` to get the llama.cpp engine pack. If you skip the command, lilbee installs the pack the first time a model needs it.

If you installed the old extra, the `lilbee-engine-llama-cpp` wheel stays in your environment. The wheel takes precedence over the pack. To use the pack, run `pip uninstall lilbee-engine-llama-cpp`, then run `lilbee engine install`.

No pack exists yet for AMD (ROCm), Vulkan, Linux on ARM, or Intel Macs. On those computers, use a bundled build or set `LILBEE_LLAMA_SERVER_PATH` to your own `llama-server`.

## 0.6.66b481

You can now keep the HTTP server running in the background under your OS launcher: `brew services start lilbee` on macOS, `systemctl --user enable --now lilbee` on Arch, or import `nixosModules.lilbee` on NixOS. All three pin the server to `127.0.0.1:42697`. The daemon helps clients that hit the HTTP REST API (the Obsidian plugin, custom GUIs, scripts). The TUI, `lilbee chat`, the MCP server, and the rest of the CLI cold-start by design and are unaffected.

## Module reorganization

Module reorganization: top-level layout grouped under core/, data/, retrieval/, catalog/, modelhub/, runtime/. External users importing from old `lilbee.X` paths must update to the new package paths. No behavior change.
