# Changelog

## 0.6.66b481

You can now keep the HTTP server running in the background under your OS launcher: `brew services start lilbee` on macOS, `systemctl --user enable --now lilbee` on Arch, or import `nixosModules.lilbee` on NixOS. All three pin the server to `127.0.0.1:42697`. The daemon helps clients that hit the HTTP REST API (the Obsidian plugin, custom GUIs, scripts). The TUI, `lilbee chat`, the MCP server, and the rest of the CLI cold-start by design and are unaffected.

## Module reorganization

Module reorganization: top-level layout grouped under core/, data/, retrieval/, catalog/, modelhub/, runtime/. External users importing from old `lilbee.X` paths must update to the new package paths. No behavior change.
