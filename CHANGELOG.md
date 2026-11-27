# Changelog

## Unreleased

You can now set the OCR mode and the OCR timeout for one run from the Python API and from the chat. `Lilbee.add`, `Lilbee.sync` and `Lilbee.rebuild` take `ocr` and `ocr_timeout`. The chat has a new `/sync` command, and `/add` and `/sync` take `--ocr MODE` and `--ocr-timeout SECONDS`. A mode or timeout that is not valid is refused with an error that names it. The run does not change the setting. `/add` refuses the options for a URL, because OCR does not apply to a crawled page.

A REST client can now cancel an add. Give `POST /api/add` or `POST /api/add/upload` an `add_id`, then send `POST /api/add/cancel` with that id. As with Ctrl+C on `lilbee add`, the cancel removes what the add did not finish: a source with no file indexed is no longer added, and an uploaded file that the server did not index is deleted. The stream of the add ends with a `done` event that carries `cancelled` and `not_added`. A stream that only closes still keeps every source.

`DELETE /api/models/{model}` now removes a model whose ref holds slashes, such as `org/repo/model.gguf`. Before, the route matched one path segment, so the server answered with an error for these refs and a REST client could not remove the model.

When an engine has a second build on disk that is not the build its pin names, `lilbee engine list`, the `/engine list` command and the Engines tab in Settings now mark that build as "differs from the pin". lilbee runs only the pin build, and Remove stays on the row of the other build. Each build in the `builds` field of `GET /api/engines` and the `engine_list` MCP tool has a new field, `differs_from_pin`.

A chat, vision or LLM reranker model now starts with a bounded prompt cache and at most 4 context checkpoints a slot. Before, the engine could keep up to 8 GiB of idle prompts and 32 checkpoints a slot in system memory, which the plan did not count, and a small machine could swap. The cache is half the system memory that the plan leaves free, at most 8 GiB. An embedding model and a cross-encoder reranker keep no prompt cache. If the plan leaves no room, the model still loads without a prompt cache, and lilbee logs one warning. If the plan leaves no room for a checkpoint, lilbee logs one warning that each turn reprocesses the whole conversation, also when the prompt cache stays.

`lilbee placement show`, `lilbee placement preview` and `GET /api/placement` now show how much memory the planner charges for each role. The figure is for one instance of the role. It is card memory for a GPU role and system memory for a CPU role. The API field is `charge_bytes`, and it is `null` when the planner has no charge for the role.

The `[engine]` extra is gone. `pip install lilbee[engine]` no longer installs an engine. Run `pip install --pre lilbee`, then run `lilbee engine install` to get the llama.cpp engine pack. If you skip the command, lilbee installs the pack the first time a model needs it.

If you installed the old extra, the `lilbee-engine-llama-cpp` wheel stays in your environment. The wheel takes precedence over the pack. To use the pack, run `pip uninstall lilbee-engine-llama-cpp`, then run `lilbee engine install`.

No pack exists yet for AMD (ROCm), Vulkan, Linux on ARM, Windows on ARM, or Intel Macs. On those computers, use a bundled build. Or set `LILBEE_LLAMA_SERVER_PATH` to your own `llama-server`.

On a computer with an AMD or Intel GPU, lilbee installs the CPU pack and logs a warning that names the GPU vendor. For a GPU build, use a bundled build or your own `llama-server`.

When every file fails to ingest because the model engine failed, `lilbee ask` now says that nothing is indexed and names the engine error, not "Your library is empty". The library records each such file with a new held state, `retry`, and the next sync tries the file again. An older lilbee cannot read a library that holds a `retry` record. To go back to an older lilbee, run `lilbee sync` with a working engine first, or run `lilbee reset`.

## 0.6.66b481

You can now keep the HTTP server running in the background under your OS launcher: `brew services start lilbee` on macOS, `systemctl --user enable --now lilbee` on Arch, or import `nixosModules.lilbee` on NixOS. All three pin the server to `127.0.0.1:42697`. The daemon helps clients that hit the HTTP REST API (the Obsidian plugin, custom GUIs, scripts). The TUI, `lilbee chat`, the MCP server, and the rest of the CLI cold-start by design and are unaffected.

## Module reorganization

Module reorganization: top-level layout grouped under core/, data/, retrieval/, catalog/, modelhub/, runtime/. External users importing from old `lilbee.X` paths must update to the new package paths. No behavior change.
