# Choose the engine of a model

Each model runs on the engine that loads its format. By default, `auto` chooses the
engine; see [How auto chooses an engine](auto.md). To run a model on another engine,
choose the engine for the model, or set a preferred engine.

## The commands

```bash
lilbee engine roles
lilbee engine use chat <model> <engine>
lilbee engine prefer <engine>
```

- `engine roles` shows the engine of each role, with a notice when a role uses
  another engine than the one you chose.
- `engine use ROLE MODEL ENGINE` sets the engine for the model of one role. The role
  is `chat`, `embed`, `rerank` or `vision`.
- `engine prefer ENGINE` sets the engine for every model with no choice of its own.

Give `auto` to `engine use` or `engine prefer` to remove the choice.

## Every surface

| Action | CLI | Chat command | REST | MCP tool |
|---|---|---|---|---|
| Show the engine of each role | `lilbee engine roles` | `/engine roles` | `GET /api/engines/roles` | `engine_roles` |
| Choose the engine of a model | `lilbee engine use ROLE MODEL ENGINE` | `/engine use ROLE MODEL ENGINE` | `PUT /api/engines/roles/{role}` | `engine_use` |
| Set the preferred engine | `lilbee engine prefer ENGINE` | `/engine prefer ENGINE` | `PUT /api/engines/preferred` | `engine_prefer` |

The body of `PUT /api/engines/roles/{role}` holds `model`, `engine` and
`confirm_embed_change`. The body of `PUT /api/engines/preferred` holds `engine` and
`confirm_embed_change`. A role name that is not valid answers 400.

On the HTTP server, the writes need the setting `allow_http_engines`. It is off by
default. With it off, a write answers 409. `GET /api/engines` and `GET /api/engines/roles` need no setting.

The MCP engine tools need the setting `mcp_engines_enabled`. It is off by default.
An MCP server registers the engine tools when it starts, only if the setting is on. If
the server started with the setting off, turn the setting on, then restart the server.
If the setting goes off while the server runs, the tools refuse, and they work again at
once when the setting is on. When the MCP server runs on the HTTP server,
the write tools also need `allow_http_engines`.

## A change to the embed engine

A change to the engine of the embed role can make the vectors in the index
unusable. If the library holds documents, `engine use embed` and `engine prefer`
ask first. On a terminal, answer `y` to confirm. Without a terminal, or with
`--json`, add `--yes` to confirm. In the TUI, the dialog opens on No, so one Enter
keeps the engine. The REST routes and the MCP tools take `confirm_embed_change`.
Without it, the REST write answers 409 with the code `embed_engine_confirm_required`,
the library name and the document count. Send the request again with
`confirm_embed_change` set to `true` to write the change.

## The saved choice

The engine commands write two keys to `config.toml`: `preferred_engine` and
`engine_choices`. `preferred_engine` is `auto` by default. `engine_choices`
maps a model reference to an engine name and is empty by default. Only the
engine commands write these keys. Do not edit them in the settings list.

An `engine_choices` entry for a model wins. `preferred_engine` covers the models
with no entry. If a saved engine cannot run, lilbee uses the default engine and
shows a notice. See [When a choice cannot run](auto.md#when-a-choice-cannot-run).

## Pick the engine in the TUI

Focus a model button on the model bar and press `e`. The engine picker opens for
the model of that role. A role with no model shows a message instead. The Fleet
screen and the model info screen take the same `e` key.

In insert mode the prompt takes the key as text. Press Esc to leave insert mode,
then focus the button and press `e`.

## Pick the engine with the model

The model picker shows each model with its engine, for example
`engine: llama.cpp (auto)`. `(auto)` means lilbee chose the engine. A model in
two formats shows its format after the name.

Press Enter on a model to open the engine picker for it. The saved engine of
the model is highlighted, or `auto` when none is saved, so Enter and Enter
again keep the choice. Press Esc to return to the model list. If only one
engine can load the model file, the picker does not open.

If the engine you pick is not installed, the picker closes and asks to
download it. The role keeps its old model while the engine installs, and the
switch happens when the install ends. If the install fails, the role keeps its
old model and the task center shows the error. If you pick another model for
the role while the engine installs, the role keeps that newer model.

Press `e` on a model, or `ctrl+l` anywhere in the picker, to change the engine
without picking the model. After you start to type a search, `e` is a letter
of the search: a letter typed on the list goes on into the search field. The
`choose_engine_on_pick` setting, on the Engines tab, turns the engine step on
Enter off. It is on by default.

## Choose in Settings

The Engines tab in Settings sits after the Models tab. It holds the preferred
engine and the `choose_engine_on_pick` setting. Its table is in [Install, list and
remove engine builds](install.md#install-and-remove-engines-in-settings).

## A running server

A running server reloads a role when its engine changes.

If the server saves the choice but cannot apply it, the REST write answers 503
with the code `engine_not_applied` and `saved` set to `true`. The choice stays
in `config.toml`. Restart lilbee to use it. A 503 without that code means the
choice was not saved.

## Pull a model with no engine

A pull of a model in a format other than GGUF needs no engine. The download
finishes and lilbee records the model as installed. The engine is needed when
the model loads or when you assign it to a role. If no engine is chosen for the
format, the pull prints a warning after the download. For the reference
`org/repo#openvino`, with no installed engine that loads it:

```
Warning: No installed engine loads the openvino format of org/repo#openvino. Run lilbee engine list (or GET /api/engines) to see the installed engines.
```

If an installed engine loads the format but is not chosen, the warning names
`lilbee engine prefer`. `lilbee use-embedder` prints the same warning after it
downloads the model.

Until an engine loads it, the model does not appear in the chat model pickers.
`lilbee ask` prints the same message as an error when the only chat models
installed have no engine. `lilbee model use` assigns such a model and prints the
message as a warning.

A GGUF pull still needs an engine, because the engine decides which file in the
repository to download.

To pull a model in a format other than GGUF, add the format name to the
reference as `org/repo#format`. Here `openvino` and `mlx` are names of weight
formats, not engines. The format name is `gguf`, `mlx`, `openvino`, `safetensors`
or `xdna`. A reference with no `#format` is GGUF.

An `xdna` reference names an NPU model by its tag, as in `qwen3:0.6b#xdna`. The pull
writes a small record and no weights. The record lists the model as installed, so you can
assign it to a role. Lemonade downloads the model the first time it loads it.
