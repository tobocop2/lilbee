# Server presets

A server preset names an OpenAI-compatible server that you run. A preset holds a name, a base URL, the roles the server serves, an optional API key, and the rerank wire. lilbee stores presets in `config.toml` under `server_presets`.

Use a preset to send the embed, rerank, chat or vision work to a server such as oMLX, vMLX, vllm-mlx, mlx-lm, TEI, Lemonade, GenieX, FastFlowLM, mistral.rs or Infinity.

## Use a preset

Name the preset as the prefix of a model ref. For the preset `omlx`, the ref `omlx/my-model` sends the request to the URL of `omlx`. The preset must serve the role of the request. If it does not, lilbee refuses the request and names the missing role.

A preset name uses lower-case letters, digits, `-` and `_`. It starts with a letter or a digit. A provider prefix such as `openai` is not a valid name.

## The roles

A preset serves one or more of these roles: `embed`, `rerank`, `chat`, `vision`. Name each role once.

## The servers

A preset works with a server that offers the OpenAI-compatible routes for the roles you give it. These servers are the targets for presets: oMLX, vMLX, vllm-mlx, mlx-lm, TEI, Lemonade, GenieX, FastFlowLM, mistral.rs and Infinity.

lilbee ran its output check against TEI, Lemonade and mistral.rs. The other servers are not run or not verified yet. See [Preset results](preset-results.md) for each server, role and result. Check which routes your server offers before you give it a role. Use `tei` as the rerank wire for TEI. For every other server, use the default `v1` only if the server documents `POST /v1/rerank`.

## The rerank wire

The `rerank_wire` field sets the shape of the rerank request.

| Wire | Request | Servers |
|---|---|---|
| `v1` | `POST /v1/rerank` with `documents` | Jina, Cohere, llama-server, Lemonade |
| `tei` | `POST /rerank` with `texts` | text-embeddings-inference (TEI) |

The default is `v1`. A wire on a preset without the rerank role is stored. It applies when you add the role.

## Manage presets

Every surface uses the same rules and the same error text. A change on one surface shows on all the others.

### CLI

```bash
lilbee preset list
lilbee preset add omlx http://127.0.0.1:8000 --roles chat,embed
lilbee preset add tei http://127.0.0.1:8080 --roles rerank --wire tei --key-stdin
lilbee preset edit tei --wire v1
lilbee preset edit tei --clear-key
lilbee preset remove tei --yes
```

`lilbee preset list` prints a table of name, URL, roles, wire and key state. The wire column shows `-` for a preset without the rerank role. The key column shows `set` or `none`. With `--json`, the command prints `has_key` and never a key.

`lilbee preset add` takes a name, a URL and `--roles`. Add `--wire v1` or `--wire tei` to set the rerank wire.

`lilbee preset edit` changes only what you give: `--url`, `--roles`, `--wire`, `--key-stdin` or `--clear-key`. A call with no change is refused. `--clear-key` removes the stored key and must stand alone.

`lilbee preset remove` asks for confirmation. Add `--yes` to skip the question.

### Enter an API key

There is no `--key` flag. A key on the command line shows in the shell history and the process list.

`--key-stdin` reads the key from one line of stdin. On a terminal it shows a hidden prompt. Leave the key out of an edit to keep the stored key.

If you change the URL to a different scheme, host or port, lilbee refuses the edit unless you give a new key or clear the key first. A stored key never follows the URL to a new server.

### Slash command

In the chat, run `/preset list`, `/preset add NAME URL --roles ROLES`, `/preset edit NAME` or `/preset remove NAME`. `/preset add` accepts `--wire`. `/preset edit NAME` with no flag opens a dialog. In the dialog you enter or clear the key. `/preset edit` also accepts `--url`, `--roles`, `--wire` and `--clear-key`.

The slash command refuses `--key`. Give the key in the dialog. No `/preset` line enters the chat input history, whatever its flags or spacing. The Presets tab in the settings screen shows the same list.

### REST

| Route | Action |
|---|---|
| `GET /api/presets` | List the presets. |
| `POST /api/presets` | Add a preset. Returns 201. |
| `PUT /api/presets/{name}` | Change a preset. |
| `DELETE /api/presets/{name}` | Remove a preset. |

The body fields are `name`, `url`, `roles`, `api_key` and `rerank_wire`. An edit also takes `clear_key`. The key is write only. The response shows `has_key`.

The errors are 404 for an unknown name, 409 for a name that exists, and 422 for any other refusal.

### MCP

The tools are `preset_list`, `preset_add`, `preset_edit` and `preset_remove`. They are listed in the MCP skill as `lilbee_preset_*`.

The tools are behind the setting `mcp_presets_enabled`. It is off by default. Turn it on, then restart the MCP server. The server decides which tools exist when it starts.

## The HTTP gate

A preset decides where requests and API keys go. So the HTTP server refuses to change presets unless the setting `allow_http_engines` is on. It is off by default.

When it is off, `POST`, `PUT` and `DELETE` on `/api/presets` return 409. The MCP tools `preset_add`, `preset_edit` and `preset_remove` return an error when the MCP server runs inside the HTTP server. `GET /api/presets` and `preset_list` work in both cases.

The gate does not apply to the CLI, the slash command or the settings screen. Turn on `allow_http_engines` only for a deployment that you alone use.

## What lilbee checks and when

lilbee checks the URL when you save a preset. It refuses a URL that is not well formed and an address in a blocked network. A host name with no DNS answer is saved.

lilbee checks the address again before the request. Before the request, it resolves the host name and refuses the request if any address is blocked or if the host has no DNS answer. The SDK resolves the host name once more when it connects. A DNS change between the check and the connection is not caught.

## Related

- [Usage](usage.md) lists the slash commands and the CLI commands.
- [Settings reference](settings.md) lists `server_presets`, `allow_http_engines` and `mcp_presets_enabled`.
