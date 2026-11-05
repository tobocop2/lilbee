# Settings, secrets and GPU placement

Open this file only when the user asks you to change a setting, a key or the GPU layout.
The full list of settings, with defaults and which surfaces can set each one, is
[docs/settings.md](https://github.com/tobocop2/lilbee/blob/main/docs/settings.md).
`lilbee_settings_list` returns the same list at run time.

## A setting that changes the tool list needs a reconnect

`wiki`, `memory_enabled` and `mcp_sessions_enabled` decide which tools register. The
server decides once, at start. `lilbee_settings_set({"wiki": true})` succeeds and persists,
and the `lilbee_wiki_*` tools are still absent. Ask the user to restart the lilbee MCP
server. Do not report the feature as broken. `lilbee_wiki_status` works either way.

## Turning a feature on does not run it

After the wiki is enabled, its pages exist only when built. See the `lilbee-mcp-wiki` skill.

## Secrets are write-only

`*_api_key`, `hf_token`, `crawl_authorization`, `crawl_password` and `crawl_cookie` are
write-only. `lilbee_settings_list` skips them, `lilbee_settings_get` refuses them, and
`lilbee_settings_set` writes them. Never assume you can read a key back.

## Errors

If `lilbee_settings_set` returns `error`, the value was rejected (unknown key, wrong type,
model not installed). Show the error to the user verbatim.

## GPU placement (multi-GPU machines)

| Tool | Effect |
|---|---|
| `lilbee_get_gpus` | Detected GPUs with free and total memory. |
| `lilbee_get_placement` | Current placement. |
| `lilbee_preview_placement(spec)` | Dry run. Changes nothing. |
| `lilbee_set_placement(spec)` | Saves and applies a placement. Rebuilds the model fleet and interrupts requests in progress. |
| `lilbee_clear_placement` | Returns to automatic placement. Also rebuilds the fleet. |

Preview first, then set, and only when the user asks. On the shared HTTP daemon, set and
clear are refused unless the host enables `allow_http_placement`. The CLI equivalent is
`lilbee placement`.
