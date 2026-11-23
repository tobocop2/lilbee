# Settings, secrets and GPU placement

Open this file only when the user asks you to change a setting, a key or the GPU layout.
The full list of settings, with defaults and which surfaces can set each one, is
[docs/settings.md](https://github.com/tobocop2/lilbee/blob/main/docs/settings.md).
`lilbee_settings_list` returns the same list at run time.

## A setting that changes the tool list needs a reconnect

`wiki`, `memory_enabled`, `mcp_sessions_enabled`, `mcp_profiles_enabled`,
`mcp_presets_enabled` and `mcp_engines_enabled` decide which tools register. The
server decides once, at start. `lilbee_settings_set({"wiki": true})` succeeds and persists,
and the other `lilbee_wiki_*` tools are still absent. Ask the user to restart the lilbee MCP
server. Do not report the feature as broken. `lilbee_wiki_status` and `lilbee_wiki_wipe`
work either way.

## Reading and changing settings

`lilbee_settings_list(group)` lists every writable setting with its value, default, type,
help text, choices and `reindex_required`. Each entry also carries two fields:

- `source` names the layer that supplies the value: `env`, `user`, `profile`, `built_in`
  or `auto`. Treat an unknown `source` as no source.
- `advanced: true` marks a setting that the TUI folds into a collapsed section. It is
  listed and settable like any other.

`lilbee_settings_get(key)` returns one setting with the same metadata. `lilbee_settings_set`
writes settings in one atomic update. `lilbee_settings_reset(keys)` removes your value for
each key, so the setting falls back to its `LILBEE_*` variable, then the profile, then the
built-in value. The default shown by the list is that fallback. `warnings` in the result
carries a notice, as it does for `lilbee_settings_set`.

## Profiles

A profile is a named bundle of settings for a kind of library. The profile tools appear
only when `mcp_profiles_enabled` is on.

- `lilbee_profile_list` returns the project's active profile and every profile. The active
  profile has a `status` (`current`, `changed`, `missing` or `broken`) and `changes`: each
  profile setting you set, with `yours`, the `profile_value` and `profile_source` it falls
  back to, and its `effect`. Each profile has `credit`, `tested_on` and its values. A
  broken file has an `error`.
- `lilbee_profile_show(name)` returns one profile and the `diff` that applying it makes.
  Each change has an `effect` (`reindex`, `new_files_only` or `now`). The result also
  lists your values that the profile keeps. Names match loosely: `court-filings` finds
  "Court filings".
- `lilbee_profile_apply(name, reindex=false)` makes a profile the project's. Your own
  values and `LILBEE_*` variables stay. When `reindex_required` is true, rebuild before
  you trust search results. With `reindex=true` the call also rebuilds when the profile
  needs it or a file is held out, and blocks until done. `rebuild` is the sync result, or
  null when no rebuild was needed. `rebuild_error` names a refused rebuild.
- `lilbee_profile_manage(action, name, ...)` changes profile files. Actions: `new`,
  `save` (the project's settings as `name`, then switch to it), `update`, `discard`,
  `duplicate` and `rename` (to `new_name`), `delete`, `export` (returns `content`), and
  `import` and `validate` (a file's `content` and its `filename`, not a path). `discard`
  returns `dropped`, `reindex_required` and `warnings`. With `reindex` true it rebuilds
  when needed and returns `rebuild`, or `rebuild_error` if refused. `folder` is `global`
  (all projects, the default) or `project`.
- `lilbee_analyze(directory, apply, save, target)` reads the indexed documents, or an
  absolute `directory` on the server, and recommends a profile. The report lists file
  types, scanned pages, tables, lengths and languages, and the `recommendation` with its
  `changes`. It saves nothing unless `apply` (save and switch) or `save` (a name, no
  switch) is set. `target` (`project` or `global`) needs one of them. Over
  `analyze_max_files` documents it reads an even sample. It runs no model, so it does not
  pin the embedder, but a large corpus takes a while.
- `lilbee_analyze_dismiss` hides the analyze tip for this project. It returns `analyzed`,
  `tip_dismissed` and `tip_shows`.

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

## Server presets

These tools register only when the user turns on `mcp_presets_enabled` (off by default).

| Tool | Use |
|---|---|
| `lilbee_preset_list()` | The stored server presets: name, URL, roles, rerank wire, memory reserve, a `reserve_note` when a remote preset has a reserve, and `has_key`. A key is never returned. |
| `lilbee_preset_add(name, url, roles, api_key, rerank_wire, memory_reserve_gb)` | Add a preset for the roles `embed`, `rerank`, `chat` or `vision`. `rerank_wire` is `v1` or `tei`. |
| `lilbee_preset_edit(name, url, roles, api_key, rerank_wire, memory_reserve_gb, clear_reserve, clear_key)` | Change a preset. Give only what changes. `clear_reserve` unsets the memory reserve. `clear_key` removes the stored key and stands alone. |
| `lilbee_preset_remove(name)` | Remove a preset. |

On the shared HTTP daemon, add, edit and remove are refused unless the host enables
`allow_http_engines` (a preset sets where requests and API keys go). List always works.
Change presets only when the user asks.

## Engine choice

These tools register only when the user turns on `mcp_engines_enabled` (off by default).

| Tool | Use |
|---|---|
| `lilbee_engine_list()` | The engine builds installed beside the models (name, version, backend, `not_run_on`: the devices the engine has not run on), `damaged`, each engine folder that lilbee cannot use (`folder` and an `error` that names the fix), `not_installed`, the engines with no installed build, and `engines_folder`, the folder that holds the builds. |
| `lilbee_engine_remove(engine, version, backend)` | Remove one installed build, as `list` shows it. Refused for the bundled engine and a build in use. |
| `lilbee_engine_roles()` | The engine of each role: model, format, engine and the notice when the role fell back or has no engine. |
| `lilbee_engine_use(role, model, engine)` | Choose the engine for the model of a role. `engine` `auto` removes the choice. |
| `lilbee_engine_prefer(engine)` | Set the engine for every model with no choice of its own. `auto` picks by model format. |
| `lilbee_engine_install(engine)` | Install an engine that is not bundled, from its pack in the packs folder. The host must pass the engine's gate. The result carries `engines_folder`, the folder that holds the builds. |

On the shared HTTP daemon, remove, use, prefer and install are refused unless the host
enables `allow_http_engines`. List and roles always work. Change the engine only when the
user asks.
