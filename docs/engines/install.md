# Install, list and remove engine builds

An engine build is one installed copy of an engine: its name, its version and its
backend. lilbee keeps engine builds in the engines folder, beside the models folder.
See [the engine table](README.md) for the engines.

## Every surface

| Action | CLI | Chat command | REST | MCP tool |
|---|---|---|---|---|
| List the builds | `lilbee engine list` | `/engine list` | `GET /api/engines` | `engine_list` |
| Install an engine | `lilbee engine install ENGINE` | `/engine install ENGINE` | `POST /api/engines/install` | `engine_install` |
| Remove a build | `lilbee engine remove ENGINE VERSION BACKEND` | `/engine remove ENGINE VERSION BACKEND` | `DELETE /api/engines/{engine}/{version}/{backend}` | `engine_remove` |

The body of `POST /api/engines/install` holds `engine`. The REST route and the MCP
tool take only an engine name. They do not take a pack file.

On the HTTP server, the install and the removal need the setting
`allow_http_engines`. The MCP engine tools need the setting `mcp_engines_enabled`.
See [Choose the engine of a model](choose.md#every-surface).

## List the builds

Run `lilbee engine list` to see the engine builds installed beside the models.
If none is installed, it prints `No engine builds are installed.` The command
and `lilbee engine install` print the line `Engines folder:` with its path. The
`/engine list` chat command ends with the same line.

The list shows only the engines that this computer can run. It prints one line
`<engine> not installed` for each of those engines with no installed build.

With `--json`, the CLI, the MCP tool and the REST route return the same five fields:

- `builds`: the installed builds.
- `damaged`: each engine folder that lilbee cannot use, as its `folder` and an `error`.
- `not_installed`: the engines with no installed build.
- `experimental`: each engine with devices that have no record, and those devices.
- `engines_folder`: the path that the `Engines folder:` line shows.

Each build in `builds` has a `not_run_on` field with the devices of its own cell.
It also has a `differs_from_pin` field. The field is `true` for a build that is not the build the pin names for this computer. lilbee runs only the pin build, so a build that differs stays on disk and runs nothing. `engine list` ends the line of that build with `, differs from the pin`. The Engines tab shows the same words, and keeps Remove on that row.

If lilbee has no record of an engine on a device, the engine is experimental
there and the line ends with `, experimental: not run on <device>`. The list then
prints a note that says what experimental means and where to report a result.
The install output prints the same words.

If the build of an engine differs from the pin, `engine list` prints one line:
the engine is not installed, and the line names the build that differs.

## The engines folder

Every data directory shares the engines folder, so `LILBEE_DATA` does not move it.
To move it, set `LILBEE_MODELS_DIR`. That variable also moves the models folder and
the Hugging Face cache. `LILBEE_ENGINE_DIR` moves the machine engine slot. To
isolate a run fully, follow the steps in [What the data directory
holds](../usage.md#what-the-data-directory-holds).

## Install an engine

```bash
lilbee engine install <engine>
```

With no engine name, the command installs `llama.cpp`. The install checks that this
computer can run the engine, then checks the digest and the licence of the pack. If
the computer fails the check, the command names the feature it lacks and installs
nothing.

Only the llama.cpp pack installs on first use, when a model needs it. Every other
engine needs this command.

An engine that is not bundled needs a pack. These packs are published:

| Engine | Packs |
|---|---|
| `llama.cpp` | Linux x86_64 (CPU, CUDA, ROCm, Vulkan), Windows x86_64 (CPU, CUDA, Vulkan), macOS 14 on Apple Silicon (Metal) |
| `ik_llama.cpp` | Linux x86_64 (CPU, CUDA) |
| `mistral.rs` | Linux x86_64 (CPU, CUDA on Ampere to Blackwell cards), Windows x86_64 (CPU), macOS 15 on Apple Silicon (Metal) |
| `tei` | Linux x86_64 (CPU, and CUDA for each card generation from compute capability 7.5), macOS 14 on Apple Silicon (Metal) |
| `mlx` | macOS 14 on Apple Silicon (Metal) |

No pack of `openvino`, `lemonade` or `ovms` is published yet, so those engines
refuse to install and print the no-pack text. A computer with no pack for its
platform gets the same text. Nothing is installed, and llama.cpp stays in use.

If the install refuses an engine it knows, for example because no pack of the
engine is published, the REST route answers 422 with the code
`engine_install_refused`. The MCP tool returns the same `code`, and
`lilbee --json engine install` prints it. An unknown engine name has no code.

The JSON answer of `lilbee --json engine install`, the REST route and the MCP tool
carry `engines_folder`, the same field as the list.

During the download, the command line shows a progress bar on stderr, and the
chat screen shows the bytes in the task bar. Each failed attempt prints its
cause and its number. The MCP tool and the REST route return one result at the
end and show no progress.

### How each llama.cpp pack reports memory

After a launch, lilbee compares the memory that the engine used with its own
estimate. An engine that serves `GET /memory` answers this directly. Upstream
llama.cpp has no `GET /memory` (upstream pull request 26130 is open). Every
llama.cpp pack is built from an upstream release, so today all 8 pack cells
read the memory from the startup log of the engine. No pack serves
`GET /memory`. A `llama-server` that you build from the lilbee source pin
serves `GET /memory`.

<!-- readback-cells:start -->
| Pack cell | Memory readback |
|---|---|
| `windows-x86_64-cpu` | startup log |
| `windows-x86_64-cuda` | startup log |
| `macos-arm64-metal` | startup log |
| `linux-x86_64-cuda` | startup log |
| `linux-x86_64-cpu` | startup log |
| `windows-x86_64-vulkan` | startup log |
| `linux-x86_64-vulkan` | startup log |
| `linux-x86_64-rocm` | startup log |
<!-- readback-cells:end -->

### A llama-server that you supply

If lilbee finds a `llama-server` of its own, it runs that one and leaves the
llama.cpp pack unused. lilbee looks in this order: the path in
`LILBEE_LLAMA_SERVER_PATH`, the `lilbee-engine-llama-cpp` wheel, `PATH`, and then
the pack. To run the pack instead of the wheel, remove the wheel with
`pip uninstall lilbee-engine-llama-cpp`, then run `lilbee engine install`.

### Install from a local pack

To install with no network, copy the pack into the `packs` folder of the engines
folder. Name the file `<engine>-<version>-<backend>.whl`. Then run:

```bash
lilbee engine install <engine>
```

If the pack file does not match the pin of this lilbee version, lilbee installs nothing and names the file. Copy the pack file again from the lilbee release that matches this version, or remove it from the `packs` folder to download it. A router archive in the `packs` folder and a pack file given with `--wheel` follow the same rule.

To install from a pack file in another place, add `--wheel`. This works only with a
pack file whose digest matches the pin of the engine:

```bash
lilbee engine install <engine> --wheel /path/to/pack.whl
```

### The router

Every engine starts behind the llama-swap router. `lilbee engine install` also
installs the router from its pinned release (MIT licence, digest checked) into the
`router` folder of the engines folder, so an install without llama.cpp can start its
engine. To install the router with no network, copy the release archive into the
`packs` folder (for example `llama-swap_272_linux_amd64.tar.gz`). The router also
installs on first use. A `llama-swap` on PATH, or one inside the llama.cpp pack, is
used when no router is installed.
A router older than v259 still starts the engine. It cannot copy the output of an engine
that writes no log of its own (TEI, MLX, OVMS and others) into the router log. In that
case lilbee writes one warning to the lilbee log and lists it in `warnings` (code
`router_outdated`) in the `GET /api/health` answer. The warning names the version found.
The terminal interface shows no notice for it. To clear it, replace the router file with
a newer release. Download a release from the
[llama-swap releases](https://github.com/mostlygeek/llama-swap/releases), or run
`go install github.com/mostlygeek/llama-swap@latest` and copy the `llama-swap` file from
your Go bin folder to the path the warning names. Then restart lilbee. The warning stays
in the health answer until lilbee starts the engine again.

### Engines you install yourself

Lemonade has no pack, so you install it yourself. Unpack the `lemonade-embeddable`
archive from the Lemonade releases, then put the folder with `lemond` on `PATH`. On
Linux, the packages from the [Lemonade install
guide](https://lemonade-server.ai/docs/guide/install/) also install `lemond`. On
Windows, use the archive: the Windows installer has no `lemond`. lilbee finds
`lemond` on `PATH`, and `lilbee engine list` no longer lists lemonade as not
installed.

On the first model load, Lemonade downloads FastFlowLM for the `flm` recipe. The NPU kernels
of FastFlowLM are not open source. They are free only for non-commercial use and for companies
below a revenue limit. See the
[terms](https://github.com/ROCm/FastFlowLM/blob/v1.0.7/TERMS.md). lilbee does not ship
Lemonade or FastFlowLM, so you accept these terms yourself. The AMD `ryzenai-server` has an
end user licence agreement (EULA) that bars distribution of the software (section 3).
lilbee does not ship it for that reason.

OpenVINO Model Server (`ovms`) has no pack, so you install it yourself. Follow the
[OpenVINO Model Server install guide](https://docs.openvino.ai/2026/model-server/ovms_docs_deploying_server_baremetal.html)
for version 2026.4.1 on Linux x86_64. The guide downloads the 2026.4.0 archive. Download
`ovms_ubuntu24_2026.4.1_python_off.tar.gz` from the
[2026.4.1 release](https://github.com/openvinotoolkit/model_server/releases/tag/v2026.4.1)
instead, then put `ovms/bin` on `PATH`. lilbee finds `ovms` there, and `lilbee engine list`
no longer lists ovms as not installed. When `ovms/lib` sits next to `ovms/bin`, lilbee adds it to
`LD_LIBRARY_PATH` for the server it starts. Without that folder on the variable, `ovms` fails
to load `libtbb`. If your `ovms` has another layout, set `LD_LIBRARY_PATH` yourself.

lilbee records the engine version as 2026.4.1, the version it was tested with, whatever
version `ovms` is.

## Remove a build

```bash
lilbee engine remove <engine> <version> <backend>
```

Use the three values that `engine list` shows. The removal frees the space of the
build.

Removing the engine build that the embedder runs on asks first, because the next
start runs the canary check on the index. `engine remove` takes `--yes`, the same
as `engine use`; see [A change to the embed engine](choose.md#a-change-to-the-embed-engine).
The REST route reads `confirm_embed_change` from the query. The MCP tool takes
`confirm_embed_change`. The `/engine remove` chat command and the Engines tab in
Settings ask the same question.

## Damaged engine folders

A damaged engine folder does not hide the other engines. `engine list`, the
`/engine list` chat command and the Engines tab in Settings show one line for each
damaged folder. The line names the folder and the fix: correct the permissions of
the folder, or remove it and run `lilbee engine install <engine>`. The line leaves
out the install if this computer cannot install the engine. If a read fails for a
cause other than the permissions, the line names that cause, and lilbee leaves the
folder as it is. `engine remove` clears a damaged build that it can read, also one
with a folder in the place of its record. A command that skips a folder in the
cleanup of old builds shows a warning with the folder.

## Install and remove engines in Settings

The Engines tab in Settings sits after the Models tab. It holds the preferred
engine, the `choose_engine_on_pick` setting and a table with one row per engine: the
version, the backend, the size, the state, and the roles that use it. Move to a row
and press Enter to install the engine, or to remove an installed build. Remove asks
first, and names the roles when a role uses the engine. An install runs as a task in
the task bar.
