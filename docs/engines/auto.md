# How auto chooses an engine

`auto` is the default engine setting. With `auto`, lilbee chooses the engine for each
role (chat, embed, rerank, vision) from the format of the model and the hardware of
the computer. See [the engine table](README.md) for the engines.

## The order of the choice

lilbee looks at three things for the model of each role, in this order:

1. The choice for the model. `lilbee engine use` writes it.
2. The preferred engine. `lilbee engine prefer` writes it. It covers every model with
   no choice of its own.
3. The default engine.

If the choice for the model and the preferred engine are both `auto`, the default
engine runs the model.

## The default engine

lilbee chooses the default from two engines only:

- On Apple Silicon (macOS on arm64), `mlx` comes first, if it is installed.
  `llama.cpp` comes second.
- On every other computer, `llama.cpp` is the only one.

lilbee takes the first engine of that order that obeys all these rules:

- The engine is installed.
- The engine loads the format of the model.
- The engine serves the role.
- lilbee has a recorded run of the engine on every device that the engine supports,
  not only on the devices of this computer. If one device has no record, lilbee does
  not choose the engine on any computer. llama.cpp needs no record.
- This computer passes the check of the engine, with no part that lilbee cannot read.
  For example, if lilbee cannot read the macOS version, it does not choose `mlx`.

`auto` never chooses an opt-in engine: `ik_llama.cpp`, `mistral.rs`, `tei`,
`openvino`, `ovms` and `lemonade` run only when you choose them.

## The default for each format

| Format | Default engine |
|---|---|
| `gguf` | `llama.cpp` |
| `mlx` | `mlx`, on Apple Silicon with macOS 14 or newer |
| `safetensors` | none |
| `openvino` | none |
| `xdna` | none |

If a format has no default engine, the role has no engine until you choose one. lilbee
shows one of four notices. `lilbee engine roles` starts each notice with
`For <role>: `, for example `For chat: `.

- If an installed engine loads the format, serves the role and can run on this computer,
  the notice is, for example,
  `No engine is chosen by default for the safetensors format of org/repo. Choose tei with lilbee engine use for this model, or with lilbee engine prefer for every model without its own choice.`
  With no role, the notice offers only `lilbee engine prefer`.
- If an installed engine loads the format but no installed engine serves the role, the
  notice is, for example,
  `No installed engine serves the safetensors format of org/repo for rerank. tei serves it. Run lilbee engine install tei.`
  If the engine cannot be installed, the notice gives the reason in place of the install
  command.
- If an installed engine loads the format and serves the role, but this computer fails the
  check of that engine, the notice names the engine and gives the reason from the check. It
  is, for example,
  `Engine ik_llama.cpp loads the gguf format of org/repo but cannot run on this host. Xeon lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON.`
  The notice names every such engine. It does not tell you to install an engine.
- If no installed engine loads the format, the notice is, for example,
  `No installed engine loads the safetensors format of org/repo. Run lilbee engine list to see the installed engines.`

The text `of org/repo` shows only when lilbee knows the name of the model.

## The llama.cpp build for each kind of hardware

The hardware also decides which llama.cpp build installs from a pack:

| Computer | llama.cpp build |
|---|---|
| macOS on Apple Silicon | Metal |
| Linux or Windows on x86_64, with an NVIDIA card | CUDA, if the driver offers CUDA 12.4 and the card has compute capability 5.0 or newer |
| Linux on x86_64, with a supported AMD card | ROCm |
| Linux or Windows on x86_64, with AMD or Intel graphics and a Vulkan loader | Vulkan |
| any other computer | CPU |

lilbee tries the rows from the top. A computer with an NVIDIA card that fails the CUDA
check gets the next build that fits. The ROCm build needs `libnuma` and a card of one
of the gfx targets that the pack carries.

## When a choice cannot run

A choice that cannot run does not stop lilbee. The role uses the default engine, and
lilbee shows a notice. `lilbee engine roles`, the model bar and the Fleet screen show
the notice.

- If the choice names an engine that is not installed, the notice is, for example,
  `Engine ik_llama.cpp is not installed. Using llama.cpp. Run lilbee engine list to see the installed engines.`
- If the choice names an engine that this computer cannot run, the notice says why
  and names the engine in use.
- If the choice for a model names an installed engine that cannot load its format,
  the role has no engine. The notice is, for example,
  `Engine llama.cpp for chat cannot load the mlx format of org/repo#mlx. Choose an engine that loads mlx.`
- If the preferred engine loads the format but does not serve the role, the role
  uses the default engine with no notice.
