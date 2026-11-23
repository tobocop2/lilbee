# Engines

An engine is the server that loads a model and runs it. lilbee starts every engine
behind the llama-swap router. The format of a model decides which engines can load it.

These pages explain the engines:

- This page lists the engines that lilbee runs.
- [How auto chooses an engine](auto.md) explains the default choice for each role and
  each kind of hardware.
- [Choose the engine of a model](choose.md) shows the CLI, the TUI, REST and MCP.
- [Install, list and remove engine builds](install.md) shows how to get an engine and
  how to remove it.

## The engines

| Engine | Formats | Roles | Hardware | How you get it | Chosen by `auto` |
|---|---|---|---|---|---|
| `llama.cpp` | `gguf` | chat, embed, rerank, vision | CPU, NVIDIA (CUDA), AMD (ROCm on Linux), AMD and Intel graphics (Vulkan), Apple Silicon (Metal) | bundled; a pack installs on first use | yes |
| `ik_llama.cpp` | `gguf` | chat, embed | Linux x86_64: CPU, and NVIDIA (CUDA) | pack | no, opt-in |
| `mistral.rs` | `gguf`, `safetensors` | chat, embed, vision | Linux x86_64: CPU; macOS on Apple Silicon: Metal | pack | no, opt-in |
| `tei` | `safetensors` | embed, rerank | Linux x86_64: CPU, and NVIDIA (CUDA); macOS on Apple Silicon: Metal | pack | no, opt-in |
| `mlx` | `mlx` | chat, embed, rerank | Apple Silicon, macOS 14 or newer | no pack is published yet | yes, on Apple Silicon |
| `openvino` | `gguf` | chat | Linux x86_64 with an Intel GPU or an Intel NPU | no pack is published yet | no, opt-in |
| `ovms` | `openvino` | chat, embed, rerank | Linux x86_64: CPU, Intel GPU, Intel NPU | you install it | no, opt-in |
| `lemonade` | `xdna` | chat, embed | AMD XDNA2 NPU, Linux or Windows x86_64 | you install it | no, opt-in |

`auto` chooses only `llama.cpp` and `mlx`. Every other engine is opt-in: it runs only
when you choose it with `lilbee engine use` or `lilbee engine prefer`. See [How auto
chooses an engine](auto.md).

`lilbee engine list` shows only the engines that this computer can run.

## Notes on each engine

llama.cpp is the bundled engine. The standalone executables and the Python package
both carry it. If no `llama-server` is on the computer, the llama.cpp pack installs
the first time a model needs it.

ik_llama.cpp needs a CPU with AVX2 (x86_64) or NEON (ARM). The CUDA build needs Linux
x86_64, a driver that offers CUDA 12.4 and a card of compute capability 7.5 or newer.
Other hosts get the CPU build.

mistral.rs serves chat, embed and vision, but not rerank. For embed, it does not load
GGUF models. On Linux it runs on the CPU, because lilbee offers no CUDA build of
mistral.rs. Windows has no pack.

The mistral.rs packs include software developed by the OpenSSL Project for use in the
OpenSSL Toolkit (http://www.openssl.org/) and cryptographic software written by Eric Young
(eay@cryptsoft.com).

TEI (text-embeddings-inference) serves embed and rerank only. On Linux x86_64 it
uses CUDA on a card of compute capability 7.5 or newer, and the CPU on other hosts.
TEI does not run on Windows.

mlx (mlx-lm) needs Apple Silicon and macOS 14 or newer. It serves chat, embed and rerank.
Chat runs on mlx-lm's server. Embed and rerank run on lilbee's own server in the same pack,
for Qwen3-Embedding and Qwen3-Reranker models only. lilbee refuses any other `#mlx` model for
the embedding or the reranker role, and names the supported family. lilbee loads `#mlx` models
with it and does not offer it on an older macOS. No pack of mlx is published yet.

OpenVINO serves chat on the CPU, an Intel GPU or an Intel NPU. lilbee offers it only on
Linux x86_64, on a computer with an Intel GPU or an Intel NPU. On the NPU, a role runs
with one slot and a context of at most 512 tokens. The OpenVINO pack carries the
OpenVINO runtime, but not the Intel GPU compute runtime or the Intel NPU driver.
Install those from your distribution or from Intel. The OpenVINO runtime in the pack
is under Apache-2.0, so no licence step is needed. lilbee does not publish the
OpenVINO pack yet, so `lilbee engine install openvino` reports that no pack is
published.

OpenVINO Model Server (`ovms`) serves chat, embed and rerank of OpenVINO IR models
(`org/repo#openvino`) on the CPU, an Intel GPU or an Intel NPU. It is the one engine
that offers embed and rerank on the Intel NPU. lilbee offers the NPU target only on a
computer with an Intel NPU. lilbee has no recorded run of OpenVINO Model Server on any
device. To install it, see [Install, list and remove engine
builds](install.md#engines-you-install-yourself).

Lemonade serves chat and embed models on an AMD XDNA2 NPU, through its `flm` recipe. It
does not serve rerank, and it does not run GGUF models. On Windows lilbee cannot read the
NPU. lilbee has no recorded run of Lemonade on an NPU. To install it, see [Install, list
and remove engine builds](install.md#engines-you-install-yourself).

## Experimental engines

NPU engines and NPU servers are experimental. lilbee has no recorded run on NPU
hardware. For an engine with a device that has no record, such as an NPU engine
(Lemonade, OpenVINO or OpenVINO Model Server), `lilbee engine list` ends the line with
", experimental: not run on" and the device names, for example ", experimental: not run
on Intel Arc graphics and Intel NPU". `auto` never chooses an experimental engine,
except llama.cpp. GenieX and FastFlowLM are servers and do not appear in
`lilbee engine list`. The tests use fixtures and fake servers. If you run lilbee on an
NPU, report the result in the [GitHub issues](https://github.com/tobocop2/lilbee/issues).
