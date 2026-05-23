# Local models with opencode

`lilbee launch opencode` wires [opencode](https://opencode.ai) to your local
lilbee chat models so an agent can search your library, read the results, and
answer with citations, all on your machine. opencode talks to lilbee over the
OpenAI-compatible API and calls the `lilbee_search` tool through MCP, so a model
only works here if it can emit tool calls in a format lilbee can read back.

This page lists the model families we have driven end to end through opencode:
opencode sends a prompt, the model calls `lilbee_search`, opencode runs the tool
against an indexed workspace, and the model writes a final answer from the
results. Each family below completed that loop.

## Verified families

| Family | Example model | Notes |
|--------|---------------|-------|
| Qwen3 | `Qwen/Qwen3-4B-GGUF` | |
| Qwen3-Coder | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | sparse-MoE coder variant |
| Llama 3.1 | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | |
| Hermes 3 | `bartowski/Hermes-3-Llama-3.1-8B-GGUF` | |
| Mistral-Nemo | `bartowski/Mistral-Nemo-Instruct-2407-GGUF` | |
| Gemma | `unsloth/gemma-4-E2B-it-GGUF` | |
| SmolLM3 | `bartowski/HuggingFaceTB_SmolLM3-3B-GGUF` | |

How reliably a model reaches for a tool depends on the model and its size, not
on lilbee: a smaller model may answer some prompts directly instead of
searching. When the model does call `lilbee_search`, lilbee reads the call and
opencode runs it.

## Not yet supported

A few families don't work through opencode today. The reason is either the
model itself or the bundled runtime, not the lilbee tool plumbing:

| Family | Why |
|--------|-----|
| Command-R7B | this GGUF's context tops out at 8K tokens, smaller than opencode's system prompt plus tool schema |
| Phi-4-mini | the bundled llama.cpp aborts while loading this GGUF |
| Functionary v3 | the functionary preset needs a tokenizer attribute the bundled llama.cpp build doesn't expose |
| GLM-4-9B-chat | the older GLM-4 chat template doesn't advertise tools in a form opencode can pick up |
| InternLM2.5 | describes the search it would run instead of emitting a tool call |
| OLMo-2 | not trained for tool calling |
| ERNIE-4.5 0.3B, LFM2 1.2B | too small to call tools reliably |
| GLM-4.5-Air | 60 GB; the download needs xet, which lilbee turns off so pull progress stays visible |

## How this was measured

The harness lives in `tools/qa/opencode/`. For each family it pulls the model,
indexes a small fixture set, launches the real opencode binary against a
lilbee server, and sends three prompts that require a `lilbee_search` call. A
family passes only when opencode dispatches the tool (not when the model prints
the call as text) and the lilbee server logs a clean chat completion with no
worker errors.
