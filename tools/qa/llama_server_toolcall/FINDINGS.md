# llama-server native tool-call coverage: findings

Spike to decide whether lilbee can rely on `llama-server --jinja`'s native tool-call
parsing (llama.cpp `common/chat.cpp`) instead of lilbee's hand-rolled `response_parser`
+ per-family schemas, which exist only because llama-cpp-python can't reach that C++
code. Run on a single H200, llama-server built from llama-cpp-python 0.3.23's vendored
llama.cpp (version 9102), `sm_90`. Three passes per model: `tool_choice=auto`, then
`required` on the auto-FAILs, then re-run with the real HF tool-aware template via
`--chat-template-file`.

## Result

| Family | Native verdict | Cause |
|--------|----------------|-------|
| qwen3 | PASS | native parses it |
| qwen3-coder | PASS | native parses it |
| llama3 | PASS | native parses it |
| mistral-nemo | PASS | native parses it |
| gemma4 | PASS | native parses it |
| functionary | PASS | native parses it (was only "partial" in-process) |
| gpt-oss | PASS | native parses Harmony (hand-fixed in lilbee last session; redundant here) |
| glm-air | PASS | native parses glm46 (hand-fixed in lilbee last session; redundant here) |
| hermes | PASS | native parses it once given its real `tool_use` template (the bartowski GGUF's embedded template was stripped) |
| granite | FAIL (real parser gap) | emits a valid `[{"name":...}]` array; native drops it. lilbee `bare_json` fallback recovers it. |
| cohere | FAIL (real parser gap) | emits a valid Command-R action block (prose + `[{"tool_name":...}]<\|END_ACTION\|>`); native 500s parsing it ("Failed to parse input at pos 0"). lilbee's cohere schema recovers it. |
| smollm | FAIL (narrates) | reasoning model explains + puts the call in a ```json fence instead of calling |
| phi4-mini | FAIL (model declines) | "as an AI I can't use external tools". Loads on llama-server (recovered from in-process SIGABRT). |
| internlm2 | FAIL (model declines) | refuses, hallucinates being "developed by OpenAI" |
| glm-4-9b | FAIL (model declines) | refuses. Loads on llama-server (recovered from in-process SIGABRT). |
| lfm2 | FAIL | too small to tool-call reliably |
| ernie | FAIL | too small to tool-call reliably |
| olmo3 | LOAD_FAIL | llama-server's jinja (minja) engine can't parse its chat template |

(hermes + cohere were re-tested with an HF token and their real `tool_use` templates; the
template borrow now picks the `tool_use` entry from the named multi-template list.)

## What this means

- **Native covers every model that actually tool-calls well**: 9 PASS, including gpt-oss
  Harmony, glm46, and hermes, the formats hand-fixed in lilbee's parser. On the
  llama-server path those fixes are redundant. (hermes needs its real `tool_use` template
  via `--chat-template-file`; the bartowski GGUF's embedded template was stripped.)
- **Most "FAILs" are model behavior, not parser deficiency.** phi4-mini, internlm2,
  glm-4-9b, lfm2, ernie either decline or are too small. No parser, lilbee's or native,
  changes that. Several were already exceptions in-process; phi4-mini + glm-4-9b were
  never tool-verified in-process at all (they SIGABRT'd) and now at least *load* on
  llama-server.
- **Two genuine native parser gaps: granite and cohere.** Both emit a valid call that
  native fails on (granite's bare-JSON array is silently dropped; cohere's Command-R
  action block 500s the parse). lilbee's `bare_json` / cohere schema recover both. This is
  the concrete case for keeping lilbee's `response_parser` as a thin native-miss fallback.
- **Template-render limits: olmo3** (and cohere's render emits a warning) hit llama-server's
  minja jinja engine, which is less complete than transformers' jinja that lilbee uses for
  `render_with_hf_template`. A few templates won't render natively.
- **Version caveat**: this is 0.3.23's vendored llama.cpp. Its PEG-based parser
  (`PEG_NATIVE`/`PEG_SIMPLE`/`PEG_GEMMA4`) is a narrower/newer design; a bumped llama.cpp
  may widen native coverage. Re-checking on a newer build is cheap and worthwhile.

## Recommendation (feeds bb-n1bj)

The evidence supports making `llama-server` lilbee's primary chat runtime (not gating it
behind multi-GPU): **native-first parsing, with lilbee's `response_parser` kept as a thin
fallback** for the formats native fails on (granite's bare-JSON array, cohere's Command-R
action block) and `--chat-template-file` for GGUFs whose embedded template is stripped
(hermes) or `render_with_hf_template`-style template borrowing for templates minja can't
render (olmo3, cohere). The decline/too-small models are model-capability limits under
either runtime.

Concretely, a llama-server-everywhere runtime keeps three pieces of lilbee's current work:
1. **Template borrowing** (`--chat-template-file` with the repo's `tool_use` template) for
   stripped/missing GGUF templates -- the same need `render_with_hf_template` serves today.
2. **The response_parser as a native-miss fallback** -- run it on the returned content when
   llama-server yields no structured `tool_calls` (granite) or 500s the parse (cohere).
3. **Message normalization** (dict args, id rewriting, role merging) as request preprocessing.
Everything else in the per-family extraction layer can retire.

Resolved this run: hermes PASSes natively with its real template; cohere is a confirmed
native parse gap (not a template-access problem). Remaining follow-up: consider bumping the
bundled llama.cpp, since its PEG parser and minja are newer/narrower than mainline.

Harness: `probe.py` + `models.py` in this directory. Per-model raw evidence (request,
response, reasoning) is under `results/` (gitignored).
