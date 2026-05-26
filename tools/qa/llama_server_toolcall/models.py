"""Model roster for the llama-server native tool-call probe.

Each entry names a GGUF and, when the GGUF ships a stripped or missing chat
template, the HF repo whose ``chat_template.jinja`` / ``tokenizer_config.json``
``llama-server --jinja`` should borrow instead (passed as --chat-template-file).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    family: str
    repo: str
    # File path within the repo snapshot that llama-server loads. For a split
    # GGUF this is the first shard; allow_patterns must pull every shard.
    gguf: str
    allow_patterns: tuple[str, ...]
    # HF repo to borrow a chat template from when the GGUF has none/stripped.
    template_repo: str | None = None
    # Skip on a single-GPU box (needs multi-GPU); kept for the later fleet spike.
    multi_gpu_only: bool = False
    notes: str = ""
    # The in-process verdict, for the comparison column in the report.
    inprocess: str = ""


def _q4(repo: str, fname: str, **kw: object) -> ModelSpec:
    return ModelSpec(repo=repo, gguf=fname, allow_patterns=(fname,), **kw)  # type: ignore[arg-type]


ROSTER: tuple[ModelSpec, ...] = (
    # --- verified in-process via lilbee's response_parser ---
    _q4("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", family="qwen3", inprocess="pass"),
    _q4(
        "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        family="qwen3-coder",
        inprocess="pass",
    ),
    _q4(
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        family="llama3",
        inprocess="pass",
    ),
    _q4(
        "bartowski/Hermes-3-Llama-3.1-8B-GGUF",
        "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf",
        family="hermes",
        inprocess="pass",
    ),
    _q4(
        "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF",
        "HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf",
        family="smollm",
        inprocess="pass",
    ),
    _q4(
        "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        family="mistral-nemo",
        inprocess="pass",
    ),
    _q4("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_K_M.gguf", family="gemma4", inprocess="pass"),
    _q4(
        "bartowski/c4ai-command-r7b-12-2024-GGUF",
        "c4ai-command-r7b-12-2024-Q4_K_M.gguf",
        family="cohere",
        template_repo="CohereLabs/c4ai-command-r7b-12-2024",
        notes="GGUF ships no chat template; borrow from the (gated) HF repo",
        inprocess="pass",
    ),
    _q4(
        "ibm-granite/granite-3.3-8b-instruct-GGUF",
        "granite-3.3-8b-instruct-Q4_K_M.gguf",
        family="granite",
        inprocess="pass(untested-this-session)",
    ),
    # --- in-process exceptions: do better natively? ---
    _q4(
        "meetkai/functionary-small-v3.2-GGUF",
        "functionary-small-v3.2.Q8_0.gguf",
        family="functionary",
        template_repo="meetkai/functionary-small-v3.2",
        notes="stripped GGUF template; borrow HF",
        inprocess="partial (multi-turn flaky)",
    ),
    _q4(
        "unsloth/Olmo-3-7B-Instruct-GGUF",
        "Olmo-3-7B-Instruct-Q4_K_M.gguf",
        family="olmo3",
        inprocess="model-limit (narrates)",
    ),
    _q4(
        "internlm/internlm2_5-7b-chat-gguf",
        "internlm2_5-7b-chat-q4_k_m.gguf",
        family="internlm2",
        inprocess="model-limit (narrates)",
    ),
    _q4(
        "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        family="phi4mini",
        inprocess="upstream SIGABRT in llama-cpp-python",
    ),
    _q4(
        "legraphista/glm-4-9b-chat-GGUF",
        "glm-4-9b-chat.Q4_K_M.gguf",
        family="glm-4-9b",
        inprocess="upstream SIGABRT in llama-cpp-python",
    ),
    _q4(
        "QuantFactory/LFM2-1.2B-Tool-GGUF",
        "LFM2-1.2B-Tool.Q4_K_M.gguf",
        family="lfm2",
        inprocess="too-small",
    ),
    _q4(
        "unsloth/ERNIE-4.5-0.3B-PT-GGUF",
        "ERNIE-4.5-0.3B-PT-Q4_K_M.gguf",
        family="ernie",
        inprocess="too-small",
    ),
    # --- large, single H200 ---
    ModelSpec(
        family="gpt-oss",
        repo="ggml-org/gpt-oss-120b-GGUF",
        gguf="gpt-oss-120b-mxfp4-00001-of-00003.gguf",
        allow_patterns=("gpt-oss-120b-mxfp4-*.gguf",),
        inprocess="pass (Harmony, lilbee parser)",
        notes="split GGUF, 3 shards, ~63GB",
    ),
    ModelSpec(
        family="glm-air",
        repo="unsloth/GLM-4.5-Air-GGUF",
        gguf="Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf",
        allow_patterns=("Q4_K_M/GLM-4.5-Air-Q4_K_M-*.gguf",),
        template_repo="zai-org/GLM-4.5-Air",
        inprocess="pass (glm46, lilbee parser)",
        notes="split GGUF, 2 shards, subdir, ~68GB",
    ),
)
