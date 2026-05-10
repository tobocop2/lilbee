"""Display names for the SDK-backed LLM backends.

This module is dependency-free so consumers (e.g.
modelhub.model_manager.types.RemoteModel) can reference the backend
names without pulling in the rest of sdk_backend.py.
"""

from enum import StrEnum


class BackendName(StrEnum):
    """Display name shown in the UI for whichever backend the SDK is talking to."""

    OLLAMA = "Ollama"
    OPENROUTER = "OpenRouter"
    GEMINI = "Gemini"
    ANTHROPIC = "Anthropic"
    OPENAI = "OpenAI"
    MISTRAL = "Mistral"
    DEEPSEEK = "DeepSeek"
    REMOTE = "Remote"
