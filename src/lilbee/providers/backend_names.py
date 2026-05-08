"""Display names for the SDK-backed LLM backends.

These exist as a separate dependency-free module so consumers (e.g.
modelhub.model_manager.types.RemoteModel) can reference REMOTE_BACKEND_NAME
without pulling in the rest of sdk_backend.py.
"""

OLLAMA_BACKEND_NAME = "Ollama"
OPENAI_BACKEND_NAME = "OpenAI"
ANTHROPIC_BACKEND_NAME = "Anthropic"
GEMINI_BACKEND_NAME = "Gemini"
REMOTE_BACKEND_NAME = "Remote"
