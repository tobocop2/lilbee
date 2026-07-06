# Dispatch layer

chat_dispatch.dispatch_chat is the canonical entry point that the OpenAI-compatible route forwards to. It resolves the model through KnownModelCache, enforces tool capability, and routes to either the local llama-server fleet or the SDK backend.