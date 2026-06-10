# Chat engine

Chat inference in lilbee runs on a managed llama-server fleet. llama-swap supervises the server processes behind an OpenAI-compatible proxy, gguf-parser estimates each model's memory footprint for placement, and tokens stream back via SSE.