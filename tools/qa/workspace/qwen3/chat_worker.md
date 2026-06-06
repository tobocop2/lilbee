# Chat worker

The chat worker subprocess in lilbee runs llama-cpp inference. It receives ChatRequest payloads over a pipe transport and streams tokens back via SSE. Cancellation is enforced through an abort flag in shared memory.