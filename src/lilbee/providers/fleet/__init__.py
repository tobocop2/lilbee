"""The local inference engine: a managed ``llama-server`` fleet.

Chat/embed/rerank/vision each run on a managed ``llama-server`` process, reached
over a thin httpx client. A single machine is a fleet-of-one; the same code
bin-packs models across N GPUs. This is the sole local engine (no in-process
binding); ``llm_provider=auto`` routes native GGUF refs here.
"""
