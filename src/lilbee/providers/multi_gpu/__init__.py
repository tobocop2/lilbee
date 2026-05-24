"""Opt-in multi-GPU managed llama-server fleet.

A sibling of the in-process llama-cpp provider: chat/embed/rerank/vision run on
llama-server sidecars bin-packed across GPUs, reached over a thin httpx client.
Inert unless ``llm_provider`` is ``multi-gpu``.
"""
