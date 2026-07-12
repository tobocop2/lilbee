# Model families tested

lilbee's placement planner and engine are exercised against real models on real GPUs, one representative per architecture family, covering every role a model can hold: chat, embedding, vision, and rerank. Every family below was pulled with `lilbee model pull` and run through the full pipeline on consumer hardware, with nothing staged by hand. For chat families verified against coding agents and tool calling, see [agent-models](agent-models.md).

The vision families ran on a single 12 GB RTX 3080 Ti with the chat and embed defaults resident beside them: the model indexed an image-only scanned PDF through OCR and answered a question whose answer exists only in the scanned text. The text families ran the same pipeline (index a document, search it, answer from it) with each candidate swapped into its role next to the other defaults.

## Vision

| Family | Projector type | Tested with |
|--------|----------------|-------------|
| LightOnOCR | lightonocr | `noctrex/LightOnOCR-2-1B-GGUF` |
| Qwen2.5-VL | qwen2.5vl merger | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` |
| Qwen3-VL | qwen3vl | `Qwen/Qwen3-VL-4B-Instruct-GGUF` |
| Gemma 3 | gemma3 | `ggml-org/gemma-3-4b-it-GGUF` |
| SmolVLM2 | idefics3 | `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` |
| MiniCPM-V | resampler | `openbmb/MiniCPM-V-2_6-gguf` |
| InternVL3 | internvl | `ggml-org/InternVL3-2B-Instruct-GGUF` |
| LLaVA 1.6 | mlp | `cjpais/llava-1.6-mistral-7b-gguf` |
| Gemma 4 | mixed vision+audio | `ggml-org/gemma-4-12B-it-GGUF` |
| dots.ocr | dots.ocr | `ggml-org/dots.ocr-GGUF` |

Every family's pull fetched the multimodal projector on its own, placement never refused, and OCR plus the grounded answer completed on the 12 GB card. Beyond these live loads, the placement decision layer is swept against a corpus of 947 unique text-model and projector pairs spanning 34 projector types, checking that the corrected estimate never undercuts the model's own floor and that no model is refused on estimate alone.

## Chat

One representative per memory-architecture class, since context sizing and KV behavior differ by attention design, not by parameter count.

| Class | Tested with |
|-------|-------------|
| Dense GQA | `hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF` |
| Dense (Qwen3) | `Qwen/Qwen3-4B-GGUF` |
| Sliding-window attention | `ggml-org/gemma-3-4b-it-GGUF` |
| Multi-head latent attention | `mradermacher/DeepSeek-V2-Lite-Chat-GGUF` |
| Mixture of experts | `bartowski/OLMoE-1B-7B-0924-Instruct-GGUF` |
| Hybrid SSM | `LiquidAI/LFM2-1.2B-GGUF` |

## Embedding

| Class | Tested with |
|-------|-------------|
| BERT encoder | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` |
| nomic-bert | `nomic-ai/nomic-embed-text-v1.5-GGUF` |
| Decoder-pooled | `Qwen/Qwen3-Embedding-0.6B-GGUF` |
| Decoder-pooled, large | `Qwen/Qwen3-Embedding-8B-GGUF` |
| XLM-RoBERTa | `gpustack/bge-m3-GGUF` |

## Rerank

| Class | Tested with |
|-------|-------------|
| Cross-encoder | `gpustack/bge-reranker-v2-m3-GGUF` |
| LLM reranker | `mradermacher/Qwen3-Reranker-0.6B-GGUF` |

A family listed here means the architecture loads, places, and completes its role's pipeline. Model quality within a family still varies by size and training; a 2B vision model can read a scan yet answer thinly where a 4B one answers well.
