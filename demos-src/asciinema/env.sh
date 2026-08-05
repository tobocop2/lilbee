# Pod-private runtime env — proven k5b template with local-NVMe paths.
# Volume is READ-ONLY during fan-out except /workspace/reels-out/<own reel>/.
export LILBEE_DATA=/root/demo-data
export LILBEE_MODELS_DIR=/root/models
export LILBEE_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
export LILBEE_LLAMA_SERVER_PATH=/root/engine/llama-server
export LILBEE_THEME=rose-pine
export LILBEE_NO_SPLASH=1
export PATH=/root/venv/bin:$HOME/.opencode/bin:$HOME/.local/bin:/root/engine:/usr/local/cuda/bin:$PATH
export COLORTERM=truecolor
export TERM=xterm-256color
export HF_HUB_DISABLE_XET=1
