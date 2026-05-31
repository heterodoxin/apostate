#!/usr/bin/env bash
# args: $1 = model path (wsl), $2 = port. self-installs uv + vllm in a py3.12 venv.
set -e
MODEL="$1"
PORT="$2"
export DEBIAN_FRONTEND=noninteractive
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates)
command -v gcc  >/dev/null || (apt-get update -qq && apt-get install -y -qq build-essential)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null)
export PATH="$HOME/.local/bin:$PATH"
V="$HOME/.apostate-vllm"
[ -x "$V/bin/python" ] || uv venv --python 3.12 "$V"
"$V/bin/python" -c 'import vllm' 2>/dev/null || uv pip install -q --python "$V/bin/python" vllm
"$V/bin/python" -c 'import bitsandbytes' 2>/dev/null || uv pip install -q --python "$V/bin/python" bitsandbytes

# /mnt (9p) can't be mmap'd by safetensors -> copy to ext4 once
case "$MODEL" in
  /mnt/*)
    DEST="$HOME/.apostate-models/$(basename "$MODEL")"
    if [ ! -d "$DEST" ]; then
      echo "copying model to WSL filesystem (one-time, large) ..."
      mkdir -p "$HOME/.apostate-models"
      cp -r "$MODEL" "$DEST.partial" && mv "$DEST.partial" "$DEST"
    fi
    MODEL="$DEST"
    ;;
esac

# native torch sampler (FlashInfer would JIT-compile and need nvcc/CUDA toolkit)
export VLLM_USE_FLASHINFER_SAMPLER=0
# 4-bit so a 7B fits a 16GB card alongside the desktop (bf16 weights ~14GB won't)
exec "$V/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name apostate --host 0.0.0.0 --port "$PORT" \
  --quantization bitsandbytes --enforce-eager \
  --gpu-memory-utilization 0.90 --max-model-len 8192
