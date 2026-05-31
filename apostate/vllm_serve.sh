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

# setup-only: deps installed, don't serve
if [ -z "$MODEL" ] || [ "$MODEL" = "setup" ]; then
  echo "vllm ready in WSL ($("$V/bin/python" -c 'import vllm;print(vllm.__version__)'))."
  exit 0
fi

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

# turboquant: bf16 (faster, bigger context) if it fits free VRAM, else 4-bit
QUANT="--quantization bitsandbytes"
MAXLEN=8192
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
SIZE_B=$(du -sb "$MODEL" 2>/dev/null | cut -f1)
if [ -n "$FREE_MB" ] && [ -n "$SIZE_B" ] && [ "$((FREE_MB * 1000000))" -gt "$((SIZE_B + 3000000000))" ]; then
  QUANT=""          # weights + ~3GB kv fit in bf16 -> no quant
  MAXLEN=32768
fi
echo "turboquant: ${QUANT:-bf16} (free ${FREE_MB}MB)"

exec "$V/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name apostate --host 0.0.0.0 --port "$PORT" \
  $QUANT --enforce-eager \
  --gpu-memory-utilization 0.90 --max-model-len "$MAXLEN"
