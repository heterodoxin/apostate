"""quant configs for inference."""

from __future__ import annotations

import glob
import os
import torch

MODES = ["auto", "bf16", "fp16", "nf4", "fp4", "int8", "gptq", "marlin", "awq"]


def _model_size_gb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for pat in ("*.safetensors", "*.bin"):
        for f in glob.glob(os.path.join(path, pat)):
            total += os.path.getsize(f)
    return total / 1e9


def auto_quant(model_path: str) -> str:
    """turboquant: fastest quant that fits free VRAM. bf16 if it fits, else nf4."""
    try:
        if not torch.cuda.is_available():
            return "bf16"                      # cpu: bnb gives no speedup
        free = torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        return "nf4"
    size = _model_size_gb(model_path)          # on-disk (~bf16) weight size
    if size and free > size * 1.25 + 1.5:      # room for weights + kv cache + overhead
        return "bf16"
    return "nf4"


def quant_kwargs(mode: str, tokenizer=None, calib=None) -> dict:
    """from_pretrained kwargs for a quant mode."""
    mode = (mode or "nf4").lower()

    if mode in ("bf16", "none"):
        return {"dtype": torch.bfloat16}
    if mode == "fp16":
        return {"dtype": torch.float16}

    from transformers import BitsAndBytesConfig
    if mode == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    if mode in ("nf4", "fp4"):
        return {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=mode,
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)}

    if mode in ("gptq", "marlin"):
        try:
            from transformers import GPTQConfig
        except Exception as e:
            raise RuntimeError(f"gptq needs a newer transformers: {e}")
        ds = calib or ["The quick brown fox jumps over the lazy dog."] * 32
        kw = dict(bits=4, dataset=ds, tokenizer=tokenizer, group_size=128)
        if mode == "marlin":
            kw["format"] = "marlin"   # fast int4 kernel
        return {"quantization_config": GPTQConfig(**kw)}

    if mode == "awq":
        return {}   # awq dir carries its own config

    raise ValueError(f"unknown quant {mode!r} (choose: {', '.join(MODES)})")
