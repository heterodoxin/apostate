from __future__ import annotations

import glob
import os
import torch

MODES = ["auto", "bf16", "fp16", "nf4", "fp4", "int8", "gptq", "marlin", "awq"]

KV_CACHE_DTYPES = [
    "auto",
    "bf16",
    "fp8",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]


def _model_size_gb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for pat in ("*.safetensors", "*.bin"):
        for f in glob.glob(os.path.join(path, pat)):
            total += os.path.getsize(f)
    return total / 1e9


def auto_quant(model_path: str) -> str:
    try:
        if not torch.cuda.is_available():
            return "bf16"
        free = torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        return "nf4"
    size = _model_size_gb(model_path)
    if size and free > size * 1.25 + 1.5:
        return "bf16"
    return "nf4"


def quant_kwargs(mode: str, tokenizer=None, calib=None) -> dict:
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
            kw["format"] = "marlin"
        return {"quantization_config": GPTQConfig(**kw)}

    if mode == "awq":
        return {}

    raise ValueError(f"unknown quant {mode!r} (choose: {', '.join(MODES)})")
