"""quant configs for inference."""

from __future__ import annotations

import torch

MODES = ["bf16", "fp16", "nf4", "fp4", "int8", "gptq", "marlin", "awq"]


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
        ds = calib or ["The quick brown fox jumps over the lazy dog."] * 128
        kw = dict(bits=4, dataset=ds, tokenizer=tokenizer)
        if mode == "marlin":
            kw["format"] = "marlin"   # fast int4 kernel
        return {"quantization_config": GPTQConfig(**kw)}

    if mode == "awq":
        return {}   # awq dir carries its own config

    raise ValueError(f"unknown quant {mode!r} (choose: {', '.join(MODES)})")
