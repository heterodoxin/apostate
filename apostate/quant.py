from __future__ import annotations

import argparse
import glob
import os
import torch

# weight quant modes for `talk` loading; KV-cache dtypes are a vLLM-only thing
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


def _low_mem_quant() -> str:
    # 4-bit needs working bitsandbytes; on ROCm-without-bnb fall back to bf16
    # rather than picking a mode that errors (or hangs) at load.
    from . import accel
    ok, _ = accel.bitsandbytes_status()
    return "nf4" if ok else "bf16"


def auto_quant(model_path: str) -> str:
    try:
        if not torch.cuda.is_available():
            return "bf16"
        free = torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        return _low_mem_quant()
    size = _model_size_gb(model_path)
    if size and free > size * 1.25 + 1.5:
        return "bf16"
    return _low_mem_quant()


def quant_kwargs(mode: str, tokenizer=None, calib=None) -> dict:
    mode = (mode or "nf4").lower()

    if mode in ("bf16", "none"):
        return {"dtype": torch.bfloat16}
    if mode == "fp16":
        return {"dtype": torch.float16}

    from transformers import BitsAndBytesConfig
    if mode in ("int8", "nf4", "fp4"):
        from . import accel
        ok, why = accel.bitsandbytes_status()
        if not ok:
            raise RuntimeError(
                f"quant '{mode}' needs bitsandbytes but it is not usable here ({why}). "
                "use --quant bf16 (or fp16), or install a ROCm-enabled bitsandbytes."
            )
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


# offline gptq baker: quantize a checkpoint to disk once (python -m apostate.quant).

def _calib(tok, n):
    # reuse our own prompt files as calibration text; fall back to a filler line
    here = os.path.dirname(__file__)
    data = os.path.join(os.path.dirname(here), "data")
    lines = []
    for fn in ("harmless.txt", "harmful.txt"):
        p = os.path.join(data, fn)
        if os.path.exists(p):
            lines += [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    return lines[:n] or ["The quick brown fox jumps over the lazy dog."] * 128


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    ap = argparse.ArgumentParser(prog="apostate.quant")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--marlin", action="store_true", help="marlin int4 kernel")
    ap.add_argument("--ncalib", type=int, default=256)
    a = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    kw = dict(bits=a.bits, dataset=_calib(tok, a.ncalib), tokenizer=tok)
    if a.marlin:
        kw["format"] = "marlin"
    print(f"quantizing {a.model} -> {a.bits}-bit{' marlin' if a.marlin else ''} (one-time) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=GPTQConfig(**kw), device_map={"": 0}, trust_remote_code=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}  (load with: apostate talk --model {a.out})", flush=True)


if __name__ == "__main__":
    main()
