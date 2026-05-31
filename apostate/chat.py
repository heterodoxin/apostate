"""chat repl."""

from __future__ import annotations

import argparse
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from .quant import quant_kwargs, MODES

_ACCEL_QUANTS = {"nf4", "fp4", "int8", "gptq", "marlin"}


def _accelerator_device() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


def _force_utf8_stdio():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _device_map(device: str | None) -> dict:
    if device == "cuda":
        return {"": 0}
    if device:
        return {"": device}
    return {"": "cpu"}


def _is_prequantized(model_id: str) -> bool:
    """already-quantized checkpoint (gptq/awq/marlin saved to disk)."""
    import json
    cfg = os.path.join(model_id, "config.json")
    if os.path.isfile(cfg):
        try:
            return "quantization_config" in json.load(open(cfg, encoding="utf-8"))
        except Exception:
            return False
    return False


def _load_model(model_id: str, quant: str, tok, device: str | None):
    if _is_prequantized(model_id):   # load saved quant as-is, no re-quantizing
        return AutoModelForCausalLM.from_pretrained(
            model_id, device_map=_device_map(device), low_cpu_mem_usage=True,
            trust_remote_code=True)
    kw = quant_kwargs(quant, tokenizer=tok)
    return AutoModelForCausalLM.from_pretrained(
        model_id, device_map=_device_map(device), low_cpu_mem_usage=True,
        trust_remote_code=True, **kw)


def main(argv=None):
    _force_utf8_stdio()

    ap = argparse.ArgumentParser(prog="apostate.chat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--quant", default="nf4", choices=MODES, help="inference quant")
    ap.add_argument("--backend", default="local", choices=["local", "vllm"], help="inference backend")
    ap.add_argument("--port", type=int, default=8000, help="vllm server port")
    ap.add_argument("--think", action="store_true", help="start with thinking enabled (Qwen3)")
    a = ap.parse_args(argv)

    # clear the screen + scrollback so we don't draw over the TUI's last frame
    print("\033[2J\033[3J\033[H", end="", flush=True)

    if a.backend == "vllm":
        from .vllm_backend import serve_and_chat
        if serve_and_chat(a.model, a.temperature, a.max_new_tokens, a.port):
            return
        print("falling back to local transformers ...", flush=True)

    print(f"loading {a.model} ({a.quant}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = _accelerator_device()
    load_quant = a.quant
    if device is None and load_quant in _ACCEL_QUANTS:
        print(f"  {load_quant} needs an accelerator; falling back to bf16 on CPU.", flush=True)
        load_quant = "bf16"
    try:
        quant_kwargs(load_quant, tokenizer=tok)
    except RuntimeError as e:
        print(f"{load_quant} backend unavailable: {e}\n  pip install gptqmodel optimum", flush=True)
        return
    if load_quant in ("gptq", "marlin"):
        print("  quantizing on first load (slow) ...", flush=True)
    try:
        model = _load_model(a.model, load_quant, tok, device)
    except Exception as e:
        fallback = "nf4" if device is not None and load_quant != "nf4" else "bf16"
        print(f"{load_quant} load failed: {str(e)[:160]}\n  falling back to {fallback}. "
              f"(gptq/marlin need a transformers version gptqmodel supports)", flush=True)
        load_quant = fallback
        model = _load_model(a.model, fallback, tok, device)
    model.eval()
    print(f"loaded on {next(model.parameters()).device} ({load_quant})", flush=True)

    think = a.think
    messages = []
    print("\nchat ready.  /reset  /think  /exit\n", flush=True)
    while True:
        try:
            user = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit", "/q"):
            break
        if user == "/reset":
            messages = []
            print("(conversation cleared)\n")
            continue
        if user == "/think":
            think = not think
            print(f"(thinking {'on' if think else 'off'})\n")
            continue

        messages.append({"role": "user", "content": user})
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
        streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        print("\033[35mmodel>\033[0m ", end="", flush=True)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=a.max_new_tokens, do_sample=a.temperature > 0,
                temperature=max(a.temperature, 1e-5), top_p=0.9,
                streamer=streamer, pad_token_id=tok.pad_token_id)
        resp = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": resp})
        print()

    print("bye.")


if __name__ == "__main__":
    main()
