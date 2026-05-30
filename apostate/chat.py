"""chat repl."""

from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from .quant import quant_kwargs, MODES


def main(argv=None):
    ap = argparse.ArgumentParser(prog="apostate.chat")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--quant", default="nf4", choices=MODES, help="inference quant")
    ap.add_argument("--think", action="store_true", help="start with thinking enabled (Qwen3)")
    a = ap.parse_args(argv)

    # clear the screen + scrollback so we don't draw over the TUI's last frame
    print("\033[2J\033[3J\033[H", end="", flush=True)

    print(f"loading {a.model} ({a.quant}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        kw = quant_kwargs(a.quant, tokenizer=tok)
    except RuntimeError as e:
        print(f"{a.quant} backend unavailable: {e}\n  pip install gptqmodel optimum", flush=True)
        return
    if a.quant in ("gptq", "marlin"):
        print("  quantizing on first load (slow) ...", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, device_map={"": 0}, trust_remote_code=True, **kw)
    except Exception as e:
        print(f"{a.quant} load failed: {str(e)[:160]}\n  falling back to nf4. "
              f"(gptq/marlin need a transformers version gptqmodel supports)", flush=True)
        from .quant import quant_kwargs as _qk
        model = AutoModelForCausalLM.from_pretrained(
            a.model, device_map={"": 0}, trust_remote_code=True, **_qk("nf4"))
    model.eval()

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
        enc = tok(prompt, return_tensors="pt").to(model.device)
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
