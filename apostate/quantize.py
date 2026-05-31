"""quantize + save once (fast reload after)."""

from __future__ import annotations

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def _calib(tok, n):
    here = os.path.dirname(__file__)
    data = os.path.join(os.path.dirname(here), "data")
    lines = []
    for fn in ("harmless.txt", "harmful.txt"):
        p = os.path.join(data, fn)
        if os.path.exists(p):
            lines += [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    return lines[:n] or ["The quick brown fox jumps over the lazy dog."] * 128


def main(argv=None):
    ap = argparse.ArgumentParser(prog="apostate.quantize")
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
