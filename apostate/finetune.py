from __future__ import annotations

import argparse
import gc
import random
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from . import accel


def _chat(tok, instruction, response):
    msgs = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]
    kw = dict(tokenize=False)
    try:
        full = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        prompt = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, enable_thinking=False, **kw)
    except TypeError:
        full = tok.apply_chat_template(msgs, **kw)
        prompt = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, **kw)
    return prompt, full


def _load_jsonl(path: str):
    import json
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instr = row.get("instruction") or row.get("input") or row.get("prompt") or ""
            resp = row.get("response") or row.get("output") or row.get("completion") or ""
            if instr and resp:
                pairs.append((instr, resp))
    return pairs


def build_examples(tok, n_code, n_math, n_general, max_len, seed=0, extra_data: list[str] | None = None):
    from datasets import load_dataset
    rng = random.Random(seed)
    pairs = []

    def take(repo, cfg, ikey, rkey, n, fmt=None):
        if n <= 0:
            return
        ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
        for i in rng.sample(range(len(ds)), min(n, len(ds))):
            row = ds[i]
            instr = fmt(row) if fmt else row[ikey]
            pairs.append((instr, row[rkey]))

    take("ise-uiuc/Magicoder-Evol-Instruct-110K", None, "instruction", "response", n_code)
    take("meta-math/MetaMathQA", None, "query", "response", n_math)
    take("tatsu-lab/alpaca", None, "instruction", "output", n_general,
         fmt=lambda r: r["instruction"] + (("\n" + r["input"]) if r.get("input") else ""))

    for path in (extra_data or []):
        loaded = _load_jsonl(path)
        print(f"[ft] loaded {len(loaded)} custom examples from {path}", flush=True)
        pairs.extend(loaded)

    rng.shuffle(pairs)

    examples = []
    for instr, resp in pairs:
        prompt, full = _chat(tok, instr, resp)
        ids = tok(full, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        plen = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(ids))
        labels = [-100] * plen + ids[plen:]
        examples.append({"input_ids": ids, "labels": labels[:len(ids)], "attention_mask": [1] * len(ids)})
    return examples


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        def p(x, v): return x + [v] * (m - len(x))
        return {
            "input_ids": torch.tensor([p(b["input_ids"], self.pad) for b in batch]),
            "labels": torch.tensor([p(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([p(b["attention_mask"], 0) for b in batch]),
        }


def qlora_finetune(model_id, out_dir, n_code=2500, n_math=1500, n_general=500, max_len=768,
                   steps=400, lr=2e-4, batch=1, grad_accum=16, r=16, seed=0,
                   extra_data: list[str] | None = None) -> str:
    dev = accel.resolve_device("auto")
    accel.require_gpu(dev)
    accel.gpu_smoke_test(dev, log=lambda m: print("[ft] " + m, flush=True))

    # QLoRA needs bitsandbytes (4-bit base + paged 8-bit optimizer). On ROCm
    # without a usable bnb, fall back to bf16 LoRA -- works without bnb, fits a
    # 24GB+ card for small models.
    bnb_ok, why = accel.bitsandbytes_status()
    use_4bit = bnb_ok and dev == "cuda"
    if not use_4bit:
        print(f"[ft] bitsandbytes not usable ({why}); using bf16 LoRA instead of 4-bit QLoRA.", flush=True)
    accel.maybe_preflight(dev, model_id=model_id, load_in_4bit=use_4bit,
                          compute_dtype="bfloat16", batch_size=batch,
                          log=lambda m: print("[ft] " + m, flush=True))

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        print(f"[ft] loading {model_id} 4-bit + LoRA(r={r}) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        print(f"[ft] loading {model_id} bf16 + LoRA(r={r}) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                     device_map={"": dev}, trust_remote_code=True)
        # Trainer enables grad-checkpointing via TrainingArguments; a non-kbit frozen
        # base still needs input grads turned on for LoRA to backprop through it.
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False

    examples = build_examples(tok, n_code, n_math, n_general, max_len, seed, extra_data)
    print(f"[ft] {len(examples)} examples (code={n_code} math={n_math} gen={n_general}); training {steps} steps ...", flush=True)
    args = TrainingArguments(
        output_dir=out_dir + "_ckpt", per_device_train_batch_size=batch, gradient_accumulation_steps=grad_accum,
        max_steps=steps, learning_rate=lr, bf16=True, logging_steps=10, save_strategy="no",
        # paged_adamw_8bit is a bitsandbytes optimizer; plain adamw_torch when bnb is unusable
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        warmup_ratio=0.03, lr_scheduler_type="cosine", report_to=[],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    Trainer(model=model, args=args, train_dataset=examples, data_collator=Collator(tok)).train()

    adapter_dir = out_dir + "_adapter"
    model.save_pretrained(adapter_dir)
    del model
    gc.collect(); torch.cuda.empty_cache()

    print("[ft] merging LoRA into bf16 base (CPU) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                device_map={"": "cpu"}, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[ft] saved finetuned model -> {out_dir}", flush=True)
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(prog="apostate.finetune")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--n-code", type=int, default=2500)
    p.add_argument("--n-math", type=int, default=1500)
    p.add_argument("--n-general", type=int, default=500)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--data", action="append", dest="extra_data", default=[],
                   metavar="PATH", help="extra JSONL dataset(s) with instruction/response pairs")
    a = p.parse_args(argv)
    qlora_finetune(a.model, a.out, a.n_code, a.n_math, a.n_general, a.max_len, a.steps, r=a.r,
                   extra_data=a.extra_data or None)


if __name__ == "__main__":
    main()
