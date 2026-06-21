# benchmarking: load models, run the factual probe + KL, and sandboxed code pass@1.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import argparse
import ast
import json
import gc
import os
import re
import subprocess
import sys
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import accel
from .model import ModelBundle
from .data import resolve_prompts, format_chat
from .evaluate import is_refusal, judge_refusal

PROBE: List[Tuple[str, List[str]]] = [
    ("What is the capital of France?", ["paris"]),
    ("What is 7 multiplied by 8?", ["56"]),
    ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("What is the chemical symbol for water?", ["h2o", "h₂o"]),
    ("What planet is known as the Red Planet?", ["mars"]),
    ("In what year did World War II end?", ["1945"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("What gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2", "co₂"]),
    ("What is the square root of 144?", ["12"]),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]),
    ("What is the boiling point of water in Celsius at sea level?", ["100"]),
    ("Which language has the most native speakers worldwide?", ["chinese", "mandarin"]),
]

BENIGN_TEXT = (
    "The water cycle describes how water moves through the environment. Water "
    "evaporates from oceans and lakes, rises into the atmosphere, condenses into "
    "clouds, and eventually falls back to the surface as rain or snow. Rivers then "
    "carry the water back to the sea, and the cycle begins again."
)


def _load(path: str):
    adapter_dir = None
    base_path = path
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_path = json.load(f)["base_model_name_or_path"]
        adapter_dir = path

    dev = accel.resolve_device("auto")
    accel.require_gpu(dev)
    accel.gpu_smoke_test(dev)
    bnb_ok, why = accel.bitsandbytes_status()
    use_4bit = bnb_ok and dev == "cuda"
    if not use_4bit:
        print(f"[bench] bitsandbytes not usable ({why}); loading {base_path} in bf16.", flush=True)
    accel.maybe_preflight(dev, model_id=base_path, load_in_4bit=use_4bit, compute_dtype="bfloat16")

    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map={"": dev}, trust_remote_code=True,
        )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    gc_cfg = getattr(model, "generation_config", None)
    if gc_cfg is not None:
        gc_cfg.do_sample = False
        for a in ("temperature", "top_p", "top_k"):
            if hasattr(gc_cfg, a):
                setattr(gc_cfg, a, None)
    return model, tok


@torch.no_grad()
def _generate(model, tok, instructions, max_new_tokens, batch_size):
    device = next(model.parameters()).device
    prompts = format_chat(tok, instructions)
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


@torch.no_grad()
def _judge_flags(model, tok, responses, batch_size, judge: str):
    if judge == "keyword":
        return [is_refusal(c) for c in responses]
    if judge != "classifier":
        raise ValueError("judge must be 'classifier' or 'keyword'")
    class _Bundle:
        pass
    b = _Bundle()
    b.model = model
    b.tokenizer = tok
    try:
        return judge_refusal(b, responses, batch_size)
    except Exception as e:
        print(f"[bench] classifier judge unavailable ({e}); keyword fallback", flush=True)
        return [is_refusal(c) for c in responses]


@torch.no_grad()
def _refusal_rate(model, tok, harmful, max_new_tokens, batch_size, judge: str = "classifier"):
    comps = _generate(model, tok, harmful, max_new_tokens, batch_size)
    flags = _judge_flags(model, tok, comps, batch_size, judge)
    return sum(flags) / max(1, len(flags))


@torch.no_grad()
def _capability(model, tok, batch_size):
    qs = [q for q, _ in PROBE]
    comps = _generate(model, tok, qs, 24, batch_size)
    ok = 0
    for (q, answers), c in zip(PROBE, comps):
        cl = c.lower()
        if any(a in cl for a in answers):
            ok += 1
    return ok / len(PROBE)


@torch.no_grad()
def _perplexity(model, tok, text):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt").to(device)
    out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())


@torch.no_grad()
def _logprobs_lastK(model, tok, instructions, K, batch_size):
    device = next(model.parameters()).device
    prompts = format_chat(tok, instructions)
    chunks = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        k = min(K, enc["input_ids"].shape[1])
        logits = model(**enc, use_cache=False).logits[:, -k:, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        if k < K:
            lp = torch.nn.functional.pad(lp, (0, 0, K - k, 0))
        chunks.append(lp.half().cpu())
    return torch.cat(chunks, dim=0)


def _kl(ref_lp: torch.Tensor, cand_lp: torch.Tensor) -> float:
    rp = ref_lp.float()
    cp = cand_lp.float()
    kl = (rp.exp() * (rp - cp)).sum(-1)
    return float(kl.mean().item())


def benchmark_model(
    path: str,
    harmful_test: List[str],
    ref_lp: Optional[torch.Tensor],
    harmless_test: List[str],
    max_new_tokens: int = 40,
    batch_size: int = 16,
    kl_K: int = 8,
    judge: str = "classifier",
) -> Tuple[dict, torch.Tensor]:
    model, tok = _load(path)
    res = {
        "refusal_rate": round(_refusal_rate(model, tok, harmful_test, max_new_tokens, batch_size, judge), 4),
        "capability": round(_capability(model, tok, batch_size), 4),
        "perplexity": round(_perplexity(model, tok, BENIGN_TEXT), 3),
    }
    cand_lp = _logprobs_lastK(model, tok, harmless_test, kl_K, batch_size)
    res["kl_vs_base"] = 0.0 if ref_lp is None else round(_kl(ref_lp, cand_lp), 4)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res, cand_lp


def run_compare(
    base: str,
    models: Dict[str, str],
    harmful_spec: str,
    harmless_spec: str,
    n: int = 64,
    out: Optional[str] = None,
    seed: int = 0,
    judge: str = "classifier",
) -> dict:
    harmful_test = resolve_prompts(harmful_spec, n, seed)
    harmless_test = resolve_prompts(harmless_spec, n, seed)
    print(f"[bench] {len(harmful_test)} harmful / {len(harmless_test)} harmless held-out prompts")

    print(f"[bench] base: {base}")
    base_res, base_lp = benchmark_model(base, harmful_test, None, harmless_test, judge=judge)

    report = {"judge": judge, "base": {"path": base, **base_res}, "models": {}}
    for label, path in models.items():
        print(f"[bench] candidate '{label}': {path}")
        res, _ = benchmark_model(path, harmful_test, base_lp, harmless_test, judge=judge)
        report["models"][label] = {"path": path, **res}

    rows = [("model", "refusal%", "KL_vs_base", "capability%", "perplexity")]
    rows.append(("BASE (original)", f"{base_res['refusal_rate']*100:.1f}", "0.0000",
                 f"{base_res['capability']*100:.1f}", f"{base_res['perplexity']:.2f}"))
    for label, r in report["models"].items():
        rows.append((label, f"{r['refusal_rate']*100:.1f}", f"{r['kl_vs_base']:.4f}",
                     f"{r['capability']*100:.1f}", f"{r['perplexity']:.2f}"))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    print("\n=== HEAD-TO-HEAD (held-out) ===")
    for j, r in enumerate(rows):
        print("  " + "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))))
        if j == 0:
            print("  " + "  ".join("-" * widths[i] for i in range(len(r))))

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[bench] wrote {out}")
    return report


# code benchmarking: solve problems and run their tests in a sandbox for executable pass@1.

_PLACEHOLDERS = [
    "# todo", "# your code", "# implement", "raise notimplementederror",
    "# fill in", "# write your", "your code here", "rest of the code",
    "# ... ", "pass  # ",
]


def load_code_problems(spec: str, n: int) -> List[dict]:
    from datasets import load_dataset
    parts = spec.split(":")
    repo = parts[0]
    split = parts[1] if len(parts) > 1 else "test"
    ds = load_dataset(repo, split=split)
    out = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        out.append({
            "prompt": row.get("prompt", ""),
            "canonical_solution": row.get("canonical_solution", ""),
            "test": row.get("test", ""),
            "entry_point": row.get("entry_point", ""),
            "test_style": "check",
        })
    return out


def entry_from_code(code: str) -> str:
    try:
        tree = ast.parse(code or "")
    except Exception:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return ""


def entry_from_tests(tests: str) -> str:
    m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", tests or "")
    return m.group(1) if m else ""


def coding_instructions(spec: str, n: int) -> List[str]:
    from datasets import load_dataset
    parts = spec.split(":")
    repo = parts[0]
    split = parts[1] if len(parts) > 1 else "train"
    col = parts[2] if len(parts) > 2 else "text"
    ds = load_dataset(repo, split=split)
    out = []
    for i in range(min(n, len(ds))):
        v = ds[i].get(col) or ds[i].get("prompt") or ds[i].get("text")
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


@torch.no_grad()
def solution_logprob(bundle: ModelBundle, problems: List[dict]) -> float:
    # cheap capability proxy: how likely the model finds the canonical solution
    tok, model = bundle.tokenizer, bundle.model
    device = next(model.parameters()).device
    vals = []
    for p in problems:
        prompt, sol = p["prompt"], p["canonical_solution"]
        if not sol:
            continue
        ids_full = tok(prompt + sol, return_tensors="pt").input_ids.to(device)
        plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
        if ids_full.shape[1] <= plen:
            continue
        logits = model(ids_full, use_cache=False).logits.float()
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids_full[:, 1:]
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        sol_logp = tok_logp[:, plen - 1:]
        if sol_logp.numel():
            vals.append(sol_logp.mean().item())
    return sum(vals) / max(1, len(vals))


def extract_code(text: str) -> str:
    if "```" in text:
        seg = text.split("```", 2)
        if len(seg) >= 2:
            block = seg[1]
            for lang in ("python", "py"):
                if block.lstrip().lower().startswith(lang):
                    block = block.lstrip()[len(lang):]
                    break
            return block.strip()
    return text.strip()


def ast_complete(code: str) -> bool:
    try:
        ast.parse(code)
    except Exception:
        return False
    low = code.lower()
    if any(ph in low for ph in _PLACEHOLDERS):
        return False
    return "def " in code


@torch.no_grad()
def _solve(bundle: ModelBundle, problems: List[dict], max_new_tokens: int, batch_size: int) -> List[str]:
    tok, model = bundle.tokenizer, bundle.model
    device = next(model.parameters()).device
    instrs = []
    for p in problems:
        if p.get("test_style") == "assert":
            instrs.append(
                "Write a Python function that solves the task. Respond with the complete "
                "function in a single ```python code block and nothing else.\n\n"
                f"Task:\n{p['prompt']}\n\nTests:\n```python\n{p.get('test', '')}\n```"
            )
        else:
            instrs.append(
                "Complete the following Python function. Respond with the complete function "
                "in a single ```python code block and nothing else.\n\n```python\n" + p["prompt"] + "\n```"
            )
    prompts = format_chat(tok, instrs)
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def _run_program(src: str, timeout: int) -> bool:
    # isolated interpreter (-I -S), temp cwd, hard timeout; never trust generated code
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-S", path],
                cwd=d, capture_output=True, timeout=timeout,
                env={"PYTHONHASHSEED": "0"},
            )
            return r.returncode == 0
        except Exception:
            return False


def program_for_problem(p: dict, code: str) -> str:
    tests = p.get("test", "")
    if p.get("test_style") == "assert":
        return code + "\n\n" + tests + "\n"
    entry = p.get("entry_point", "")
    if entry and ("def " + entry) in code:
        body = code
    else:
        body = p.get("prompt", "") + code
    if entry:
        return body + "\n" + tests + f"\ncheck({entry})\n"
    return body + "\n" + tests + "\n"


@torch.no_grad()
def pass_at_1(
    bundle: ModelBundle, problems: List[dict], max_new_tokens: int,
    batch_size: int, execute: bool, timeout: int = 10,
) -> Tuple[float, float]:
    gens = _solve(bundle, problems, max_new_tokens, batch_size)
    passed = 0
    complete = 0

    programs = []
    for p, g in zip(problems, gens):
        code = extract_code(g)
        if ast_complete(code):
            complete += 1
        if execute:
            programs.append(program_for_problem(p, code))

    if execute and programs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=None) as ex:
            futures = [ex.submit(_run_program, prog, timeout) for prog in programs]
            for f in as_completed(futures):
                if f.result():
                    passed += 1

    n = max(1, len(problems))
    return (passed / n if execute else 0.0), complete / n


# cli: compare a base against candidate models on the probe set.

def main(argv=None):
    p = argparse.ArgumentParser(prog="apostate.benchmark")
    p.add_argument("--base", required=True, help="reference base model id/path")
    p.add_argument("--models", required=True, help="comma list of label=path")
    p.add_argument("--harmful", default="mlabonne/harmful_behaviors:test:text")
    p.add_argument("--harmless", default="mlabonne/harmless_alpaca:test:text")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--out", default="benchmark.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", default="classifier", choices=["classifier", "keyword"])
    a = p.parse_args(argv)
    models = {}
    for item in a.models.split(","):
        label, path = item.split("=", 1)
        models[label.strip()] = path.strip()
    run_compare(a.base, models, a.harmful, a.harmless, a.n, a.out, a.seed, a.judge)


if __name__ == "__main__":
    main()
