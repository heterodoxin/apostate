"""coding eval."""

from __future__ import annotations

from typing import List, Tuple
import ast
import os
import subprocess
import sys
import tempfile
import torch

from .model import ModelBundle
from .data import format_chat

_PLACEHOLDERS = [
    "# todo", "# your code", "# implement", "raise notimplementederror",
    "# fill in", "# write your", "your code here", "rest of the code",
    "# ... ", "pass  # ",
]


def load_code_problems(spec: str, n: int) -> List[dict]:
    """Load HumanEval-style problems: prompt, canonical_solution, test, entry_point."""
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
        })
    return out


def coding_instructions(spec: str, n: int) -> List[str]:
    """Plain coding instructions for the activation contrast (HumanEval/MBPP text)."""
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


# ---------------------------------------------------------------------------
@torch.no_grad()
def solution_logprob(bundle: ModelBundle, problems: List[dict]) -> float:
    """Mean per-token log-prob of canonical solutions (higher == better coding alignment)."""
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
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
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
    instrs = [
        "Complete the following Python function. Respond with the complete function "
        "in a single ```python code block and nothing else.\n\n```python\n" + p["prompt"] + "\n```"
        for p in problems
    ]
    prompts = format_chat(tok, instrs)
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def _run_program(src: str, timeout: int) -> bool:
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


@torch.no_grad()
def pass_at_1(
    bundle: ModelBundle, problems: List[dict], max_new_tokens: int,
    batch_size: int, execute: bool, timeout: int = 10,
) -> Tuple[float, float]:
    """Return (pass@1, completeness_rate). pass@1 is 0.0 if execute is False.

    Parallelizes test execution across CPU cores for ~4-8x speedup.
    """
    gens = _solve(bundle, problems, max_new_tokens, batch_size)
    passed = 0
    complete = 0

    # Extract code and build test programs (single-threaded, uses GPU for generation)
    programs = []
    for p, g in zip(problems, gens):
        code = extract_code(g)
        if ast_complete(code):
            complete += 1
        if execute:
            if ("def " + p["entry_point"]) in code:
                program = code + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
            else:
                program = p["prompt"] + code + "\n" + p["test"] + f"\ncheck({p['entry_point']})\n"
            programs.append(program)

    # Parallel test execution (CPU-bound)
    if execute and programs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=None) as ex:  # None = #CPUs
            futures = [ex.submit(_run_program, prog, timeout) for prog in programs]
            for f in as_completed(futures):
                if f.result():
                    passed += 1

    n = max(1, len(problems))
    return (passed / n if execute else 0.0), complete / n
