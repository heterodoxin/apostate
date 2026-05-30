"""prompt loading."""

from __future__ import annotations

from typing import List, Optional
import os
import random


def _read_lines(path: str, limit: Optional[int] = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if limit is not None:
        lines = lines[:limit]
    return lines


def load_prompts(path: str, n: int, seed: int = 0) -> List[str]:
    lines = _read_lines(path)
    rng = random.Random(seed)
    rng.shuffle(lines)
    if n and n < len(lines):
        lines = lines[:n]
    return lines


def maybe_load_hf(spec: str, n: int, seed: int = 0) -> List[str]:
    """hf spec: 'repo:split:col' or 'repo@config:split:col'."""
    from datasets import load_dataset
    repo, _, rest = spec.partition(":")
    config = None
    if "@" in repo:
        repo, config = repo.split("@", 1)
    parts = rest.split(":")
    split = parts[0] if parts and parts[0] else "train"
    col = parts[1] if len(parts) > 1 else "text"
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        v = ds[i].get(col)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        if len(out) >= n:
            break
    return out


def _resolve_one(spec: str, n: int, seed: int) -> List[str]:
    if os.path.exists(spec):
        return load_prompts(spec, n, seed)
    is_win = len(spec) > 2 and spec[1] == ":" and spec[0].isalpha()
    if spec and ":" in spec and not is_win:
        try:
            return maybe_load_hf(spec, n, seed)
        except Exception as e:
            print(f"[apostate] skip source {spec!r}: {e}", flush=True)
            return []
    return load_prompts(spec, n, seed)


def resolve_prompts(path_or_spec: str, n: int, seed: int = 0) -> List[str]:
    """sources joined by '|'; pool + dedup + shuffle, capped at n."""
    sources = [s.strip() for s in path_or_spec.split("|") if s.strip()]
    seen, pool = set(), []
    for src in sources:
        for p in _resolve_one(src, n, seed):
            if p not in seen:
                seen.add(p)
                pool.append(p)
    random.Random(seed).shuffle(pool)   # mix sources so tuning subsets are representative
    return pool[:n]


def format_chat(tokenizer, instructions: List[str]) -> List[str]:
    """Wrap raw instructions in the model's chat template, ready for generation."""
    out = []
    for ins in instructions:
        msg = [{"role": "user", "content": ins}]
        try:
            # disable hybrid "thinking" (Qwen3 etc.) so refusals/answers stay front-loaded
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = ins
        out.append(text)
    return out
