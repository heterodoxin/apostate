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
    sources = [s.strip() for s in path_or_spec.split("|") if s.strip()]
    seen, pool = set(), []
    for src in sources:
        for p in _resolve_one(src, n, seed):
            if p not in seen:
                seen.add(p)
                pool.append(p)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content)


def fallback_chat_text(tokenizer, messages, add_generation_prompt: bool = True) -> Optional[str]:
    special = getattr(tokenizer, "special_tokens_map", {}) or {}
    sot = special.get("sot_token") or getattr(tokenizer, "sot_token", None)
    eot = special.get("eot_token") or getattr(tokenizer, "eot_token", None)
    if sot and eot:
        bos = getattr(tokenizer, "bos_token", None) or ""
        chunks = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            chunks.append(f"{sot}{role}\n{_content_text(msg.get('content', ''))}{eot}")
        text = bos + "\n".join(chunks)
        if add_generation_prompt:
            text += f"\n{sot}model\n"
        return text
    return None


def format_messages(tokenizer, messages, add_generation_prompt: bool = True) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        return fallback_chat_text(tokenizer, messages, add_generation_prompt=add_generation_prompt) or _content_text(
            messages[-1].get("content", "") if messages else ""
        )


def format_chat(tokenizer, instructions: List[str]) -> List[str]:
    out = []
    for ins in instructions:
        msg = [{"role": "user", "content": ins}]
        out.append(format_messages(tokenizer, msg, add_generation_prompt=True))
    return out


def format_chat_pairs(tokenizer, instructions: List[str], responses: List[str]) -> List[str]:
    out = []
    for ins, response in zip(instructions, responses):
        msg = [
            {"role": "user", "content": ins},
            {"role": "assistant", "content": response},
        ]
        out.append(format_messages(tokenizer, msg, add_generation_prompt=False))
    return out
