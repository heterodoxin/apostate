from __future__ import annotations

from typing import List
import torch

from .model import ModelBundle
from .data import format_chat
from .evaluate import refusal_rate, kl_harmless


@torch.no_grad()
def block_influence(bundle: ModelBundle, instructions: List[str], batch_size: int) -> List[float]:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = format_chat(tok, instructions)
    n = bundle.num_layers
    num = [0.0] * n
    den = 0
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        mask = enc["attention_mask"].bool()
        m = mask.sum().item()
        den += m
        for L in range(n):
            a = hs[L][mask].float()
            b = hs[L + 1][mask].float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            num[L] += float((1.0 - cos).sum())
        del out, hs
    return [x / max(1, den) for x in num]


class LayerSkip:
    def __init__(self, bundle: ModelBundle):
        self.layers = bundle.layers()
        self.skip = set()
        self._handles = []
        for idx, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._mk(idx)))

    def _mk(self, idx):
        def hook(_m, inp, out):
            if idx not in self.skip:
                return out
            h = inp[0]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def set(self, idxs):
        self.skip = set(idxs)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def select_prune(bundle, controller, eval_harmful, eval_harmless, cfg) -> List[int]:
    n = bundle.num_layers
    bi = block_influence(bundle, eval_harmless + eval_harmful, cfg.batch_size)
    order = sorted(range(n), key=lambda i: bi[i])
    cap = int(cfg.prune_max_frac * n)

    skip = LayerSkip(bundle)
    try:
        with controller.active():
            base_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        budget = base_kl + cfg.prune_kl
        kept: List[int] = []
        for L in order:
            if len(kept) >= cap:
                break
            skip.set(kept + [L])
            kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
            with controller.active():
                ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
            if kl <= budget and ref <= cfg.target_refusal:
                kept.append(L)
        skip.set([])
    finally:
        skip.remove()
    return sorted(kept)
