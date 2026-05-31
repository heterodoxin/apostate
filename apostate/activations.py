"""activations"""

from __future__ import annotations

from typing import List
import torch

from .model import ModelBundle
from .data import format_chat


@torch.no_grad()
def collect_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    preformatted: bool = False,
) -> torch.Tensor:
    """collect activations"""
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)

    per_layer: List[List[torch.Tensor]] = [[] for _ in range(bundle.num_layers)]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # hidden states
        # left pad
        for layer in range(bundle.num_layers):
            h = hs[layer + 1][:, -1, :].float().cpu()
            per_layer[layer].append(h)
        del out, hs

    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer], dim=0)
