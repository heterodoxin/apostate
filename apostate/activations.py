from __future__ import annotations

from typing import List
import torch

from .model import ModelBundle
from .data import format_chat


def _out_tensor(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "hidden_states"):
        return out.hidden_states[-1]
    return out


def _prompt_batches(bundle, tok, prompts, batch_size, device):
    cache = getattr(bundle, "_act_enc_cache", None)
    if cache is None:
        cache = {}
        setattr(bundle, "_act_enc_cache", cache)
    key = (tuple(prompts), batch_size, str(device))
    if key not in cache:
        batches = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            batches.append({k: v.to(device) for k, v in enc.items()})
        cache[key] = batches
    return cache[key]


def _hidden_state_collect(model, batches, num_layers):
    per_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    for enc in batches:
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for layer in range(num_layers):
            per_layer[layer].append(hs[layer + 1][:, -1, :].detach().float().cpu())
    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer], dim=0)


def _hidden_state_collect_layer(model, batches, layer_idx):
    chunks: List[torch.Tensor] = []
    for enc in batches:
        out = model(**enc, output_hidden_states=True, use_cache=False)
        chunks.append(out.hidden_states[layer_idx + 1][:, -1, :].detach().float().cpu())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def collect_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    preformatted: bool = False,
) -> torch.Tensor:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    per_layer: List[List[torch.Tensor]] = [[] for _ in range(bundle.num_layers)]
    handles = []

    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_layer[layer].append(t[:, -1, :].detach().float().cpu())
        return hook

    for layer, mod in enumerate(bundle.layers()):
        handles.append(mod.register_forward_hook(make_hook(layer)))

    try:
        for enc in batches:
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(not chunks for chunks in per_layer):
        return _hidden_state_collect(model, batches, bundle.num_layers)

    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer], dim=0)


@torch.inference_mode()
def collect_layer_activations(
    bundle: ModelBundle,
    instructions: List[str],
    layer_idx: int,
    batch_size: int = 16,
    preformatted: bool = False,
) -> torch.Tensor:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    chunks: List[torch.Tensor] = []

    def hook(_mod, _inp, out):
        t = _out_tensor(out)
        chunks.append(t[:, -1, :].detach().float().cpu())

    handle = bundle.layers()[layer_idx].register_forward_hook(hook)
    try:
        for enc in batches:
            model(**enc, use_cache=False)
    finally:
        handle.remove()

    if not chunks:
        return _hidden_state_collect_layer(model, batches, layer_idx)
    return torch.cat(chunks, dim=0)
