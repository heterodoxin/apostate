# collect last-token / response-mean activations for harmful and harmless prompts.

from __future__ import annotations

from typing import Dict, List
import torch

from .model import ModelBundle
from .data import format_chat, format_messages


def _out_tensor(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "hidden_states"):
        return out.hidden_states[-1]
    return out


def _masked_mean(t: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None or t.dim() < 3:
        return t[:, -1, :]
    m = mask.to(t.device).to(t.dtype).unsqueeze(-1)
    return (t * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


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


def _pad_batch(items, pad_id: int, left: bool, device):
    width = max(len(x["input_ids"]) for x in items)
    input_ids, attention, response_mask = [], [], []
    for item in items:
        ids = item["input_ids"]
        mask = item["response_mask"]
        pad = width - len(ids)
        if left:
            input_ids.append([pad_id] * pad + ids)
            attention.append([0] * pad + [1] * len(ids))
            response_mask.append([0] * pad + mask)
        else:
            input_ids.append(ids + [pad_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            response_mask.append(mask + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attention, device=device),
        "response_mask": torch.tensor(response_mask, device=device),
    }


def _response_batches(bundle, instructions, responses, batch_size, device):
    tok = bundle.tokenizer
    cache = getattr(bundle, "_response_act_enc_cache", None)
    if cache is None:
        cache = {}
        setattr(bundle, "_response_act_enc_cache", cache)
    key = (tuple(instructions), tuple(responses), batch_size, str(device))
    if key in cache:
        return cache[key]
    items = []
    for instruction, response in zip(instructions, responses):
        prompt = format_messages(tok, [{"role": "user", "content": instruction}], add_generation_prompt=True)
        full = format_messages(
            tok,
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            add_generation_prompt=False,
        )
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"]
        start = min(len(pids), len(ids) - 1)
        mask = [0] * len(ids)
        for i in range(start, len(ids)):
            mask[i] = 1
        if not any(mask) and mask:
            mask[-1] = 1
        items.append({"input_ids": ids, "response_mask": mask})
    left = getattr(tok, "padding_side", "left") == "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = [
        _pad_batch(items[i : i + batch_size], pad_id, left, device)
        for i in range(0, len(items), batch_size)
    ]
    cache[key] = batches
    return batches


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


@torch.inference_mode()
def collect_response_activations(
    bundle: ModelBundle,
    instructions: List[str],
    responses: List[str],
    batch_size: int = 16,
) -> torch.Tensor:
    model = bundle.model
    device = next(model.parameters()).device
    batches = _response_batches(bundle, instructions, responses, batch_size, device)
    per_layer: List[List[torch.Tensor]] = [[] for _ in range(bundle.num_layers)]
    handles = []
    current_mask = [None]

    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_layer[layer].append(_masked_mean(t, current_mask[0]).detach().float().cpu())
        return hook

    for layer, mod in enumerate(bundle.layers()):
        handles.append(mod.register_forward_hook(make_hook(layer)))

    try:
        for batch in batches:
            current_mask[0] = batch["response_mask"]
            enc = {k: v for k, v in batch.items() if k != "response_mask"}
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer], dim=0)


@torch.inference_mode()
def collect_kv_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    layers: List[int] | None = None,
    preformatted: bool = False,
) -> Dict[str, Dict[int, torch.Tensor]]:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    selected = set(bundle.kv_source_layers() if layers is None else layers)
    per_part: Dict[str, Dict[int, List[torch.Tensor]]] = {"k": {}, "v": {}}
    handles = []
    current_mask = [None]

    def make_hook(part: str, layer_idx: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_part[part].setdefault(layer_idx, []).append(
                _masked_mean(t, current_mask[0]).detach().float().cpu()
            )
        return hook

    for layer_idx, layer in enumerate(bundle.layers()):
        if layer_idx not in selected:
            continue
        for part, mod in bundle.kv_writers(layer):
            handles.append(mod.register_forward_hook(make_hook(part, layer_idx)))

    try:
        for enc in batches:
            current_mask[0] = enc.get("attention_mask")
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, Dict[int, torch.Tensor]] = {"k": {}, "v": {}}
    for part, by_layer in per_part.items():
        for layer_idx, chunks in by_layer.items():
            if chunks:
                out[part][layer_idx] = torch.cat(chunks, dim=0)
    return out


@torch.inference_mode()
def collect_query_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    layers: List[int] | None = None,
    preformatted: bool = False,
) -> Dict[int, torch.Tensor]:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    selected = set(bundle.query_layer_candidates() if layers is None else layers)
    per_layer: Dict[int, List[torch.Tensor]] = {}
    handles = []
    current_mask = [None]

    def make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_layer.setdefault(layer_idx, []).append(
                _masked_mean(t, current_mask[0]).detach().float().cpu()
            )
        return hook

    for layer_idx, layer in enumerate(bundle.layers()):
        if layer_idx not in selected:
            continue
        for mod in bundle.query_writers(layer):
            handles.append(mod.register_forward_hook(make_hook(layer_idx)))

    try:
        for enc in batches:
            current_mask[0] = enc.get("attention_mask")
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {
        layer_idx: torch.cat(chunks, dim=0)
        for layer_idx, chunks in per_layer.items()
        if chunks
    }


@torch.inference_mode()
def collect_response_kv_activations(
    bundle: ModelBundle,
    instructions: List[str],
    responses: List[str],
    batch_size: int = 16,
    layers: List[int] | None = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    model = bundle.model
    device = next(model.parameters()).device
    batches = _response_batches(bundle, instructions, responses, batch_size, device)
    selected = set(bundle.kv_source_layers() if layers is None else layers)
    per_part: Dict[str, Dict[int, List[torch.Tensor]]] = {"k": {}, "v": {}}
    handles = []
    current_mask = [None]

    def make_hook(part: str, layer_idx: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_part[part].setdefault(layer_idx, []).append(
                _masked_mean(t, current_mask[0]).detach().float().cpu()
            )
        return hook

    for layer_idx, layer in enumerate(bundle.layers()):
        if layer_idx not in selected:
            continue
        for part, mod in bundle.kv_writers(layer):
            handles.append(mod.register_forward_hook(make_hook(part, layer_idx)))

    try:
        for batch in batches:
            current_mask[0] = batch["response_mask"]
            enc = {k: v for k, v in batch.items() if k != "response_mask"}
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, Dict[int, torch.Tensor]] = {"k": {}, "v": {}}
    for part, by_layer in per_part.items():
        for layer_idx, chunks in by_layer.items():
            if chunks:
                out[part][layer_idx] = torch.cat(chunks, dim=0)
    return out


@torch.inference_mode()
def collect_response_query_activations(
    bundle: ModelBundle,
    instructions: List[str],
    responses: List[str],
    batch_size: int = 16,
    layers: List[int] | None = None,
) -> Dict[int, torch.Tensor]:
    model = bundle.model
    device = next(model.parameters()).device
    batches = _response_batches(bundle, instructions, responses, batch_size, device)
    selected = set(bundle.query_layer_candidates() if layers is None else layers)
    per_layer: Dict[int, List[torch.Tensor]] = {}
    handles = []
    current_mask = [None]

    def make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_layer.setdefault(layer_idx, []).append(
                _masked_mean(t, current_mask[0]).detach().float().cpu()
            )
        return hook

    for layer_idx, layer in enumerate(bundle.layers()):
        if layer_idx not in selected:
            continue
        for mod in bundle.query_writers(layer):
            handles.append(mod.register_forward_hook(make_hook(layer_idx)))

    try:
        for batch in batches:
            current_mask[0] = batch["response_mask"]
            enc = {k: v for k, v in batch.items() if k != "response_mask"}
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return {
        layer_idx: torch.cat(chunks, dim=0)
        for layer_idx, chunks in per_layer.items()
        if chunks
    }


@torch.inference_mode()
def collect_ple_gate_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    preformatted: bool = False,
) -> torch.Tensor | None:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    modules = [bundle.ple_writers(layer) for layer in bundle.layers()]
    if not any(modules):
        return None
    per_layer: List[List[torch.Tensor]] = [[] for _ in range(bundle.num_layers)]
    handles = []
    current_mask = [None]

    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_layer[layer].append(_masked_mean(t, current_mask[0]).detach().float().cpu())
        return hook

    for layer, mods in enumerate(modules):
        for mod in mods:
            handles.append(mod.register_forward_hook(make_hook(layer)))

    try:
        for enc in batches:
            current_mask[0] = enc.get("attention_mask")
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(not chunks for chunks in per_layer):
        return None
    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer], dim=0)


@torch.inference_mode()
def collect_ple_projection_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    preformatted: bool = False,
) -> torch.Tensor | None:
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    modules = [bundle.ple_projection_writers(layer) for layer in bundle.layers()]
    if not any(modules):
        return None
    per_layer: List[List[torch.Tensor]] = [[] for _ in range(bundle.num_layers)]
    handles = []
    current_mask = [None]

    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            t = _out_tensor(out)
            per_layer[layer].append(_masked_mean(t, current_mask[0]).detach().float().cpu())
        return hook

    for layer, mods in enumerate(modules):
        for mod in mods:
            handles.append(mod.register_forward_hook(make_hook(layer)))

    try:
        for enc in batches:
            current_mask[0] = enc.get("attention_mask")
            model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(not chunks for chunks in per_layer):
        return None
    return torch.stack([torch.cat(chunks, dim=0) for chunks in per_layer], dim=0)


@torch.inference_mode()
def collect_ple_embed_activations(
    bundle: ModelBundle,
    instructions: List[str],
    batch_size: int = 16,
    preformatted: bool = False,
) -> torch.Tensor | None:
    mod = bundle.ple_embed()
    if mod is None:
        return None
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    prompts = instructions if preformatted else format_chat(tok, instructions)
    batches = _prompt_batches(bundle, tok, prompts, batch_size, device)
    chunks: List[torch.Tensor] = []
    current_mask = [None]

    def hook(_mod, _inp, out):
        t = _out_tensor(out)
        chunks.append(_masked_mean(t, current_mask[0]).detach().float().cpu())

    handle = mod.register_forward_hook(hook)
    try:
        for enc in batches:
            current_mask[0] = enc.get("attention_mask")
            model(**enc, use_cache=False)
    finally:
        handle.remove()

    if not chunks:
        return None
    return torch.cat(chunks, dim=0)
