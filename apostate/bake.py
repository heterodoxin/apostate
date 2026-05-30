"""bake edits into weights."""

from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ApostateConfig
from .model import ModelBundle

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _edit_linear(W: torch.Tensor, R: torch.Tensor, coeff: float) -> torch.Tensor:
    Wf = W.float()
    return (Wf + coeff * (R @ (R.t() @ Wf))).to(W.dtype)


def _edit_vec(b: torch.Tensor, R: torch.Tensor, coeff: float) -> torch.Tensor:
    bf = b.float()
    return (bf + coeff * (R @ (R.t() @ bf))).to(b.dtype)


def _edit_embed(W: torch.Tensor, R: torch.Tensor, coeff: float) -> torch.Tensor:
    Wf = W.float()
    return (Wf + coeff * ((Wf @ R) @ R.t())).to(W.dtype)


@torch.no_grad()
def bake(cfg: ApostateConfig, export: dict, tokenizer=None, drop_layers=None) -> str:
    edits = export.get("edits", [])
    if not edits:
        raise ValueError("Nothing to bake: no edits.")
    save_dtype = _DTYPES[cfg.save_dtype]

    print("[bake] loading model for editing...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=save_dtype, low_cpu_mem_usage=True,
        device_map={"": "cpu"}, trust_remote_code=True,
    )

    if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "lm_head"):
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
        model.config.tie_word_embeddings = False

    bundle = ModelBundle(model=model, tokenizer=tokenizer, num_layers=model.config.num_hidden_layers,
                         hidden_size=model.config.hidden_size)
    emb = bundle.embed()
    layers = bundle.layers()

    print("[bake] applying edits...", flush=True)
    for e in edits:
        R = e["R"].float()
        sign = float(e["sign"])
        a_emb = float(e["embed_alpha"])
        if a_emb != 0:
            emb.weight.data = _edit_embed(emb.weight.data, R, sign * a_emb)
        for L, layer in enumerate(layers):
            a = float(e["layer_alphas"][L])
            if a == 0:
                continue
            for mod in (bundle.attn_writer(layer), bundle.mlp_writer(layer)):
                mod.weight.data = _edit_linear(mod.weight.data, R, sign * a)
                if getattr(mod, "bias", None) is not None:
                    mod.bias.data = _edit_vec(mod.bias.data, R, sign * a)

    # drop redundant layers (faster generation)
    if drop_layers:
        drop = set(drop_layers)
        keep = [layers[i] for i in range(len(layers)) if i not in drop]
        dec = bundle._decoder()
        dec.layers = torch.nn.ModuleList(keep)
        model.config.num_hidden_layers = len(keep)
        for new_i, layer in enumerate(keep):       # reindex for kv cache
            for an in ("self_attn", "attention", "attn"):
                attn = getattr(layer, an, None)
                if attn is not None and hasattr(attn, "layer_idx"):
                    attn.layer_idx = new_i
        print(f"[bake] pruned {len(drop)} layers -> {len(keep)} remain", flush=True)

    os.makedirs(cfg.output_dir, exist_ok=True)
    print("[bake] saving...", flush=True)
    try:
        model.save_pretrained(cfg.output_dir, safe_serialization=True)
    except Exception as e:
        print(f"[bake] save failed: {e}, retrying with config only...", flush=True)
        model.config.save_pretrained(cfg.output_dir)
        model.save_pretrained(cfg.output_dir, safe_serialization=True, max_shard_size="5GB")

    tok = tokenizer or AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    tok.save_pretrained(cfg.output_dir)
    print("[bake] done", flush=True)
    return cfg.output_dir
