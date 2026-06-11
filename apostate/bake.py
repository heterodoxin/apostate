# fold the final projection into residual-writer weights and save a standalone checkpoint.

from __future__ import annotations

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ApostateConfig
from .model import ModelBundle, model_metadata, set_config_value, _is_conv1d

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# R is the left/removal basis; U is the right co-vector (oblique mean-preserving edit).
# when U is None the edit is the usual symmetric projection (U == R).

def _edit_linear(W: torch.Tensor, R: torch.Tensor, coeff: float, U: torch.Tensor = None) -> torch.Tensor:
    right = R if U is None else U
    Wf = W.float()
    return (Wf + coeff * (R @ (right.t() @ Wf))).to(W.dtype)


def _edit_vec(b: torch.Tensor, R: torch.Tensor, coeff: float, U: torch.Tensor = None) -> torch.Tensor:
    right = R if U is None else U
    bf = b.float()
    return (bf + coeff * (R @ (right.t() @ bf))).to(b.dtype)


def _edit_embed(W: torch.Tensor, R: torch.Tensor, coeff: float, U: torch.Tensor = None) -> torch.Tensor:
    right = R if U is None else U
    Wf = W.float()
    return (Wf + coeff * ((Wf @ right) @ R.t())).to(W.dtype)


def _edit_out(mod, R: torch.Tensor, coeff: float, U: torch.Tensor = None):
    # project R out of what `mod` writes to the residual. Conv1D weight is [in, out]
    # (transposed vs Linear [out, in]), so the output axis is columns, not rows.
    if _is_conv1d(mod):
        mod.weight.data = _edit_embed(mod.weight.data, R, coeff, U)
    else:
        mod.weight.data = _edit_linear(mod.weight.data, R, coeff, U)
    if getattr(mod, "bias", None) is not None:
        mod.bias.data = _edit_vec(mod.bias.data, R, coeff, U)


def _edit_in(mod, R: torch.Tensor, coeff: float):
    # project R out of `mod`'s input (reader side); input axis flips for Conv1D.
    if _is_conv1d(mod):
        mod.weight.data = _edit_linear(mod.weight.data, R, coeff)
    else:
        mod.weight.data = _edit_embed(mod.weight.data, R, coeff)


def _edit_head(W: torch.Tensor, R: torch.Tensor, coeff: float, U: torch.Tensor = None) -> torch.Tensor:
    # input-side fold (lm_head reads the hidden): (W @ Rbake) @ U.t(), so R=Rbake, outer=U.
    outer = R if U is None else U
    Wf = W.float()
    return (Wf + coeff * ((Wf @ R) @ outer.t())).to(W.dtype)


def _is_packed_writer(mod) -> bool:
    down = getattr(mod, "down_proj", None)
    return isinstance(down, torch.nn.Parameter) and down.dim() == 3


def _edit_writer(mod, R: torch.Tensor, coeff: float, U: torch.Tensor = None):
    # fixed linear op, so per-expert slices compose under the router gates -> packed MoE ok.
    if _is_packed_writer(mod):
        down = mod.down_proj
        edited = [_edit_linear(down.data[i], R, coeff, U) for i in range(down.shape[0])]
        down.data = torch.stack(edited, dim=0)
        return
    _edit_out(mod, R, coeff, U)


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

    n_layers, hidden = model_metadata(model)
    bundle = ModelBundle(model=model, tokenizer=tokenizer, num_layers=n_layers, hidden_size=hidden)
    emb = bundle.embed()
    head = bundle.lm_head()
    layers = bundle.layers()

    print("[bake] applying edits...", flush=True)
    for e in edits:
        R = e["R"].float()
        sign = float(e["sign"])
        if e.get("kind") == "reader":
            # post-norm models: project the (per-layer) direction out of each reader's input columns
            R_layers = e.get("R_layers")
            for L, layer in enumerate(layers):
                a = float(e["layer_alphas"][L])
                if a == 0:
                    continue
                RL = R
                if R_layers is not None and L < len(R_layers) and R_layers[L] is not None:
                    RL = R_layers[L].float()
                for mod in bundle.reader_modules(layer):
                    if isinstance(getattr(mod, "weight", None), torch.Tensor):
                        _edit_in(mod, RL, sign * a)
            continue
        if e.get("kind") == "ple_gate":
            for L, layer in enumerate(layers):
                a = float(e["layer_alphas"][L])
                if a == 0:
                    continue
                for mod in bundle.ple_writers(layer):
                    mod.weight.data = _edit_linear(mod.weight.data, R, sign * a)
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.data = _edit_vec(mod.bias.data, R, sign * a)
            continue
        if e.get("kind") == "ple_residual":
            for L, layer in enumerate(layers):
                a = float(e["layer_alphas"][L])
                if a == 0:
                    continue
                for mod in bundle.ple_projection_writers(layer):
                    _edit_writer(mod, R, sign * a)
            continue
        if e.get("kind") == "ple_embed":
            mod = bundle.ple_embed()
            a = float(e["embed_alpha"])
            if mod is not None and a != 0:
                mod.weight.data = _edit_embed(mod.weight.data, R, sign * a)
            continue
        if e.get("kind") == "ple_model_projection":
            mod = bundle.ple_model_projection()
            a = float(e["embed_alpha"])
            if mod is not None and a != 0:
                mod.weight.data = _edit_linear(mod.weight.data, R, sign * a)
            continue
        if str(e.get("kind", "")).startswith("kv"):
            kind = e.get("kind")
            for L, layer in enumerate(layers):
                a = float(e["layer_alphas"][L])
                if a == 0:
                    continue
                for part, mod in bundle.kv_writers(layer):
                    if kind == "kv_key" and part != "k":
                        continue
                    if kind == "kv_value" and part != "v":
                        continue
                    mod.weight.data = _edit_linear(mod.weight.data, R, sign * a)
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.data = _edit_vec(mod.bias.data, R, sign * a)
            continue
        if e.get("kind") == "query":
            for L, layer in enumerate(layers):
                a = float(e["layer_alphas"][L])
                if a == 0:
                    continue
                for mod in bundle.query_writers(layer):
                    mod.weight.data = _edit_linear(mod.weight.data, R, sign * a)
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.data = _edit_vec(mod.bias.data, R, sign * a)
            continue
        # oblique (mean-preserving) edit: left = Rbake, right co-vector = U. symmetric when absent.
        U = e["U"].float() if e.get("U") is not None else None
        left = e["Rbake"].float() if e.get("Rbake") is not None else R
        # embed rows and the lm-head input are not residual writers; under writers-only they
        # stay symmetric (matching the runtime hook, which skips oblique for those modules).
        writers_only = bool(e.get("oblique_writers_only", False))
        emb_left, emb_U = (R, None) if writers_only else (left, U)
        a_emb = float(e["embed_alpha"])
        if a_emb != 0:
            emb.weight.data = _edit_embed(emb.weight.data, emb_left, sign * a_emb, emb_U)
        a_head = float(e.get("head_alpha", 0.0))
        if a_head != 0 and head is not None:
            head.weight.data = _edit_head(head.weight.data, emb_left, sign * a_head, emb_U)
        for L, layer in enumerate(layers):
            a = float(e["layer_alphas"][L])
            if a == 0:
                continue
            for mod in bundle.layer_writers(layer):
                _edit_writer(mod, left, sign * a, U)

    if drop_layers:
        drop = set(drop_layers)
        keep = [layers[i] for i in range(len(layers)) if i not in drop]
        dec = bundle._decoder()
        if hasattr(dec, "embed_tokens_per_layer"):
            raise ValueError("Layer pruning is not supported for per-layer embeddings.")
        dec.layers = torch.nn.ModuleList(keep)
        section = set_config_value(model.config, "num_hidden_layers", len(keep))
        layer_types = None
        if isinstance(section, dict):
            layer_types = section.get("layer_types")
        else:
            layer_types = getattr(section, "layer_types", None)
        if layer_types is not None and len(layer_types) == len(layers):
            new_types = [layer_types[i] for i in range(len(layers)) if i not in drop]
            if isinstance(section, dict):
                section["layer_types"] = new_types
            else:
                section.layer_types = new_types
        for new_i, layer in enumerate(keep):
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
