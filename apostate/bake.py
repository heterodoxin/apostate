# fold the final projection into residual-writer weights and save a standalone checkpoint.

from __future__ import annotations

import os
import shutil
import torch
from pathlib import Path
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


def _edit_in(mod, R: torch.Tensor, coeff: float, U: torch.Tensor = None):
    # input-side fold: W + coeff*(W @ (U or R)) @ R.t(). symmetric removal when U is None.
    # the contrastive reader passes R=detector(D), U=removal(refusal R) so the baked weight
    # matches the pre-hook x - a(x@D)R (detect along D, remove along R). Conv1D flips the axis.
    if _is_conv1d(mod):
        mod.weight.data = _edit_linear(mod.weight.data, R, coeff, U)
    else:
        mod.weight.data = _edit_embed(mod.weight.data, R, coeff, U)


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


def _packed_reader_param(mod):
    # the experts' input-reading packed weight (gate/up), distinct from down_proj (the writer).
    for name in ("gate_up_proj", "gate_proj", "w1"):
        p = getattr(mod, name, None)
        if isinstance(p, torch.nn.Parameter) and p.dim() == 3:
            return p
    return None


def _edit_reader(mod, R: torch.Tensor, coeff: float, U: torch.Tensor = None) -> bool:
    """Input-side reader fold. Plain Linear -> _edit_in. Packed 3D MoE experts -> per-expert
    gate_up_proj slices (each a fixed linear op, so the fold composes under the router gates),
    so the abliteration bakes into packed-MoE experts that aren't editable nn.Linears."""
    if isinstance(getattr(mod, "weight", None), torch.Tensor):
        _edit_in(mod, R, coeff, U)
        return True
    p = _packed_reader_param(mod)
    if p is not None:
        p.data = torch.stack([_edit_embed(p.data[i], R, coeff, U) for i in range(p.shape[0])], dim=0)
        return True
    return False


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
            # post-norm models: project the (per-layer) direction out of each reader's input columns.
            # D_layers (contrastive co-vector) is the removal direction; falls back to RL.
            R_layers = e.get("R_layers")
            D_layers = e.get("D_layers")
            for L, layer in enumerate(layers):
                a = float(e["layer_alphas"][L])
                if a == 0:
                    continue
                RL = R
                if R_layers is not None and L < len(R_layers) and R_layers[L] is not None:
                    RL = R_layers[L].float()
                DL = None
                if D_layers is not None and L < len(D_layers) and D_layers[L] is not None:
                    DL = D_layers[L].float()
                for mod in bundle.reader_modules(layer):
                    # detect along D, remove along R (matches the reader pre-hook); DL absent ->
                    # symmetric removal along R. handles plain Linears AND packed MoE experts.
                    if DL is not None:
                        _edit_reader(mod, DL, sign * a, RL)
                    else:
                        _edit_reader(mod, RL, sign * a)
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
        U_layers = e.get("U_layers")  # per-layer predictive co-vector D_L, indexed by layer
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
            U_L = U
            if U_layers is not None and L < len(U_layers) and U_layers[L] is not None:
                U_L = U_layers[L].float()
            for mod in bundle.layer_writers(layer):
                _edit_writer(mod, left, sign * a, U_L)

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

    # Copy extra files save_pretrained skips: the SentencePiece tokenizer.model (Gemma 1/2/3/3n
    # etc. need it; fast-tokenizer save_pretrained writes only tokenizer.json), plus vision/video
    # processor configs. Resolve from the local dir if present, else the HF cache/hub -- for hub
    # models tok.name_or_path is the repo id (not a path), so the local check alone always missed.
    _extra = [
        "tokenizer.model",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
    ]
    src_dir = Path(tok.name_or_path)
    for fname in _extra:
        dst = Path(cfg.output_dir) / fname
        if dst.exists():
            continue  # save_pretrained already wrote it
        src_file = src_dir / fname
        if not src_file.exists():
            try:  # hub model: fetch from the HF cache (downloads only if not already cached)
                from transformers.utils import cached_file
                resolved = cached_file(
                    cfg.model, fname, _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False)
                src_file = Path(resolved) if resolved else None
            except Exception:
                src_file = None
        if src_file and src_file.exists():
            shutil.copy2(src_file, dst)
            print(f"[bake] copied {fname}", flush=True)

    print("[bake] done", flush=True)
    return cfg.output_dir
