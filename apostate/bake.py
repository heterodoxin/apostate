# Bake projections into a standalone checkpoint

from __future__ import annotations

import os
import shutil
from typing import Optional

import torch
from pathlib import Path
from transformers import AutoTokenizer

from .config import ApostateConfig
from .model import ModelBundle, _is_conv1d, _resolve_model_loader, model_metadata, set_config_value

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _post_bake_preservation_metrics(
    actual_delta: torch.Tensor,
    ideal_delta: torch.Tensor,
    preserve_basis: Optional[torch.Tensor],
) -> dict:
    """Measure benign-nullspace leakage after the writer is cast to its storage dtype."""

    if preserve_basis is None:
        return {}
    basis = preserve_basis.to(device=actual_delta.device, dtype=torch.float32)
    if basis.ndim != 2 or basis.shape[0] != actual_delta.shape[-1]:
        raise ValueError(
            "KCRN preserve basis must have shape [writer_input_width, benign_rank]"
        )
    if basis.shape[1] == 0:
        return {
            "post_bake_preservation_error": 0.0,
            "post_bake_preservation_relative": 0.0,
            "ideal_preservation_error": 0.0,
            "serialization_residual": 0.0,
        }
    actual_leak = actual_delta @ basis
    ideal_leak = ideal_delta @ basis
    actual_norm = torch.linalg.norm(actual_delta).clamp_min(1e-8)
    ideal_norm = torch.linalg.norm(ideal_delta).clamp_min(1e-8)
    serialization_residual = torch.linalg.norm(actual_delta - ideal_delta) / ideal_norm
    return {
        "post_bake_preservation_error": float(torch.linalg.norm(actual_leak).item()),
        "post_bake_preservation_relative": float(
            (torch.linalg.norm(actual_leak) / actual_norm).item()
        ),
        "ideal_preservation_error": float(torch.linalg.norm(ideal_leak).item()),
        "serialization_residual": float(serialization_residual.item()),
    }


def _edit_kcrn_writer(
    module,
    left: torch.Tensor,
    right: torch.Tensor,
    preserve_basis: Optional[torch.Tensor] = None,
) -> dict:
    """Add a factorized KCRN update to a Linear, Conv1D, or packed expert writer."""

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        if getattr(weight, "quant_state", None) is not None or not weight.is_floating_point():
            raise TypeError("KCRN requires a floating-point writer weight")
        left = left.to(device=weight.device, dtype=torch.float32)
        right = right.to(device=weight.device, dtype=torch.float32)
        ideal_delta = left @ right
        before = weight.data.float()
        delta = ideal_delta
        if _is_conv1d(module):
            delta = delta.T
        after = (before + delta).to(weight.dtype).float()
        weight.data = after.to(weight.dtype)
        actual_delta = after - before
        if _is_conv1d(module):
            actual_delta = actual_delta.T
        return _post_bake_preservation_metrics(actual_delta, ideal_delta, preserve_basis)
    packed = getattr(module, "down_proj", None)
    if isinstance(packed, torch.Tensor) and packed.ndim == 3:
        if not packed.is_floating_point():
            raise TypeError("KCRN requires floating-point packed expert weights")
        left = left.to(device=packed.device, dtype=torch.float32)
        right = right.to(device=packed.device, dtype=torch.float32)
        ideal_delta = left @ right
        before = packed.data.float()
        after = (before + ideal_delta.unsqueeze(0)).to(packed.dtype).float()
        packed.data = after.to(packed.dtype)
        actual_delta = after - before
        return _post_bake_preservation_metrics(actual_delta, ideal_delta.unsqueeze(0), preserve_basis)
    raise TypeError(f"unsupported KCRN writer type: {type(module).__name__}")


def _kcrn_writer(bundle, edit: dict):
    """Resolve one exported KCRN writer from its layer and writer index."""

    layer_index = int(edit["layer"])
    writer_index = int(edit["writer_index"])
    layers = bundle.layers()
    if not 0 <= layer_index < len(layers):
        raise IndexError(f"KCRN layer index out of range: {layer_index}")
    writers = bundle.layer_writers(layers[layer_index])
    if not 0 <= writer_index < len(writers):
        raise IndexError(f"KCRN writer index out of range: {writer_index} at layer {layer_index}")
    return writers[writer_index]


# R removes while U detects

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
    # Remove R from residual outputs
    if _is_conv1d(mod):
        mod.weight.data = _edit_embed(mod.weight.data, R, coeff, U)
    else:
        mod.weight.data = _edit_linear(mod.weight.data, R, coeff, U)
    if getattr(mod, "bias", None) is not None:
        mod.bias.data = _edit_vec(mod.bias.data, R, coeff, U)


def _edit_in(mod, R: torch.Tensor, coeff: float, U: torch.Tensor = None):
    # Fold reader projection into input weights
    if _is_conv1d(mod):
        mod.weight.data = _edit_linear(mod.weight.data, R, coeff, U)
    else:
        mod.weight.data = _edit_embed(mod.weight.data, R, coeff, U)


def _edit_head(W: torch.Tensor, R: torch.Tensor, coeff: float, U: torch.Tensor = None) -> torch.Tensor:
    # Fold projection into the language head
    outer = R if U is None else U
    Wf = W.float()
    return (Wf + coeff * ((Wf @ R) @ outer.t())).to(W.dtype)


def _is_packed_writer(mod) -> bool:
    down = getattr(mod, "down_proj", None)
    return isinstance(down, torch.nn.Parameter) and down.dim() == 3


def _edit_writer(mod, R: torch.Tensor, coeff: float, U: torch.Tensor = None):
    # Apply the linear edit per packed expert
    if _is_packed_writer(mod):
        down = mod.down_proj
        edited = [_edit_linear(down.data[i], R, coeff, U) for i in range(down.shape[0])]
        down.data = torch.stack(edited, dim=0)
        return
    _edit_out(mod, R, coeff, U)


def _packed_reader_param(mod):
    # Locate packed expert reader weights
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
def bake(
    cfg: ApostateConfig,
    export: dict,
    tokenizer=None,
    drop_layers=None,
    model=None,
    preserve_bases: Optional[dict[tuple[int, int], torch.Tensor]] = None,
    post_bake_metrics: Optional[dict[str, dict]] = None,
) -> str:
    edits = export.get("edits", [])
    if not edits:
        raise ValueError("Nothing to bake: no edits.")
    save_dtype = _DTYPES[cfg.save_dtype]

    if model is None:
        print("[bake] loading model for editing...", flush=True)
        loader = _resolve_model_loader(cfg.model, trust_remote_code=True)
        model = loader.from_pretrained(
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
        if e.get("kind") == "kcrn":
            layer_index = int(e["layer"])
            writer_index = int(e["writer_index"])
            key = (layer_index, writer_index)
            metrics = _edit_kcrn_writer(
                _kcrn_writer(bundle, e),
                e["left"],
                e["right"],
                None if preserve_bases is None else preserve_bases.get(key),
            )
            if post_bake_metrics is not None and metrics:
                post_bake_metrics[f"{layer_index}:{writer_index}"] = {
                    "layer": layer_index,
                    "writer_index": writer_index,
                    **metrics,
                }
            continue
        R = e["R"].float()
        sign = float(e["sign"])
        if e.get("kind") == "reader":
            # Project each post-norm reader input
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
                    # Match runtime reader projection semantics
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
        # Apply an oblique or symmetric writer edit
        U = e["U"].float() if e.get("U") is not None else None
        U_layers = e.get("U_layers")  # Per-layer detector co-vectors
        left = e["Rbake"].float() if e.get("Rbake") is not None else R
        # Keep embeddings and the language head symmetric
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

    # Copy tokenizer and processor files omitted by save_pretrained
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
            continue  # Already saved
        src_file = src_dir / fname
        if not src_file.exists():
            try:  # Resolve Hub files through the cache
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
