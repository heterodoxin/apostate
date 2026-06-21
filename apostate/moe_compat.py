from __future__ import annotations

from typing import List

import torch

from . import model as _model


def _dedupe_modules(mods: List[torch.nn.Module]) -> List[torch.nn.Module]:
    seen, uniq = set(), []
    for mod in mods:
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        uniq.append(mod)
    return uniq


def _iter_expert_modules(experts) -> List[torch.nn.Module]:
    # Returns expert modules from both list-like and packed MoE containers.
    # Packed modules like Qwen3_5MoeExperts are a single torch.nn.Module without len().
    if experts is None:
        return []

    if isinstance(experts, torch.nn.ModuleList):
        return [e for e in experts if isinstance(e, torch.nn.Module)]

    if isinstance(experts, (list, tuple)):
        return [e for e in experts if isinstance(e, torch.nn.Module)]

    try:
        items = list(experts)
    except TypeError:
        items = None

    if items is not None:
        return [e for e in items if isinstance(e, torch.nn.Module)]

    if isinstance(experts, torch.nn.Module):
        return [experts]

    return []


def _shared_expert_modules(mod) -> List[torch.nn.Module]:
    out = []
    for sname in ("shared_expert", "shared_experts"):
        shared = getattr(mod, sname, None)
        out.extend(_iter_expert_modules(shared))
    return _dedupe_modules(out)


def _packed_expert_reader(mod):
    if mod is None:
        return None
    for name in ("gate_proj", "up_proj", "w1", "w3", "gate_up_proj"):
        value = getattr(mod, name, None)
        if isinstance(value, torch.nn.Parameter) and value.dim() == 3:
            return mod
    return None


def _compatible_mlp_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
    mlp = self._mlp(layer)
    out = []

    if mlp is not None:
        packed = self._packed_expert_writer(mlp)
        if packed is not None:
            out.append(packed)

        experts = _iter_expert_modules(getattr(mlp, "experts", None))
        if experts:
            for expert in experts:
                packed = self._packed_expert_writer(expert)
                out.append(packed if packed is not None else self._down_proj(expert))
        else:
            out.append(self._down_proj(mlp))

        for shared in _shared_expert_modules(mlp):
            packed = self._packed_expert_writer(shared)
            out.append(packed if packed is not None else self._down_proj(shared))

    for expert in _iter_expert_modules(getattr(layer, "experts", None)):
        packed = self._packed_expert_writer(expert)
        out.append(packed if packed is not None else self._down_proj(expert))

    for shared in _shared_expert_modules(layer):
        packed = self._packed_expert_writer(shared)
        out.append(packed if packed is not None else self._down_proj(shared))

    return _dedupe_modules([w for w in out if w is not None])


def _compatible_mlp_readers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
    mlp = self._mlp(layer)
    if mlp is None:
        return []

    out = []
    experts = _iter_expert_modules(getattr(mlp, "experts", None))

    if experts:
        for expert in experts:
            packed = _packed_expert_reader(expert)
            if packed is not None:
                out.append(packed)
            out.extend(self._mlp_readers(expert))

        for shared in _shared_expert_modules(mlp):
            packed = _packed_expert_reader(shared)
            if packed is not None:
                out.append(packed)
            out.extend(self._mlp_readers(shared))

        gate = getattr(mlp, "gate", None)  # MoE router
        if isinstance(gate, torch.nn.Module):
            out.append(gate)
    else:
        packed = _packed_expert_reader(mlp)
        if packed is not None:
            out.append(packed)
        out.extend(self._mlp_readers(mlp))

        for shared in _shared_expert_modules(mlp):
            packed = _packed_expert_reader(shared)
            if packed is not None:
                out.append(packed)
            out.extend(self._mlp_readers(shared))

    return _dedupe_modules([m for m in out if isinstance(m, torch.nn.Module)])


def apply_model_bundle_patches() -> None:
    bundle = _model.ModelBundle
    if getattr(bundle, "_packed_moe_compat_patched", False):
        return
    bundle.mlp_writers = _compatible_mlp_writers
    bundle.mlp_readers = _compatible_mlp_readers
    bundle._packed_moe_compat_patched = True
